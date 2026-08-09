#!/usr/bin/env python3
"""
真实柔性关节机械臂 - 基于Simulink模型的物理动力学
Simulink模型包含：
  1. 电流环一阶动态: gm/(s+gm)
  2. 电机动力学: Jm·ddth + Bm·dth + cm·sign(dth) = tau_m - tau_flex
  3. 连杆动力学: Ja·ddq + Ba·dq + cnta·sign(dq) = tau_flex - tau_gravity
  4. 柔性耦合: tau_flex = Kf * (th - q)
  5. 重力矩: tau_gravity = mglx*sin(q) + mgly*cos(q)
"""

import time
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class RealFlexArmTrackingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 render_mode=None,
                 dt=0.001,
                 max_steps=5000,
                 target_type='sine',
                 target_params=None,
                 real_time_factor=2.0):

        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.max_steps = max_steps
        self.target_type = target_type

        if target_params is None:
            target_params = {}

        self.fixed_target = target_params.get('fixed_angle', 0.0)

        self.sine_amp = np.radians(target_params.get('sine_amplitude', 45.0))
        self.sine_freq = target_params.get('sine_frequency', 2.0)
        self.sine_offset = np.radians(target_params.get('sine_offset', 0.0))

        self.amp_range = (self.sine_amp - np.radians(45.0), self.sine_amp + np.radians(45.0))
        self.freq_range = (self.sine_freq * 0.8, self.sine_freq * 1.2)
        self.offset_range = (self.sine_offset - np.radians(5.0), self.sine_offset + np.radians(5.0))

        self._sample_target_params()

        self.current_target_params = {
            'sine_amplitude': np.degrees(self.sine_amp),
            'sine_frequency': self.sine_freq,
            'sine_offset': np.degrees(self.sine_offset)
        }

        self.random_min = np.radians(target_params.get('random_min', -60.0))
        self.random_max = np.radians(target_params.get('random_max', 60.0))
        self.random_hold_steps = target_params.get('random_hold_steps', 1000)

        # ===== 新增：阶跃响应参数 =====
        self.step_final_target = target_params.get('final_target', np.radians(45.0))
        self.step_time = target_params.get('step_time', 2.0)          # 秒
        self.initial_q = target_params.get('initial_q', 0.0)
        self.initial_dq = target_params.get('initial_dq', 0.0)
        self.initial_th = target_params.get('initial_th', 0.0)
        self.initial_dth = target_params.get('initial_dth', 0.0)
        self.use_initial_state = target_params.get('use_initial_state', False)
        # 用于记录阶跃切换是否已经发生（避免重复触发）
        self._step_triggered = False

        # ========== Simulink物理参数 ==========
        # 电机侧参数
        self.Jm = 0.2935
        self.Bm = 4.0816
        self.cm = 3.1480  # 电机库仑摩擦
        self.cntm = 0.0357  # 电机静摩擦（Stribeck）

        # 连杆侧参数
        self.Ja = 0.3451
        self.Ba = 0.1616
        self.mglx = 8.856  # 重力矩主要分量
        self.mgly = 0.0893  # 重力矩次要分量
        self.cnta = 0.0421  # 连杆静摩擦

        # 耦合与驱动参数
        self.Kt = 0.094  # 转矩常数
        self.Kf = 2.19 * 180 / np.pi  # 125.4778 柔性刚度

        # 控制器增益（用于参考，环境本身不直接使用）
        self.wa = np.sqrt(self.Kf / self.Ja)
        self.K = np.sqrt(5)
        self.wm = self.K * self.wa
        self.Kr = 4.0 / self.Ja
        self.Kp = self.wa ** 2
        self.Kv = 4.0 * self.wa

        # 电流环参数（Subsystem2核心）
        self.gm = 300.0  # 电流环带宽
        self.alpha = 1.0 - np.exp(-self.gm * self.dt)  # 一阶离散化系数

        # 重力补偿与观测器
        self.G0 = 1.0
        self.xigema = 1.0
        self.gv = 700.0  # 速度观测器增益

        # 扰动观测器参数
        self.wn = 20.0
        self.L1 = self.Ja * self.wn ** 2 / self.Kf - 1.0
        self.L2 = 2.0 * self.Ja * self.xigema * self.wn / self.Kf - self.Ba / self.Kf
        self.M = self.Ja * self.wn ** 2 / self.Kf

        # LQR权重（参考）
        self.q1 = 1.0
        self.q2 = 1.0
        self.p2 = self.q1 / (2.0 * self.wn ** 2)
        self.p3 = (2.0 * self.p2 + self.q2) / (4.0 * self.xigema * self.wn)
        self.p1 = 2.0 * self.xigema * self.wn * self.p2 + self.wn * self.p3

        self.current_max = 250.0
        self.action_space = spaces.Box(low=-self.current_max, high=self.current_max,
                                       shape=(1,), dtype=np.float32)

        obs_low = np.array([-np.pi, -10.0, -2 * np.pi, -20.0, -2 * np.pi], dtype=np.float32)
        obs_high = np.array([np.pi, 10.0, 2 * np.pi, 20.0, 2 * np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 状态变量
        self.q = self.dq = self.th = self.dth = 0.0
        self.last_current = 0.0  # 上一时刻电流（用于电流环离散化）
        self.last_dth = 0.0  # 上一时刻电机速度（用于反电动势）
        self.q_target = 0.0
        self.current_step = 0
        self.physics_client = None
        self.random_counter = 0
        self.real_time_factor = real_time_factor
        self._last_step_time = None

        self._best_error = None
        self._precision_threshold_deg = 0.1

    def _sample_target_params(self):
        self.sine_amp = np.random.uniform(*self.amp_range)
        self.sine_freq = np.random.uniform(*self.freq_range)
        self.sine_offset = np.random.uniform(*self.offset_range)

    def _create_scene(self):
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setTimeStep(self.dt)

        # base
        base_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=1.0, height=0.01)
        base_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=1.0, length=0.01,
                                          rgbaColor=[0.95, 0.95, 0.95, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_shape,
                          baseVisualShapeIndex=base_visual, basePosition=[0, 0, 0])

        # angle markers
        for angle_deg in range(-180, 181, 30):
            angle_rad = np.radians(angle_deg + 90)
            r = 0.8
            x = r * np.cos(angle_rad)
            y = r * np.sin(angle_rad)
            color = [1, 0, 0] if angle_deg == 0 else [0.3, 0.3, 0.3]
            size = 0.04 if angle_deg == 0 else 0.025
            mark = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=[*color, 0.8])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=mark, basePosition=[x, y, 0.02])
            if angle_deg % 90 == 0:
                p.addUserDebugText(f"{angle_deg}", [x * 1.15, y * 1.15, 0.03],
                                   textColorRGB=[0, 0, 0], textSize=2)

        p.addUserDebugLine([0, 0, 0.03], [0, 0.9, 0.03], [1, 0, 0], 4)
        p.addUserDebugLine([0, 0, 0.03], [0.9, 0, 0.03], [0, 0.7, 0], 3)
        p.addUserDebugLine([0, 0, 0.03], [-0.9, 0, 0.03], [0, 0.7, 0], 3)
        p.addUserDebugText("0 deg (UP)", [0, 0.95, 0.03], [1, 0, 0], 2)

        p.addUserDebugLine([0, 0, 0.05], [0, -0.6, 0.05], [0, 0, 0.8], 4)
        p.addUserDebugText("g", [0, -0.7, 0.05], [0, 0, 0.8], 2)

        # link
        link_len = 0.7
        link_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, link_len / 2, 0.03])
        link_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04, link_len / 2, 0.03],
                                          rgbaColor=[0.2, 0.5, 0.9, 1])
        self.link_id = p.createMultiBody(baseMass=self.Ja * 3,
                                         baseCollisionShapeIndex=link_shape,
                                         baseVisualShapeIndex=link_visual,
                                         basePosition=[0, link_len / 2, 0.1])

        # motor
        motor_len = 0.5
        motor_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, motor_len / 2, 0.02])
        motor_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, motor_len / 2, 0.02],
                                           rgbaColor=[1.0, 0.6, 0.2, 0.9])
        self.motor_id = p.createMultiBody(baseMass=0.1,
                                          baseCollisionShapeIndex=motor_shape,
                                          baseVisualShapeIndex=motor_visual,
                                          basePosition=[0, motor_len / 2, 0.15])

        joint_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=0.1,
                                        rgbaColor=[1, 0.9, 0, 1])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=joint_vis, basePosition=[0, 0, 0.1])

        target_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[0, 1, 0, 0.6])
        self.target_visual = p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_vis,
                                               basePosition=[0, 0.5, 0.1])
        p.resetDebugVisualizerCamera(1.5, 0, -89, [0, 0, 0])

    def _angle_to_visual(self, angle, length):
        visual_angle = np.pi / 2 - angle
        cx = (length / 2) * np.cos(visual_angle)
        cy = (length / 2) * np.sin(visual_angle)
        orn = p.getQuaternionFromEuler([0, 0, -angle])
        return [cx, cy, 0.1], orn

    def _update_target(self):
        t = self.current_step * self.dt
        if self.target_type == 'fixed':
            self.q_target = self.fixed_target
        elif self.target_type == 'sine':
            self.q_target = (self.sine_amp * np.sin(2 * np.pi * self.sine_freq * t)
                             + self.sine_offset)
        elif self.target_type == 'random':
            if self.current_step % self.random_hold_steps == 0:
                self.q_target = self.np_random.uniform(self.random_min, self.random_max)
        # ===== 新增：阶跃响应目标更新 =====
        elif self.target_type == 'step':
            if t >= self.step_time and not self._step_triggered:
                self.q_target = self.step_final_target
                self._step_triggered = True

    def _update_visuals(self):
        pos, orn = self._angle_to_visual(self.q, 0.7)
        p.resetBasePositionAndOrientation(self.link_id, pos, orn)

        mpos, morn = self._angle_to_visual(self.th, 0.5)
        mpos[2] = 0.15
        p.resetBasePositionAndOrientation(self.motor_id, mpos, morn)

        tpos, _ = self._angle_to_visual(self.q_target, 1.5)
        p.resetBasePositionAndOrientation(self.target_visual, tpos, [0, 0, 0, 1])

        if self.current_step % 10 == 0:
            err = self.q - self.q_target
            p.addUserDebugText(
                f"Step {self.current_step} | q={np.degrees(self.q):.1f} | "
                f"target={np.degrees(self.q_target):.1f} | err={np.degrees(err):.1f}",
                [0, -1.1, 0.05], [0, 0, 0], 1.5, replaceItemUniqueId=1
            )

    # ========== 核心：基于Simulink的动力学模型 ==========
    def _dynamics(self, q, dq, th, dth, current, last_current, last_dth):
        # --- 柔性耦合力矩 ---
        tau_flex = self.Kf * (th - q)

        # --- 重力矩（连杆侧）---
        tau_gravity = self.mglx * np.sin(q) * self.G0

        # --- 反电动势 / 速度反馈 ---
        tau_dth = dth * self.gm * self.Jm

        # --- 电流处理（目标角度->实际电流） ---
        current_improve = current * self.Kp + (tau_gravity - tau_flex) * self.Kr - dth * self.Kv

        # --- 电流环动态（Subsystem2: gm/(s+gm)）---
        # 输入电流 = 控制器输出 + 上一时刻电流（积分效果）
        input_current = current * self.Jm / self.Kt + (last_current - last_dth * self.gm * self.Jm) / self.Kt

        # 一阶低通滤波离散化
        next_current = (self.alpha * (input_current * self.Kt + tau_dth) +(1.0 - self.alpha) * last_current)

        # 电机输出转矩（经过电流限幅）
        clip_current = input_current * self.Kt

        # --- 电机侧动力学 ---
        ddth = (clip_current - tau_flex) / self.Jm

        # --- 连杆侧动力学 ---
        link_friction = self.Ba * q
        ddq = (tau_flex - tau_gravity - link_friction) / self.Ja

        return ddq, ddth, next_current

    def _check_target_change(self):
        if self.target_type == 'random':
            if self.current_step % self.random_hold_steps == 0:
                return True
        return False

    def _compute_reward(self):
        """计算基础奖励（不含边界惩罚）"""
        angle_error_deg = np.degrees(abs(self.q - self.q_target))

        # 1. 高斯跟踪奖励：峰值1.0，宽度约5°
        tracking_reward = np.exp(-0.5 * (angle_error_deg / 5.0) ** 2)

        # 2. 精度奖励：误差<0.5°时线性增加，上限0.2
        if angle_error_deg < 0.5:
            precision_reward = 0.2 * (1.0 - angle_error_deg / 0.5)
        else:
            precision_reward = 0.0

        # 3. 柔性/速度小惩罚
        flex_penalty = 0.02 * min(abs(self.th - self.q) / np.radians(30.0), 1.0)
        velocity_penalty = 0.005 * ((self.dq / 10.0) ** 2 + (self.dth / 20.0) ** 2)

        reward = tracking_reward

        if self._best_error is None:
            self._best_error = angle_error_deg
        else:
            self._best_error = min(self._best_error, angle_error_deg)

        return float(reward), {
            'tracking_reward': float(tracking_reward),
            'precision_reward': float(precision_reward),
            'flex_penalty': float(-flex_penalty),
            'velocity_penalty': float(-velocity_penalty),
            'best_error': float(self._best_error),
            'in_precision_zone': angle_error_deg < self._precision_threshold_deg,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.physics_client is not None:
            try:
                p.disconnect()
            except:
                pass
        self._create_scene()
        self._sample_target_params()
        self.current_target_params = {
            'sine_amplitude': np.degrees(self.sine_amp),
            'sine_frequency': self.sine_freq,
            'sine_offset': np.degrees(self.sine_offset)
        }
        self.random_counter = 0
        self._step_triggered = False   # 重置阶跃标志

        # ===== 新增：根据 target_type 初始化状态 =====
        if self.target_type == 'step':
            # 使用自定义初始状态（若 use_initial_state=True 则用用户参数，否则用随机）
            if self.use_initial_state:
                self.q = self.initial_q
                self.dq = self.initial_dq
                self.th = self.initial_th
                self.dth = self.initial_dth
            else:
                self.q = 0.0 + self.np_random.uniform(-0.1, 0.1)
                self.dq = 0.0
                self.th = 0.0 + self.np_random.uniform(-0.05, 0.05)
                self.dth = 0.0
            # 初始目标 = 初始 q（实现无误差起始）
            self.q_target = self.q
        else:
            # 原有随机初始化逻辑（保持兼容）
            self.q = 0.0 + self.np_random.uniform(-0.1, 0.1)
            self.dq = 0.0
            self.th = 0.0 + self.np_random.uniform(-0.05, 0.05)
            self.dth = 0.0
            if self.target_type == 'random':
                self.q_target = self.np_random.uniform(self.random_min, self.random_max)
            else:
                self._update_target()

        self.last_current = 0.0
        self.last_dth = 0.0
        self.current_step = 0
        self._update_visuals()

        initial_error_deg = np.degrees(abs(self.q - self.q_target))
        self._best_error = initial_error_deg
        obs = np.array([self.q, self.dq, self.th, self.dth, self.q_target], dtype=np.float32)
        self._last_step_time = self.current_step
        return obs, {'target': self.q_target, 'initial_error': initial_error_deg}

    def step(self, action):
        current = float(np.clip(action[0], -self.current_max, self.current_max))

        # 调用Simulink模型动力学
        ddq, ddth, self.last_current = self._dynamics(
            self.q, self.dq, self.th, self.dth,
            current, self.last_current, self.last_dth
        )

        # 状态积分（前向欧拉）
        self.dq += ddq * self.dt
        self.q += self.dq * self.dt
        self.last_dth = self.dth
        self.dth += ddth * self.dt
        self.th += self.dth * self.dt

        self.current_step += 1
        self._update_target()
        if self._check_target_change():
            new_error_deg = np.degrees(abs(self.q - self.q_target))
            self._best_error = new_error_deg

        self._update_visuals()
        p.stepSimulation()

        # 基础奖励
        reward, reward_info = self._compute_reward()

        # 硬边界处理
        terminated = False
        boundary_penalty = 0.0
        if abs(self.q) > np.pi:
            boundary_penalty = - (500 * (1 - self.current_step / self.max_steps))
            reward += boundary_penalty
            terminated = True

        truncated = self.current_step >= self.max_steps
        obs = np.array([self.q, self.dq, self.th, self.dth, self.q_target], dtype=np.float32)
        angle_error_deg = np.degrees(abs(self.q - self.q_target))

        info = {
            'current_A': current,
            'target_deg': np.degrees(self.q_target),
            'error_deg': angle_error_deg,
            'tracking_error': np.radians(angle_error_deg),
            'sine_amp': self.sine_amp,
            'sine_freq': self.sine_freq,
            'sine_offset': self.sine_offset,
            'reward_tracking': reward_info['tracking_reward'],
            'reward_precision': reward_info['precision_reward'],
            'reward_boundary': boundary_penalty,
            'reward_flex': reward_info['flex_penalty'],
            'reward_velocity': reward_info['velocity_penalty'],
            'reward_total': float(reward),
            'best_error_deg': reward_info['best_error'],
            'in_precision_zone': reward_info['in_precision_zone'],
            # 额外调试信息
            'tau_flex': self.Kf * (self.th - self.q),
            'tau_gravity': self.mglx * np.sin(self.q) + self.mgly * np.cos(self.q),
        }

        if self.render_mode == "human" and self.real_time_factor > 0:
            if self._last_step_time is not None:
                expected_step_duration = self.dt / self.real_time_factor
                if expected_step_duration * (self.current_step - self._last_step_time) > 0.02:
                    time.sleep(0.02)
                    self._last_step_time = self.current_step

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect()
            except:
                pass


if __name__ == "__main__":
    print("=" * 60)
    print("Simulink模型物理动力学测试（含阶跃响应）")
    print("=" * 60)
    # 阶跃参数示例
    params = {
        'final_target': np.radians(45.0),
        'step_time': 2.0,
        'initial_q': 0.0,
        'use_initial_state': True
    }
    env = RealFlexArmTrackingEnv(render_mode="human", target_type='random',
                                 target_params=params, max_steps=50000, real_time_factor=1)
    obs, info = env.reset()
    print(f"初始目标: {info['target']:.2f} rad ({np.degrees(info['target']):.1f}度)")
    print(f"将在 {params['step_time']} 秒后跳转到 {np.degrees(params['final_target']):.1f} 度")

    for i in range(50000):
        # 简单PD控制测试
        action = np.array([0])#.5 * (env.q_target - env.q)
        obs, reward, done, truncated, info = env.step(action)
        if i % 500 == 0 or done or truncated:
            print(f"Step {i:4d}: q={np.degrees(obs[0]):5.1f}°, "
                  f"th={np.degrees(obs[2]):5.1f}°, "
                  f"target={np.degrees(obs[4]):5.1f}°, "
                  f"err={info['error_deg']:5.1f}°, "
                  f"reward={reward:6.3f}")
        if done or truncated:
            obs, info = env.reset()
    env.close()