#!/usr/bin/env python3
"""
真实柔性关节机械臂 - 纯动力学计算（无PyBullet渲染）
基于Simulink模型的物理动力学，剥离所有可视化，专注训练速度
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RealFlexArmTrackingEnv(gym.Env):
    """
    纯计算环境，无渲染开销。
    如需可视化，请使用训练好的策略在外部脚本中回放。
    """

    def __init__(self,
                 dt=0.001,
                 max_steps=5000,
                 target_type='sine',
                 target_params=None,
                 render_mode=None,
                 real_time_factor=0.0):

        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.target_type = target_type

        if target_params is None:
            target_params = {}

        self.fixed_target = target_params.get('fixed_angle', 0.0)

        # 目标参数（支持在范围内随机采样）
        self.sine_amp = np.radians(target_params.get('sine_amplitude', 15.0))
        self.sine_freq = target_params.get('sine_frequency', 2.0)
        self.sine_offset = np.radians(target_params.get('sine_offset', 0.0))

        self.amp_range = (self.sine_amp - np.radians(15.0), self.sine_amp + np.radians(15.0))
        self.freq_range = (self.sine_freq * 0.8, self.sine_freq * 1.2)
        self.offset_range = (self.sine_offset - np.radians(5.0), self.sine_offset + np.radians(5.0))

        self.random_min = np.radians(target_params.get('random_min', -60.0))
        self.random_max = np.radians(target_params.get('random_max', 60.0))
        self.random_hold_steps = target_params.get('random_hold_steps', 1000)

        # ========== Simulink物理参数 ==========
        self.Jm = 0.2935
        self.Bm = 4.0816
        self.cm = 3.1480
        self.cntm = 0.0357

        self.Ja = 0.3451
        self.Ba = 0.1616
        self.mglx = 8.856
        self.mgly = 0.0893
        self.cnta = 0.0421

        self.Kt = 0.094
        self.Kf = 2.19 * 180 / np.pi  # 125.4778

        # 控制器增益（参考）
        self.wa = np.sqrt(self.Kf / self.Ja)
        self.K = np.sqrt(5)
        self.wm = self.K * self.wa
        self.Kr = 4.0 / self.Ja
        self.Kp = self.wa ** 2
        self.Kv = 4.0 * self.wa

        # 电流环参数
        self.gm = 300.0
        self.alpha = 1.0 - np.exp(-self.gm * self.dt)

        # 重力补偿与观测器
        self.G0 = 1.0
        self.xigema = 1.0
        self.gv = 700.0

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

        self.current_max = 500.0
        self.action_space = spaces.Box(
            low=-self.current_max, high=self.current_max,
            shape=(1,), dtype=np.float32
        )

        obs_low = np.array([-np.pi, -10.0, -2 * np.pi, -20.0, -2 * np.pi], dtype=np.float32)
        obs_high = np.array([np.pi, 10.0, 2 * np.pi, 20.0, 2 * np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 状态变量
        self.q = self.dq = self.th = self.dth = 0.0
        self.last_current = 0.0
        self.last_dth = 0.0
        self.q_target = 0.0
        self.current_step = 0
        self.random_counter = 0

        self._best_error = None
        # 六个精度阈值（度）
        self._precision_threshold_deg = [
            np.radians(0.1),
            np.radians(0.5),
            np.radians(1.0),
            np.radians(5.0),
            np.radians(10.0),
            np.radians(30.0)
        ]
        self._threshold_names = ['0.1', '0.5', '1', '5', '10', '30']

        # ========== 精度计数相关 ==========
        self.precision_count = 0
        self.reach_count = 0
        self.precision_threshold = 5000
        self.precision_bonus = 0.0

    def _sample_target_params(self):
        self.sine_amp = np.random.uniform(*self.amp_range)
        self.sine_freq = np.random.uniform(*self.freq_range)
        self.sine_offset = np.random.uniform(*self.offset_range)

    # ========== 核心：基于Simulink的动力学模型 ==========
    def _dynamics(self, q, dq, th, dth, current, last_current, last_dth):
        tau_flex = self.Kf * (th - q)
        tau_gravity = self.mglx * np.sin(q) * self.G0
        tau_dth = dth * self.gm * self.Jm

        current_improve = current * self.Kp + (tau_gravity - tau_flex) * self.Kr - dth * self.Kv

        input_current = current * self.Jm / self.Kt + (last_current - last_dth * self.gm * self.Jm) / self.Kt
        next_current = (self.alpha * (input_current * self.Kt + tau_dth) + (1.0 - self.alpha) * last_current)

        clip_current = input_current * self.Kt
        ddth = (clip_current - tau_flex) / self.Jm

        link_friction = self.Ba * q
        ddq = (tau_flex - tau_gravity - link_friction) / self.Ja

        return ddq, ddth, next_current

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

    def _check_target_change(self):
        if self.target_type == 'random':
            if self.current_step % self.random_hold_steps == 0:
                return True
        return False

    def _compute_reward(self, ddq, ddth,current):
        angle_error = abs(self.q - self.q_target)

        tracking_reward = angle_error / 60.0

        precision_reward = 0.5 * pow(np.e, -angle_error / np.radians(10)) * (0.05 * self.reach_count + 1)

        flex_penalty = 0.2 * min(abs(current / 500), 1.0)
        accelerate_penalty = 0.002 * (abs(ddq / 1.0))
        velocity_penalty = 0.01 * abs(0.5 * (self.q - self.q_target) - self.dq)

        reward = precision_reward - flex_penalty

        if self._best_error is None:
            self._best_error = angle_error
        else:
            self._best_error = min(self._best_error, angle_error)

        # 生成六个精度标志
        precision_flags = {}
        for name, th in zip(self._threshold_names, self._precision_threshold_deg):
            precision_flags[f'in_precision_{name}'] = angle_error < th

        return float(reward), {
            'tracking_reward': float(-tracking_reward),
            'precision_reward': float(precision_reward),
            'accelerate_penalty': float(-accelerate_penalty),
            'flex_penalty': float(-flex_penalty),
            'velocity_penalty': float(-velocity_penalty),
            'best_error': float(self._best_error),
            **precision_flags
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._sample_target_params()
        self.current_target_params = {
            'sine_amplitude': np.degrees(self.sine_amp),
            'sine_frequency': self.sine_freq,
            'sine_offset': np.degrees(self.sine_offset)
        }
        self.random_counter = 0

        if self.target_type == 'random':
            self.q_target = self.np_random.uniform(self.random_min, self.random_max)
        else:
            self._update_target()

        self.q = 0.0 + self.np_random.uniform(-0.1, 0.1)
        self.dq = 0.0
        self.th = 0.0 + self.np_random.uniform(-0.05, 0.05)
        self.dth = 0.0
        self.last_current = 0.0
        self.last_dth = 0.0
        self.current_step = 0

        initial_error_deg = np.degrees(abs(self.q - self.q_target))
        self._best_error = initial_error_deg
        self.precision_count = 0
        self.reach_count = 0

        obs = np.array([self.q, self.dq, self.th, self.dth, self.q_target], dtype=np.float32)
        return obs, {'target': self.q_target, 'initial_error': initial_error_deg}

    def step(self, action):
        current = float(np.clip(action[0], -self.current_max, self.current_max))

        ddq, ddth, self.last_current = self._dynamics(
            self.q, self.dq, self.th, self.dth,
            current, self.last_current, self.last_dth
        )

        self.dq += ddq * self.dt
        self.q += self.dq * self.dt
        self.last_dth = self.dth
        self.dth += ddth * self.dt
        self.th += self.dth * self.dt

        self.current_step += 1
        self._update_target()

        # 常规目标切换（重置精度计数）
        if self._check_target_change():
            self._best_error = np.degrees(abs(self.q - self.q_target))
            self.precision_count = 0

        reward, reward_info = self._compute_reward(ddq, ddth, current)

        # ===== 精度计数与奖励触发 =====
        # bonus_triggered = False
        # if self.target_type == 'random':
        #     deg_error = np.degrees(abs(self.q - self.q_target))
        #     if deg_error < 0.1:
        #         self.precision_count += 5
        #     elif deg_error < 0.5:
        #         self.precision_count += 2
        #     elif deg_error < 1.0:
        #         self.precision_count += 1
        #     else:
        #         self.precision_count = max(self.precision_count - 10 , 0)
        #
        #     if self.precision_count >= self.precision_threshold:
        #         reward += self.precision_bonus
        #         bonus_triggered = True
        #         self.q_target = self.np_random.uniform(self.random_min, self.random_max)
        #         self.precision_count = 0
        #         self.reach_count += 1
        #         self._best_error = np.degrees(abs(self.q - self.q_target))

        terminated = False
        boundary_penalty = 0.0
        if abs(self.q) > np.pi:
            boundary_penalty = 0
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
            'reward_accelerate': reward_info['accelerate_penalty'],
            'reward_total': float(reward),
            'best_error_deg': reward_info['best_error'],
            # 六个精度标志
            'in_precision_zone_0.1': reward_info['in_precision_0.1'],
            'in_precision_zone_0.5': reward_info['in_precision_0.5'],
            'in_precision_zone_1': reward_info['in_precision_1'],
            'in_precision_zone_5': reward_info['in_precision_5'],
            'in_precision_zone_10': reward_info['in_precision_10'],
            'in_precision_zone_30': reward_info['in_precision_30'],
            'tau_flex': self.Kf * (self.th - self.q),
            'tau_gravity': self.mglx * np.sin(self.q) + self.mgly * np.cos(self.q),
            'precision_count': self.precision_count,
            'precision_bonus_triggered': 0#bonus_triggered,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("Simulink模型物理动力学测试（无渲染）")
    print("=" * 60)
    params = {'sine_amplitude': 45.0, 'sine_frequency': 0.5, 'sine_offset': 0.0}
    env = RealFlexArmTrackingEnv(target_type='sine', target_params=params, max_steps=50000)
    obs, info = env.reset()
    print(f"初始目标: {info['target']:.2f} rad ({np.degrees(info['target']):.1f}度)")
    print(f"wa={env.wa:.4f}, Kp={env.Kp:.4f}, Kv={env.Kv:.4f}, Kr={env.Kr:.4f}")
    print(f"gm={env.gm}, alpha={env.alpha:.6f}")

    import time
    start = time.time()
    total_reward = 0.0

    for i in range(50000):
        action = np.array([0.5 * (env.q_target - env.q)])
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if i % 5000 == 0 or done or truncated:
            print(f"Step {i:5d}: q={np.degrees(obs[0]):6.1f}°, "
                  f"th={np.degrees(obs[2]):6.1f}°, "
                  f"err={info['error_deg']:6.1f}°, "
                  f"reward={reward:7.3f}, "
                  f"flex={info['tau_flex']:7.2f}, "
                  f"grav={info['tau_gravity']:7.2f}, "
                  f"prec_cnt={info['precision_count']}")
        if done or truncated:
            print(f"  -> Episode结束, 总奖励={total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0

    elapsed = time.time() - start
    print(f"\n50000步耗时: {elapsed:.3f}s, 平均 {50000/elapsed:.0f} steps/s")
    env.close()