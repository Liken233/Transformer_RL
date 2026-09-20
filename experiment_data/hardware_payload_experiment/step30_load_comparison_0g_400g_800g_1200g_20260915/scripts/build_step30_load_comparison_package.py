from pathlib import Path
import hashlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


RESULTS_ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "step30_load_comparison_0g_400g_800g_1200g_20260915"
PACKAGE_DIR = RESULTS_ROOT / PACKAGE_NAME

EXPERIMENTS = [
    {
        "label": "No load",
        "slug": "0g",
        "mass_g": 0,
        "directory": "step30_directraw_v3_unscaled_20260915_005551",
        "figure": "step30_unscaled_2x2",
        "script": "plot_step30_unscaled_2x2.py",
        "color": "#165DFF",
    },
    {
        "label": "400 g",
        "slug": "400g",
        "mass_g": 400,
        "directory": "step30_directraw_v3_unscaled_20260915_014215",
        "figure": "step30_400g_2x2",
        "script": "plot_step30_400g_2x2.py",
        "color": "#00A870",
    },
    {
        "label": "800 g",
        "slug": "800g",
        "mass_g": 800,
        "directory": "step30_directraw_v3_unscaled_20260915_014631",
        "figure": "step30_800g_2x2",
        "script": "plot_step30_800g_2x2.py",
        "color": "#E67E22",
    },
    {
        "label": "1200 g",
        "slug": "1200g",
        "mass_g": 1200,
        "directory": "step30_directraw_v3_unscaled_20260915_015051",
        "figure": "step30_1200g_2x2",
        "script": "plot_step30_1200g_2x2.py",
        "color": "#D94E5D",
    },
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def panel_label(ax, label):
    ax.text(0.5, -0.25, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=11)


def validate_source(source_dir, experiment):
    required = [
        source_dir / "step30_directraw_v3.csv",
        source_dir / "step30_directraw_v3.mat",
        source_dir / f"{experiment['figure']}.png",
        source_dir / f"{experiment['figure']}.pdf",
        source_dir / experiment["script"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing))


def prepare_experiment(experiment, original_dir, processed_dir,
                       individual_dir, scripts_dir):
    source_dir = RESULTS_ROOT / experiment["directory"]
    validate_source(source_dir, experiment)

    source_csv = source_dir / "step30_directraw_v3.csv"
    source_mat = source_dir / "step30_directraw_v3.mat"
    shutil.copy2(source_csv, original_dir / f"step30_{experiment['slug']}_full.csv")
    shutil.copy2(source_mat, original_dir / f"step30_{experiment['slug']}.mat")
    shutil.copy2(
        source_dir / f"{experiment['figure']}.png",
        individual_dir / f"step30_{experiment['slug']}_2x2.png",
    )
    shutil.copy2(
        source_dir / f"{experiment['figure']}.pdf",
        individual_dir / f"step30_{experiment['slug']}_2x2.pdf",
    )
    shutil.copy2(source_dir / experiment["script"],
                 scripts_dir / f"plot_step30_{experiment['slug']}_2x2.py")

    data = pd.read_csv(source_csv)
    step_rows = np.flatnonzero(
        data["external_target_rad"].to_numpy() > np.deg2rad(15.0)
    )
    if len(step_rows) == 0:
        raise RuntimeError(f"30-deg step not found for {experiment['label']}")

    step_index = int(step_rows[0])
    view = data.iloc[step_index:].copy()
    step_clock = float(view["controller_clock_s"].iloc[0])
    view["time_s"] = view["controller_clock_s"] - step_clock
    view["target_angle_deg"] = np.rad2deg(view["external_target_rad"])
    view["link_angle_deg"] = np.rad2deg(view["q_link_rad"])
    view["motor_angle_deg"] = np.rad2deg(view["theta_joint_rad"])
    view["tracking_error_deg"] = (
        view["target_angle_deg"] - view["link_angle_deg"]
    )

    output_columns = [
        "time_s",
        "controller_clock_s",
        "target_angle_deg",
        "link_angle_deg",
        "dq_link_rad_s",
        "motor_angle_deg",
        "dtheta_joint_rad_s",
        "tracking_error_deg",
        "policy_output_numeric",
        "safe_command_raw",
        "pre_integer_command_raw",
        "pdo_command_int16_raw",
        "actual_torque_raw",
        "fault_code",
        "armed",
    ]
    processed = view[output_columns].rename(columns={
        "policy_output_numeric": "policy_output_raw",
        "dq_link_rad_s": "link_velocity_rad_s",
        "dtheta_joint_rad_s": "motor_velocity_rad_s",
        "pre_integer_command_raw": "pre_integer_command_raw",
        "pdo_command_int16_raw": "pdo_command_raw",
    })
    processed_path = processed_dir / f"step30_{experiment['slug']}_post_step.csv"
    processed.to_csv(processed_path, index=False, float_format="%.12g")

    report = loadmat(source_mat, squeeze_me=True, struct_as_record=False)["report"]
    tail_mask = view["time_s"] >= view["time_s"].max() - 0.3
    final_angle = float(view.loc[tail_mask, "link_angle_deg"].mean())
    target_angle = float(view["target_angle_deg"].max())
    peak_angle = float(view["link_angle_deg"].max())
    tracking_error = view["tracking_error_deg"].to_numpy()

    metrics = {
        "load_label": experiment["label"],
        "load_mass_g": experiment["mass_g"],
        "test_completed": int(getattr(report, "completed")),
        "test_passed": int(getattr(report, "passed")),
        "target_angle_deg": target_angle,
        "final_link_angle_deg": final_angle,
        "peak_link_angle_deg": peak_angle,
        "peak_overshoot_percent": 100.0 * (peak_angle - target_angle) / target_angle,
        "final_tracking_error_deg": target_angle - final_angle,
        "tracking_rmse_deg": float(np.sqrt(np.mean(tracking_error ** 2))),
        "max_abs_policy_output_raw": float(np.abs(view["policy_output_numeric"]).max()),
        "max_abs_safe_command_raw": float(np.abs(view["safe_command_raw"]).max()),
        "max_abs_actual_torque_raw": float(np.abs(view["actual_torque_raw"]).max()),
        "post_step_sample_count": len(view),
        "post_step_duration_s": float(view["time_s"].max()),
        "source_csv_sha256": sha256(source_csv),
        "source_directory": experiment["directory"],
    }
    return processed, metrics


def make_comparison_figure(processed_by_slug, comparison_dir):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    })

    common_end = min(float(frame["time_s"].max())
                     for frame in processed_by_slug.values())
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.11,
                        wspace=0.24, hspace=0.47)
    fig.suptitle("30-deg Step Response under Different Loads", fontsize=13, y=0.98)

    ax = axes[0, 0]
    target_handle, = ax.plot([0, common_end], [30, 30], "k--", linewidth=1.5,
                             label="Target")
    load_handles = []
    for experiment in EXPERIMENTS:
        frame = processed_by_slug[experiment["slug"]]
        mask = frame["time_s"] <= common_end
        handle, = ax.plot(frame.loc[mask, "time_s"],
                          frame.loc[mask, "link_angle_deg"],
                          color=experiment["color"], linewidth=1.35,
                          label=experiment["label"])
        load_handles.append(handle)
    ax.set_title("Trajectory Tracking")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_xlim(0, common_end)
    ax.set_ylim(0, 35)
    ax.grid(True, color="0.82", linewidth=0.6)

    inset = ax.inset_axes([0.50, 0.08, 0.47, 0.39])
    inset.plot([0.5, 2.0], [30, 30], "k--", linewidth=1.0)
    for experiment in EXPERIMENTS:
        frame = processed_by_slug[experiment["slug"]]
        mask = (frame["time_s"] >= 0.5) & (frame["time_s"] <= 2.0)
        inset.plot(frame.loc[mask, "time_s"],
                   frame.loc[mask, "link_angle_deg"],
                   color=experiment["color"], linewidth=1.0)
    inset.set_xlim(0.5, 2.0)
    inset.set_ylim(25, 35)
    inset.set_xticks([0.5, 1.0, 1.5, 2.0])
    inset.set_yticks([25, 30, 35])
    inset.tick_params(labelsize=7)
    inset.grid(True, color="0.86", linewidth=0.45)
    inset.set_title("Zoomed View", fontsize=8, pad=2)
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    ax.axhline(30.0, color="0.25", linestyle="--", linewidth=1.0)
    for experiment in EXPERIMENTS:
        frame = processed_by_slug[experiment["slug"]]
        mask = frame["time_s"] <= common_end
        ax.plot(frame.loc[mask, "time_s"],
                frame.loc[mask, "motor_angle_deg"],
                color=experiment["color"], linewidth=1.35)
    ax.set_title("Motor-Side Angle")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor angle (deg)")
    ax.set_xlim(0, common_end)
    ax.set_ylim(0, 35)
    ax.grid(True, color="0.82", linewidth=0.6)
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    for experiment in EXPERIMENTS:
        frame = processed_by_slug[experiment["slug"]]
        mask = frame["time_s"] <= common_end
        ax.plot(frame.loc[mask, "time_s"],
                frame.loc[mask, "policy_output_raw"],
                color=experiment["color"], linewidth=1.2)
    ax.set_title("Policy Network Output")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Policy output (raw)")
    ax.set_xlim(0, common_end)
    ax.set_ylim(-150, 150)
    ax.grid(True, color="0.82", linewidth=0.6)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    for experiment in EXPERIMENTS:
        frame = processed_by_slug[experiment["slug"]]
        mask = frame["time_s"] <= common_end
        ax.plot(frame.loc[mask, "time_s"],
                frame.loc[mask, "tracking_error_deg"],
                color=experiment["color"], linewidth=1.25)
    ax.set_title("Tracking Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Target - link angle (deg)")
    ax.set_xlim(0, common_end)
    ax.set_ylim(-5, 32)
    ax.grid(True, color="0.82", linewidth=0.6)
    panel_label(ax, "(d)")

    fig.legend([target_handle] + load_handles,
               ["Target"] + [item["label"] for item in EXPERIMENTS],
               loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=5, frameon=True)

    png_path = comparison_dir / "step30_load_comparison_2x2.png"
    pdf_path = comparison_dir / "step30_load_comparison_2x2.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main():
    original_dir = PACKAGE_DIR / "data" / "original"
    processed_dir = PACKAGE_DIR / "data" / "processed"
    individual_dir = PACKAGE_DIR / "figures" / "individual"
    comparison_dir = PACKAGE_DIR / "figures" / "comparison"
    scripts_dir = PACKAGE_DIR / "scripts"
    for directory in [original_dir, processed_dir, individual_dir,
                      comparison_dir, scripts_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    processed_by_slug = {}
    metrics = []
    for experiment in EXPERIMENTS:
        processed, row = prepare_experiment(
            experiment, original_dir, processed_dir, individual_dir, scripts_dir
        )
        processed_by_slug[experiment["slug"]] = processed
        metrics.append(row)

    summary_path = PACKAGE_DIR / "data" / "summary_metrics.csv"
    pd.DataFrame(metrics).to_csv(summary_path, index=False, float_format="%.12g")
    png_path, pdf_path = make_comparison_figure(processed_by_slug, comparison_dir)

    readme = """30-deg step load comparison package

Experiments
- No load: 2026-09-15 00:55:51
- 400 g: 2026-09-15 01:42:15
- 800 g: 2026-09-15 01:46:31
- 1200 g: 2026-09-15 01:50:51

Common conditions
- External target: 30 deg
- Policy target scale: 1.0
- Policy output: direct raw command
- Timing: 1 s zero-target hold followed by 4 s of step control
- Processed time origin: the first recorded 30-deg target sample
- Processed data use recorded samples directly; no interpolation was applied
- The no-load run completed and saved valid data, but its source MAT report records passed=0 for the combined gate; the other three runs record passed=1

Package contents
- data/original: complete source CSV files and MATLAB MAT files
- data/processed: post-step CSV files with time reset to 0 s; policy_target omitted
- data/summary_metrics.csv: comparable result metrics and source checksums
- figures/individual: original paper-format figures for each load
- figures/comparison: unified 2x2 comparison in PNG and PDF
- scripts: plotting scripts used for the individual and comparison figures

Comparison figure
- (a) Link trajectory tracking for all loads, with an inset covering x=0.5-2.0 s and y=25-35 deg
- (b) Motor-side angle
- (c) Policy network output in raw units
- (d) Tracking error, defined as target angle minus link angle
"""
    (PACKAGE_DIR / "README.txt").write_text(readme, encoding="utf-8")
    shutil.copy2(Path(__file__), scripts_dir / Path(__file__).name)

    zip_path = RESULTS_ROOT / f"{PACKAGE_NAME}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip",
                        root_dir=PACKAGE_DIR.parent, base_dir=PACKAGE_DIR.name)

    print(PACKAGE_DIR)
    print(png_path)
    print(pdf_path)
    print(summary_path)
    print(zip_path)


if __name__ == "__main__":
    main()
