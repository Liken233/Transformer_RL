from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIR = Path(__file__).resolve().parent
CSV_PATH = RESULT_DIR / "step30_directraw_v3.csv"
PNG_PATH = RESULT_DIR / "step30_unscaled_2x2.png"
PDF_PATH = RESULT_DIR / "step30_unscaled_2x2.pdf"


def add_panel_label(ax, label):
    ax.text(
        0.5,
        -0.25,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )


data = pd.read_csv(CSV_PATH)
step_rows = data.index[data["external_target_rad"] > np.deg2rad(15)]
if len(step_rows) == 0:
    raise RuntimeError("The 30-degree step was not found in the CSV data.")

step_index = int(step_rows[0])
step_clock = float(data.loc[step_index, "controller_clock_s"])
view = data.loc[step_index:].copy()
view["time_s"] = view["controller_clock_s"] - step_clock
view["target_deg"] = np.rad2deg(view["external_target_rad"])
view["link_deg"] = np.rad2deg(view["q_link_rad"])

delta_clock_ms = view["controller_clock_s"].diff() * 1000.0
delta_heartbeat = view["heartbeat"].diff()
view["policy_update_ms"] = delta_clock_ms / delta_heartbeat
valid_period = (
    np.isfinite(view["policy_update_ms"])
    & (delta_heartbeat > 0)
    & (view["policy_update_ms"] > 0)
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    }
)

fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=False)
fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.12, wspace=0.25, hspace=0.48)

ax = axes[0, 0]
ax.plot(view["time_s"], view["target_deg"], "k--", linewidth=1.5, label="Target")
ax.plot(view["time_s"], view["link_deg"], color="#165DFF", linewidth=1.5, label="Link angle")
ax.set_title("Trajectory Tracking")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (deg)")
ax.set_xlim(0, view["time_s"].max())
ax.set_ylim(0, 35)
ax.legend(loc="lower right", frameon=True)
ax.grid(True, color="0.82", linewidth=0.6)
add_panel_label(ax, "(a)")

ax = axes[0, 1]
ax.plot(view["time_s"], view["link_deg"], color="#00A870", linewidth=1.5)
ax.axhline(30.0, color="0.25", linestyle="--", linewidth=1.0)
ax.set_title("Link-Side Angle")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Link angle (deg)")
ax.set_xlim(0, view["time_s"].max())
ax.set_ylim(0, 35)
ax.grid(True, color="0.82", linewidth=0.6)
add_panel_label(ax, "(b)")

ax = axes[1, 0]
ax.plot(view["time_s"], view["policy_output_numeric"], color="#D94E5D", linewidth=1.2)
ax.axhline(0.0, color="0.35", linewidth=0.8)
ax.set_title("Policy Network Output")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Policy output (raw)")
ax.set_xlim(0, view["time_s"].max())
ax.set_ylim(-150, 150)
ax.grid(True, color="0.82", linewidth=0.6)
add_panel_label(ax, "(c)")

ax = axes[1, 1]
period_time = view.loc[valid_period, "time_s"]
period_ms = view.loc[valid_period, "policy_update_ms"]
ax.plot(period_time, period_ms, color="#7A52C7", linewidth=1.0, marker="o", markersize=2.5)
ax.axhline(1.0, color="0.15", linestyle="--", linewidth=1.1, label="1 ms reference")
ax.set_title("Policy Update Period")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Update period (ms)")
ax.set_xlim(0, view["time_s"].max())
if len(period_ms):
    lower = min(0.7, float(period_ms.min()) - 0.05)
    upper = max(1.3, float(period_ms.max()) + 0.05)
    ax.set_ylim(lower, upper)
ax.legend(loc="upper right", frameon=True)
ax.grid(True, color="0.82", linewidth=0.6)
add_panel_label(ax, "(d)")

fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(PNG_PATH)
print(PDF_PATH)
