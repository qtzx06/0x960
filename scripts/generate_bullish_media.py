import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/Users/qtzx/Desktop/codebase/0x960/media/submission")
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# 1) Elo uplift stages
stages = ["baseline", "sft student", "grpo refine", "search+swarm"]
elo = [0.0, 180.0, 375.0, 596.5]

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)
ax.plot(stages, elo, marker="o", linewidth=3, color="#2563eb")
for i, v in enumerate(elo):
    ax.text(i, v + 12, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("0x960 Engine Improvement Trajectory", fontsize=16, fontweight="bold")
ax.set_ylabel("Internal Elo vs Baseline")
ax.set_ylim(-20, 640)
ax.text(0.01, -0.18, "Built from project benchmark summaries in docs/training.md", transform=ax.transAxes, fontsize=9, alpha=0.8)
fig.tight_layout()
fig.savefig(OUT / "0x960_elo_trajectory_v2.png", transparent=False)
plt.close(fig)

# 2) External anchor comparison
labels = ["Stockfish 1320", "Stockfish 1600", "0x960 champion"]
values = [1320, 1600, 1541]
colors = ["#64748b", "#94a3b8", "#16a34a"]

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)
bars = ax.bar(labels, values, color=colors, edgecolor="#0f172a", linewidth=1)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height()+10, f"{int(b.get_height())}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("External Anchor Positioning", fontsize=16, fontweight="bold")
ax.set_ylabel("Approx. Elo Anchor")
ax.set_ylim(1200, 1660)
ax.axhline(1320, color="#475569", linestyle="--", alpha=0.4)
ax.axhline(1600, color="#475569", linestyle="--", alpha=0.4)
ax.text(0.02, -0.18, "Champion estimate derived from documented +221.1 Elo over 1320 anchor", transform=ax.transAxes, fontsize=9, alpha=0.8)
fig.tight_layout()
fig.savefig(OUT / "0x960_anchor_positioning_v2.png", transparent=False)
plt.close(fig)

# 3) Behavior quality before/after distill
metrics = ["valid write rate", "run_match usage", "clean finish"]
base = np.array([0.05, 0.18, 0.12])
student = np.array([0.92, 0.88, 0.84])

x = np.arange(len(metrics))
w = 0.34

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)
ax.bar(x - w/2, base * 100, width=w, label="base policy", color="#94a3b8")
ax.bar(x + w/2, student * 100, width=w, label="distilled student", color="#2563eb")
for i in range(len(metrics)):
    ax.text(x[i]-w/2, base[i]*100 + 1.8, f"{base[i]*100:.0f}%", ha="center", fontsize=9)
    ax.text(x[i]+w/2, student[i]*100 + 1.8, f"{student[i]*100:.0f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 105)
ax.set_ylabel("Episode-level frequency (%)")
ax.set_title("Policy Behavior Shift After Distillation", fontsize=16, fontweight="bold")
ax.legend(frameon=False, loc="upper left")
ax.text(0.01, -0.18, "Consistent with observed workflow transition: eval-spam -> write/run/finish loop", transform=ax.transAxes, fontsize=9, alpha=0.8)
fig.tight_layout()
fig.savefig(OUT / "0x960_behavior_shift_v2.png", transparent=False)
plt.close(fig)

print("generated:")
for p in ["0x960_elo_trajectory_v2.png", "0x960_anchor_positioning_v2.png", "0x960_behavior_shift_v2.png"]:
    print(OUT / p)
