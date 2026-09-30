"""Renders the paper's figures from results/*.json into docs/figures/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
INK, GREY, ACCENT = "#1f2937", "#9ca3af", "#2563eb"

res = json.load(open("results/results.json"))

# Figure 1: architecture.
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.set_xlim(0, 120)
ax.set_ylim(0, 55)
ax.axis("off")


def box(cx, cy, w, h, lines, accent=False):
    ax.add_patch(
        FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                       fc="#dbeafe" if accent else "white", ec=ACCENT if accent else INK, lw=1.4 if accent else 0.9)
    )
    ax.text(cx, cy + 2.2, lines[0], ha="center", va="center", fontsize=8.5, color=INK, weight="bold")
    ax.text(cx, cy - 2.6, lines[1], ha="center", va="center", fontsize=7.5, color=INK)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=INK, lw=0.9))


W, H = 21, 11
top, bot = 36, 10
xs = [11, 35, 59, 83, 108]
box(xs[0], top, W, H, ["Observation o_t", "2 x (16x16) frames"])
box(xs[1], top, W, H, ["Encoder E", "512-128-16, tanh"])
box(xs[2], top, W, H, ["Latent z_t", "16 dimensions"], accent=True)
box(xs[3], top, W, H, ["Dynamics F", "residual, 18-128-16"], accent=True)
box(xs[4], top, W, H, ["Imagined z", "steps t+1 ... t+H"], accent=True)
box(xs[2], bot, W, H, ["Decoder D", "16-128-256"])
box(xs[3], bot, W, H, ["Reward head R", "16-64-1"])
box(xs[4], bot, W, H, ["Loss terms", "pixel, reward, latent"])
ax.text(xs[3], 51, "action a_t (2-d)", ha="center", fontsize=8, color=INK)
arrow(xs[3], 49.5, xs[3], top + H / 2 + 0.5)
for i in range(4):
    arrow(xs[i] + W / 2 + 0.4, top, xs[i + 1] - W / 2 - 0.4, top)
arrow(xs[2], top - H / 2 - 0.4, xs[2], bot + H / 2 + 0.4)
arrow(xs[4] - 4, top - H / 2 - 0.4, xs[3] + 3, bot + H / 2 + 0.4)
arrow(xs[4], top - H / 2 - 0.4, xs[4], bot + H / 2 + 0.4)
ax.text(xs[4] + 1.5, 23, "unroll H = 8", fontsize=7.5, color="#6b7280")
ax.text(1, 21, "Planning: CEM scores action\nsequences by summed R(z)\ninside the latent space;\nno pixels are decoded.", fontsize=7.5, color="#6b7280", va="center")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_architecture.png", dpi=220)
plt.close(fig)

# Figure 2: training curves (log scale, one panel per loss term).
c = res["curve"]
steps = [r["step"] for r in c]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=True)
for ax, key, title in zip(axes, ["recon", "reward", "consistency"],
                          ["Pixel reconstruction (MSE)", "Reward prediction (MSE)", "Latent consistency (MSE)"]):
    ax.plot(steps, [r[key] for r in c], color=ACCENT, lw=1.4)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("gradient step")
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_training.png", dpi=220)
plt.close(fig)

# Figure 3: open-loop prediction error vs horizon.
p = res["prediction"]
k = np.arange(len(p["model"]))
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.plot(k, p["copyLast"], color=GREY, lw=1.6, marker="o", ms=3, label="Copy last frame (baseline)")
ax.plot(k, p["model"], color=ACCENT, lw=1.8, marker="o", ms=3, label="NextWM open-loop imagination")
ax.set_xlabel("prediction horizon k (steps)")
ax.set_ylabel("pixel MSE vs. true frame")
ax.grid(axis="y", color="#e5e7eb", lw=0.6)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_prediction.png", dpi=220)
plt.close(fig)

# Figure 4: qualitative rollout, real (top) vs imagined (bottom).
if os.path.exists("results/rollout.json"):
    r = json.load(open("results/rollout.json"))
    idx = [0, 2, 4, 6, 8, 10, 12, 15]
    fig, axes = plt.subplots(2, len(idx), figsize=(7.2, 2.1))
    for j, t in enumerate(idx):
        for i, key in enumerate(["real", "imagined"]):
            a = axes[i, j]
            a.imshow(np.array(r[key][t]).reshape(16, 16), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(f"k={t}", fontsize=8)
        axes[0, 0].set_ylabel("real", fontsize=8)
        axes[1, 0].set_ylabel("imagined", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_rollout.png", dpi=220)
    plt.close(fig)

# Figure 5: ablation, ranking correlation and closed-loop final distance.
runs = [("ou-base", "OU data, base loss"), ("ou-full", "OU data, reward-weighted"),
        ("hold-base", "Hold data, base loss"), ("results", "NextWM (hold, reward-weighted)")]
if os.path.exists("results/iter1.json"):
    runs.append(("iter1", "NextWM + 1 planner round"))
rows = []
for key, label in runs:
    d = json.load(open(f"results/{key}.json"))
    c = {x["policy"]: x for x in d["control"]}
    rows.append((label, d["ranking"], c["nextwm"]["meanFinalDistance"]))
rand = c["random"]["meanFinalDistance"]
orac = c["oracle"]["meanFinalDistance"]
labels = [r[0] for r in rows]
y = np.arange(len(rows))[::-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
cols = [GREY] * (len(rows) - 1) + [ACCENT]
if len(rows) == 5:
    cols = [GREY, GREY, GREY, "#93c5fd", ACCENT]
a1.barh(y, [r[1] for r in rows], color=cols)
for yi, r in zip(y, rows):
    a1.text(r[1] + 0.01, yi, f"{r[1]:.2f}", va="center", fontsize=8)
a1.set_xlim(0, 1)
a1.set_yticks(y)
a1.set_yticklabels(labels, fontsize=8)
a1.set_title("Return-ranking correlation (higher is better)", fontsize=9)
a2.barh(y, [r[2] for r in rows], color=cols)
for yi, r in zip(y, rows):
    a2.text(r[2] + 0.01, yi, f"{r[2]:.2f}", va="center", fontsize=8)
a2.axvline(rand, color=INK, ls="--", lw=1)
a2.text(rand, y[-1] - 0.75, "random", fontsize=7.5, ha="center")
a2.axvline(orac, color=INK, ls=":", lw=1)
a2.text(orac + 0.02, y[-1] - 0.75, "oracle", fontsize=7.5)
a2.set_title("Final distance to goal (lower is better)", fontsize=9)
a2.set_xlim(0, max(rand, max(r[2] for r in rows)) * 1.15)
a2.set_ylim(y[-1] - 1.1, y[0] + 0.6)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_ablation.png", dpi=220)
plt.close(fig)
print("figures written to", OUT)
