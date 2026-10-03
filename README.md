import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(HERE / "results" / "crowdhuman_detector_comparison.csv")))
for r in rows:
    for k in r:
        if k != "Model":
            r[k] = float(r[k])

BLUE, ORANGE, GREY = "#3578BE", "#E08336", "#B8BEC6"
def color(m):
    return BLUE if m == "YOLO26s" else ORANGE if m == "YOLO26m" else GREY
plt.rcParams.update({"font.size": 14})

# 1) mAP50-95 ranking
rs = sorted(rows, key=lambda r: r["mAP50-95"])
fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
bars = ax.barh([r["Model"] for r in rs], [r["mAP50-95"] for r in rs], color=[color(r["Model"]) for r in rs])
for b in bars:
    ax.text(b.get_width() + 0.002, b.get_y() + b.get_height()/2, f"{b.get_width():.3f}", va="center", fontsize=12)
ax.set_xlabel("mAP@50:95 on CrowdHuman \u00b7 higher is better")
ax.set_xlim(0, max(r["mAP50-95"] for r in rows) * 1.15)
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "crowdhuman_map.png")
plt.close(fig)

# 2) accuracy vs speed
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
for r in rows:
    sel = r["Model"] in ("YOLO26s", "YOLO26m")
    ax.scatter(r["GPU_FPS"], r["mAP50-95"], s=260 if sel else 110, color=color(r["Model"]),
               edgecolor="white", zorder=3)
    off = (8, -16) if r["Model"] in ("YOLO11n", "YOLO11s") else (8, 6)
    ax.annotate(r["Model"], (r["GPU_FPS"], r["mAP50-95"]), textcoords="offset points",
                xytext=off, fontsize=11, fontweight="bold" if sel else "normal")
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 50, 100, 200])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("GPU FPS (log scale) \u00b7 higher is better")
ax.set_ylabel("mAP@50:95 on CrowdHuman \u00b7 higher is better")
ax.grid(alpha=0.3)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "crowdhuman_map_vs_speed.png")
print("done")
