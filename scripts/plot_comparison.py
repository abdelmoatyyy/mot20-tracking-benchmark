"""Render benchmark figures from the committed result CSV files."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'docs' / 'benchmark' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

with (RESULTS / 'comparison.csv').open(newline='') as handle:
    runs = list(csv.DictReader(handle))
with (RESULTS / 'detection_metrics.csv').open(newline='') as handle:
    detection = {row['']: row for row in csv.DictReader(handle)}

trackers = ['ByteTrack', 'OC-SORT', 'Deep-OC-SORT', 'BoostTrack']
models = ['yolo26s', 'yolo26m']
colors = {'yolo26s': '#3478C0', 'yolo26m': '#E38435'}
by_key = {(row['detector'], row['tracker']): row for row in runs}

plt.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.alpha': 0.2,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def grouped_chart(field: str, title: str, ylabel: str, filename: str, digits: int = 1) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.8), layout='constrained')
    x = np.arange(len(trackers))
    width = 0.35
    for offset, model in [(-width / 2, 'yolo26s'), (width / 2, 'yolo26m')]:
        values = [float(by_key[(model, tracker)][field]) for tracker in trackers]
        bars = ax.bar(x + offset, values, width, label=model.upper(), color=colors[model])
        ax.bar_label(bars, labels=[f'{value:.{digits}f}' for value in values], padding=3, fontsize=9)
    ax.set_xticks(x, trackers)
    ax.set_title(title, loc='left', weight='bold', pad=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


grouped_chart('HOTA', 'Tracking accuracy · HOTA', 'HOTA (%) · higher is better', 'hota.png')
grouped_chart('IDF1', 'Identity accuracy · IDF1', 'IDF1 (%) · higher is better', 'idf1.png')
grouped_chart('MOTA', 'Tracking accuracy · MOTA', 'MOTA (%) · higher is better', 'mota.png')
grouped_chart('FPS', 'Processing speed', 'Frames per second · higher is better', 'fps.png')
grouped_chart('mean_latency_ms', 'Mean processing latency', 'Milliseconds · lower is better', 'latency.png', 0)
grouped_chart('peak_GPU_allocated_MB', 'Peak GPU allocation', 'MB · lower is better', 'gpu_memory.png', 0)
grouped_chart('ID_switches', 'Identity switches', 'Count · lower is better', 'id_switches.png', 0)
grouped_chart('fragmentations', 'Track fragmentations', 'Count · lower is better', 'fragmentations.png', 0)

fig, ax = plt.subplots(figsize=(9, 5.4), layout='constrained')
for tracker in trackers:
    for model in models:
        run = by_key[(model, tracker)]
        ax.scatter(float(run['FPS']), float(run['HOTA']), s=120, color=colors[model],
                   marker={'ByteTrack': 'o', 'OC-SORT': 's', 'Deep-OC-SORT': '^', 'BoostTrack': 'D'}[tracker])
        ax.annotate(f'{tracker} {model[-1]}', (float(run['FPS']), float(run['HOTA'])),
                    xytext=(6, 5), textcoords='offset points', fontsize=8)
ax.set_xlim(0, 24)
ax.set_ylim(15, 35)
ax.set_xlabel('Frames per second · higher is better')
ax.set_ylabel('HOTA (%) · higher is better')
ax.set_title('Tracking accuracy versus speed', loc='left', weight='bold', pad=14)
fig.savefig(FIGURES / 'accuracy_vs_speed.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.4), layout='constrained')
metrics = [('precision', 'Precision'), ('recall', 'Recall'), ('map50', 'mAP@50'), ('map50_95', 'mAP@50:95')]
x = np.arange(len(metrics))
width = 0.35
for offset, model in [(-width / 2, 'yolo26s'), (width / 2, 'yolo26m')]:
    values = [float(detection[model][key]) for key, _ in metrics]
    bars = ax.bar(x + offset, values, width, label=model.upper(), color=colors[model])
    ax.bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=3, fontsize=9)
ax.set_xticks(x, [label for _, label in metrics])
ax.set_ylim(0, 0.78)
ax.set_ylabel('Score (0–1) · higher is better')
ax.set_title('Detector-only validation', loc='left', weight='bold', pad=14)
ax.legend(frameon=False, ncol=2)
fig.savefig(FIGURES / 'detection.png', dpi=180)
plt.close(fig)

print(f'Wrote 10 figures to {FIGURES.relative_to(ROOT)}')
