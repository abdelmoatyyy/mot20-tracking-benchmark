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
with (RESULTS / 'botsorttrack' / 'model_metrics.csv').open(newline='') as handle:
    botsort = {row['Model']: row for row in csv.DictReader(handle)}

trackers = ['ByteTrack', 'OC-SORT', 'Deep-OC-SORT', 'BoostTrack']
accuracy_trackers = ['ByteTrack', 'BoT-SORT', 'OC-SORT', 'Deep-OC-SORT', 'BoostTrack']
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


def grouped_chart(field: str, title: str, ylabel: str, filename: str,
                  digits: int = 1, include_botsort: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.8), layout='constrained')
    labels = accuracy_trackers if include_botsort else trackers
    x = np.arange(len(labels))
    width = 0.35
    for offset, model in [(-width / 2, 'yolo26s'), (width / 2, 'yolo26m')]:
        values = [
            float(botsort[model][{
                'HOTA': 'HOTA (%)', 'IDF1': 'IDF1 (%)', 'MOTA': 'MOTA (%)',
                'ID_switches': 'ID Switches', 'fragmentations': 'Frag',
            }[field]]) if tracker == 'BoT-SORT' else float(by_key[(model, tracker)][field])
            for tracker in labels
        ]
        bars = ax.bar(x + offset, values, width, label=model.upper(), color=colors[model])
        ax.bar_label(bars, labels=[f'{value:.{digits}f}' for value in values], padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_title(title, loc='left', weight='bold', pad=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


grouped_chart('HOTA', 'Tracking accuracy · HOTA', 'HOTA (%) · higher is better', 'hota.png', include_botsort=True)
grouped_chart('IDF1', 'Identity accuracy · IDF1', 'IDF1 (%) · higher is better', 'idf1.png', include_botsort=True)
grouped_chart('MOTA', 'Tracking accuracy · MOTA', 'MOTA (%) · higher is better', 'mota.png', include_botsort=True)
grouped_chart('FPS', 'Processing speed', 'Frames per second · higher is better', 'fps.png')
grouped_chart('mean_latency_ms', 'Mean processing latency', 'Milliseconds · lower is better', 'latency.png', 0)
grouped_chart('peak_GPU_allocated_MB', 'Peak GPU allocation', 'MB · lower is better', 'gpu_memory.png', 0)
grouped_chart('ID_switches', 'Identity switches', 'Count · lower is better', 'id_switches.png', 0, include_botsort=True)
grouped_chart('fragmentations', 'Track fragmentations', 'Count · lower is better', 'fragmentations.png', 0, include_botsort=True)

fig, (ax, missing_ax) = plt.subplots(
    1, 2, figsize=(11.5, 5.4), layout='constrained',
    gridspec_kw={'width_ratios': [3.4, 1.4]},
)
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
missing_ax.set_title('BoT-SORT', loc='left', weight='bold', pad=14)
for y, model in enumerate(models):
    value = float(botsort[model]['HOTA (%)'])
    missing_ax.barh(y, value, height=0.46, color=colors[model])
    missing_ax.text(value + 0.4, y, f'{value:.1f}%', va='center', fontsize=9)
missing_ax.set_yticks(range(len(models)), ['YOLO26s', 'YOLO26m'])
missing_ax.invert_yaxis()
missing_ax.set_xlim(0, 34)
missing_ax.set_xlabel('HOTA (%)')
missing_ax.text(0.5, -0.2, 'End-to-end FPS not reported', transform=missing_ax.transAxes,
                ha='center', va='top', fontsize=9)
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
