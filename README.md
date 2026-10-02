# MOT20 pedestrian tracking benchmark

A reproducible comparison of **YOLO26s** and **YOLO26m** for pedestrian detection and multi-object tracking on the four sequences of the **MOT20 training split**. The evaluated trackers are **ByteTrack**, **OC-SORT**, **Deep-OC-SORT**, and **BoostTrack**. **BoT-SORT** is listed in the intended comparison, but no BoT-SORT measurements were present in the supplied artifacts; its results remain open.

Inspired by the clear experiment-and-results layout of [Person-ReID-BenchMark](https://github.com/BASSAT-BASSAT/Person-ReID-BenchMark).

## Contents

- [Setup](#setup)
- [Results](#results)
- [Detailed tables and graphs](docs/benchmark/tables.md)
- [Interpretation](#interpretation)
- [Repository layout](#repository-layout)
- [Reproduce the experiments](#reproduce-the-experiments)
- [Scope and limitations](#scope-and-limitations)

## Setup

| Item | Recorded setup |
| --- | --- |
| Dataset | MOT20 **train**: MOT20-01, MOT20-02, MOT20-03, MOT20-05; **8,931 frames** total |
| Detectors | Pretrained `yolo26s.pt` and `yolo26m.pt`; person class only |
| Input / inference | 1280 px; confidence 0.10; NMS IoU 0.70; FP32 |
| Hardware | NVIDIA Tesla T4 GPU (Kaggle notebooks) |
| Evaluation | TrackEval for tracking; Ultralytics validation for ByteTrack notebook detector metrics |
| Software | Ultralytics 8.4.171 and PyTorch 2.10.0+cu128 recorded for OC-SORT, Deep-OC-SORT, and BoostTrack; BoxMOT 25.0.0 recorded for BoostTrack |

Tracker settings and any run-specific differences are recorded in the [configuration files](results/). ByteTrack uses Ultralytics `bytetrack.yaml`; OC-SORT and Deep-OC-SORT use the supplied YAML files. BoostTrack uses an OSNet ReID model and the installed implementation defaults recorded in its configuration. These runs were performed separately, so timing comparisons should be treated as indicative rather than a controlled simultaneous benchmark.

## Results

![HOTA comparison](docs/benchmark/figures/hota.png)

![IDF1 comparison](docs/benchmark/figures/idf1.png)

Tracking scores are percentages; higher HOTA, IDF1, MOTA, and FPS are better. Lower ID switches, latency, and GPU allocation are better. Values below are rounded from [comparison.csv](results/comparison.csv); full precision and additional columns remain in the source CSVs.

| Detector | Tracker | HOTA ↑ | IDF1 ↑ | MOTA ↑ | ID switches ↓ | FPS ↑ | Mean latency (ms) ↓ | Peak GPU (MB) ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26s | ByteTrack | 22.70 | 28.00 | 20.90 | 1,500 | 15.57 | 64.99 | 196.78 |
| YOLO26m | ByteTrack | 25.00 | 32.80 | 25.90 | 1,818 | 10.35 | 96.76 | 351.73 |
| YOLO26s | OC-SORT | 19.97 | 23.46 | 16.82 | 1,840 | 16.02 | 62.41 | 196.03 |
| YOLO26m | OC-SORT | 21.90 | 27.50 | 20.75 | 2,164 | 11.63 | 85.96 | 350.98 |
| YOLO26s | Deep-OC-SORT | 18.73 | 21.36 | 14.76 | 2,024 | 20.22 | 49.45 | 213.34 |
| YOLO26m | Deep-OC-SORT | 20.47 | 25.13 | 18.22 | 2,336 | 11.19 | 89.39 | 383.44 |
| YOLO26s | BoostTrack | 27.21 | 37.03 | 28.13 | 2,471 | 5.67 | 176.36 | 375.02 |
| YOLO26m | BoostTrack | **29.72** | **41.17** | **32.71** | 2,547 | 4.92 | 203.09 | 442.81 |
| YOLO26s | BoT-SORT | — | — | — | — | — | — | — |
| YOLO26m | BoT-SORT | — | — | — | — | — | — | — |

**Detection-only results** from the ByteTrack notebook's validation stage:

![Detector validation metrics](docs/benchmark/figures/detection.png)

| Detector | Precision ↑ | Recall ↑ | mAP@50 ↑ | mAP@50:95 ↑ |
| --- | ---: | ---: | ---: | ---: |
| YOLO26s | 0.6040 | 0.6072 | 0.5165 | 0.2109 |
| YOLO26m | **0.6246** | **0.6418** | **0.5461** | **0.2216** |

The detection table uses unit-scale values (0–1); the tracking table uses percentages (0–100). Detection was evaluated once per detector, not once per tracker.

For MOTA, throughput, GPU memory, and additional tables, see the [full benchmark report](docs/benchmark/tables.md).

## Interpretation

- **Highest tracking accuracy:** YOLO26m + BoostTrack leads the recorded runs in HOTA (29.72), IDF1 (41.17), and MOTA (32.71), with 4.92 FPS.
- **Speed among YOLO26s runs:** Deep-OC-SORT records 20.22 FPS, with HOTA 18.73. ByteTrack records 15.57 FPS and HOTA 22.70.
- **Detector size:** YOLO26m improves HOTA for every evaluated tracker, with lower FPS in each corresponding run.
- **Identity switches:** ByteTrack has the lowest count for each detector, though switch counts alone do not determine HOTA or IDF1.

## Repository layout

```text
notebooks/
  bytetrack.ipynb                     # detection, ByteTrack, TrackEval, profiling, visuals
  ocsort_deepocsort_boosttrack.ipynb  # three tracker runs and evaluation
results/
  comparison.csv                      # eight measured detector/tracker combinations
  detection_metrics.csv               # detector-only validation
  tracking_metrics.csv                # ByteTrack tracking metrics
  system_metrics.csv                  # ByteTrack timing and memory
  system_per_sequence.csv             # ByteTrack sequence timing
  ocsort/                             # configuration, summaries, sequence detail
  deepocsort/                         # configuration, summaries, sequence detail
  boosttrack/                          # configuration, summaries, sequence detail
docs/benchmark/
  tables.md                            # detailed tables and charts
  figures/                             # generated PNG bar charts
scripts/
  plot_comparison.py                   # regenerate all figures from the CSVs
```

The notebooks are committed with cell outputs cleared to keep the repository manageable. Original ground-truth copies, prediction text files, videos, and checkpoint weights are omitted. Download MOT20 from its official source before rerunning.

## Reproduce the experiments

1. Obtain the MOT20 training images and labels from the [MOTChallenge dataset page](https://motchallenge.net/data/MOT20/), accepting its terms. The notebooks expect the Kaggle layout under `/kaggle/input`; adjust the path discovery cells if running elsewhere.
2. Open [the ByteTrack notebook](notebooks/bytetrack.ipynb) in order for detector validation, ByteTrack tracking, timing, and TrackEval. Its setup cells install the needed dependencies and locate the dataset.
3. Open [the other-trackers notebook](notebooks/ocsort_deepocsort_boosttrack.ipynb) for OC-SORT, Deep-OC-SORT, and BoostTrack. Run its sections in order; the notebook contains installation and TrackEval setup cells.
4. Compare outputs with [comparison.csv](results/comparison.csv) and the tracker-specific summary and per-sequence CSVs under `results/`.

To regenerate the report figures from the committed CSVs, install `matplotlib` and `numpy`, then run `python scripts/plot_comparison.py`.

The notebooks were written for Kaggle and contain environment-specific paths such as `/kaggle/working`. They are experiment records, not a single command-line benchmark runner. The BoostTrack OSNet checkpoint is not included, so reproducing that run requires obtaining the same checkpoint. No BoT-SORT run is included.

## Scope and limitations

These are results on the **MOT20 training split**, not MOT20 test leaderboard scores. The detectors use pretrained weights without MOT20 fine-tuning. Scores reflect the recorded single runs and tracker configurations, and some tracker implementations and ReID settings differ. FPS is taken from each run's own summary, so compare speed with that context. `R@1` and `R@5` fields in some raw summaries are blank and are not reported here.

## Acknowledgments

[MOTChallenge / MOT20](https://motchallenge.net/data/MOT20/), [Ultralytics](https://github.com/ultralytics/ultralytics), [TrackEval](https://github.com/JonathonLuiten/TrackEval), and [BoxMOT](https://github.com/mikel-brostrom/boxmot).
