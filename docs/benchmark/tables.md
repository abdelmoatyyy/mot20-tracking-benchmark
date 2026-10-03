# MOT20 pedestrian tracking benchmark

A reproducible comparison of **YOLO26s** and **YOLO26m** for pedestrian detection and multi-object tracking on the four sequences of the **MOT20 training split**. The evaluated trackers are **ByteTrack**, **BoT-SORT**, **OC-SORT**, **Deep-OC-SORT**, and **BoostTrack**.
=======
All tracking scores are percentages on the MOT20 training sequences. Timings and memory are from each run’s recorded summary. BoT-SORT has tracking scores but no end-to-end timing or memory measurements. See the [main README](../../README.md) for setup and limitations.

Inspired by the clear experiment-and-results layout of [Person-ReID-BenchMark](https://github.com/BASSAT-BASSAT/Person-ReID-BenchMark).

## Contents

- [Setup](#setup)
- [Results](#results)
- [Performance comparisons](#performance-comparisons)
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
=======
| Detector | Tracker | HOTA ↑ | IDF1 ↑ | MOTA ↑ | ID switches ↓ | Fragmentations ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| yolo26s | ByteTrack | 22.70 | 28.00 | 20.90 | 1,500 | 3,714 |
| yolo26m | ByteTrack | 25.00 | 32.80 | 25.90 | 1,818 | 4,469 |
| yolo26s | BoT-SORT | 23.20 | 29.10 | 21.20 | 1,361 | 3,000 |
| yolo26m | BoT-SORT | 26.00 | 34.40 | 26.40 | 1,596 | 3,571 |
| yolo26s | OC-SORT | 19.97 | 23.46 | 16.82 | 1,840 | 10,652 |
| yolo26m | OC-SORT | 21.90 | 27.50 | 20.75 | 2,164 | 12,055 |
| yolo26s | Deep-OC-SORT | 18.73 | 21.36 | 14.76 | 2,024 | 12,327 |
| yolo26m | Deep-OC-SORT | 20.47 | 25.13 | 18.22 | 2,336 | 14,359 |
| yolo26s | BoostTrack | 27.21 | 37.03 | 28.13 | 2,471 | 11,083 |
| yolo26m | BoostTrack | 29.72 | 41.17 | 32.71 | 2,547 | 11,152 |

Tracker settings and any run-specific differences are recorded in the [configuration files](results/). ByteTrack uses Ultralytics `bytetrack.yaml`; OC-SORT and Deep-OC-SORT use the supplied YAML files. BoostTrack uses an OSNet ReID model and the installed implementation defaults recorded in its configuration. BoT-SORT scores and timing come from its own run summaries; its environment details were not recorded in the supplied artifacts. These runs were performed separately, so timing comparisons should be treated as indicative rather than a controlled simultaneous benchmark.

## Results

![HOTA comparison](docs/benchmark/figures/hota (2).png)

![IDF1 comparison](docs/benchmark/figures/idf1.png)

![MOTA comparison](docs/benchmark/figures/mota.png)

Tracking scores are percentages; higher HOTA, IDF1, MOTA, and FPS are better. Lower ID switches, latency, and GPU allocation are better. Values below are rounded from [comparison.csv](results/comparison.csv); full precision and additional columns remain in the source CSVs.

| Detector | Tracker | HOTA ↑ | IDF1 ↑ | MOTA ↑ | ID switches ↓ | FPS ↑ | Mean latency (ms) ↓ | Peak GPU (MB) ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26s | ByteTrack | 22.70 | 28.00 | 20.90 | 1,500 | 15.57 | 64.99 | 196.78 |
| YOLO26m | ByteTrack | 25.00 | 32.80 | 25.90 | 1,818 | 10.35 | 96.76 | 351.73 |
| YOLO26s | BoT-SORT | 23.20 | 29.10 | 21.20 | **1,361** | 16.00 | 62.60 | 212.20 |
| YOLO26m | BoT-SORT | 26.00 | 34.40 | 26.40 | **1,596** | 10.20 | 97.70 | 435.00 |
| YOLO26s | OC-SORT | 19.97 | 23.46 | 16.82 | 1,840 | 16.02 | 62.41 | 196.03 |
| YOLO26m | OC-SORT | 21.90 | 27.50 | 20.75 | 2,164 | 11.63 | 85.96 | 350.98 |
| YOLO26s | Deep-OC-SORT | 18.73 | 21.36 | 14.76 | 2,024 | 20.22 | 49.45 | 213.34 |
| YOLO26m | Deep-OC-SORT | 20.47 | 25.13 | 18.22 | 2,336 | 11.19 | 89.39 | 383.44 |
| YOLO26s | BoostTrack | 27.21 | 37.03 | 28.13 | 2,471 | 5.67 | 176.36 | 375.02 |
| YOLO26m | BoostTrack | **29.72** | **41.17** | **32.71** | 2,547 | 4.92 | 203.09 | 442.81 |

**BoT-SORT combined metrics** (TrackEval, all four sequences):

| Detector | HOTA ↑ | MOTA ↑ | IDF1 ↑ | DetA ↑ | AssA ↑ | ID switches ↓ | Frag ↓ | FPS ↑ | Mean latency (ms) ↓ | Peak GPU (MB) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26s | 23.2 | 21.2 | 29.1 | 17.0 | 32.0 | 1,361 | 3,000 | 16.0 | 62.6 | 212.2 |
| YOLO26m | 26.0 | 26.4 | 34.4 | 21.5 | 31.7 | 1,596 | 3,571 | 10.2 | 97.7 | 435.0 |

**BoT-SORT per-sequence results** (HOTA / MOTA / IDF1 in %, from TrackEval):

| Detector | Sequence | HOTA ↑ | DetA ↑ | AssA ↑ | MOTA ↑ | IDF1 ↑ | ID switches ↓ | Frag ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26s | MOT20-01 | 41.0 | 42.2 | 40.7 | 49.6 | 50.7 | 190 | 297 |
| YOLO26s | MOT20-02 | 35.8 | 37.0 | 35.1 | 43.9 | 46.2 | 702 | 1,240 |
| YOLO26s | MOT20-03 | 30.0 | 26.1 | 34.6 | 34.8 | 45.9 | 267 | 1,054 |
| YOLO26s | MOT20-05 | 11.0 | 6.7 | 18.1 | 8.3 | 12.0 | 202 | 409 |
| YOLO26s | **Combined** | 23.2 | 17.0 | 32.0 | 21.2 | 29.1 | 1,361 | 3,000 |
| YOLO26m | MOT20-01 | 43.3 | 44.4 | 43.3 | 50.8 | 54.3 | 177 | 332 |
| YOLO26m | MOT20-02 | 37.0 | 39.2 | 35.4 | 45.6 | 46.7 | 792 | 1,452 |
| YOLO26m | MOT20-03 | 32.0 | 27.9 | 36.7 | 36.5 | 49.1 | 216 | 992 |
| YOLO26m | MOT20-05 | 16.4 | 13.0 | 20.7 | 16.2 | 20.9 | 411 | 795 |
| YOLO26m | **Combined** | 26.0 | 21.5 | 31.7 | 26.4 | 34.4 | 1,596 | 3,571 |

MOT20-05 is the hardest sequence for both detectors, and it is where YOLO26m gains the most over YOLO26s (HOTA 11.0 → 16.4).


**Detection-only results** from the ByteTrack notebook's validation stage:
=======
Each point in the left panel has measured HOTA and end-to-end FPS. Moving upward improves HOTA; moving right improves FPS. The adjacent BoT-SORT panel shows its measured HOTA without inventing an FPS value. Timing reflects each run's own setup.


![Detector validation metrics](docs/benchmark/figures/detection.png)

| Detector | Precision ↑ | Recall ↑ | mAP@50 ↑ | mAP@50:95 ↑ |
| --- | ---: | ---: | ---: | ---: |
| YOLO26s | 0.6040 | 0.6072 | 0.5165 | 0.2109 |
| YOLO26m | **0.6246** | **0.6418** | **0.5461** | **0.2216** |


The detection table uses unit-scale values (0–1); the tracking table uses percentages (0–100). Detection was evaluated once per detector, not once per tracker.

**Detector-only GPU profile** (separate run supplied with the BoT-SORT results; not end-to-end tracking):
=======
| Tracker | ΔHOTA (pt) | ΔIDF1 (pt) | ΔMOTA (pt) | ΔFPS | ΔLatency (ms) | ΔGPU (MB) | ΔID switches | ΔFragmentations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ByteTrack | +2.30 | +4.80 | +5.00 | -5.22 | +31.77 | +154.94 | +318 | +755 |
| BoT-SORT | +2.80 | +5.30 | +5.20 | n/r | n/r | n/r | +235 | +571 |
| OC-SORT | +1.93 | +4.04 | +3.93 | -4.39 | +23.55 | +154.94 | +324 | +1,403 |
| Deep-OC-SORT | +1.74 | +3.77 | +3.47 | -9.04 | +39.94 | +170.10 | +312 | +2,032 |
| BoostTrack | +2.51 | +4.14 | +4.58 | -0.75 | +26.73 | +67.79 | +76 | +69 |


| Detector | Parameters (M) | GPU inference (ms) | GPU FPS | Precision | Recall | mAP@50 | mAP@50:95 | Peak GPU allocated (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26s | 10.01 | 37.00 | 27.02 | 0.8492 | 0.3835 | 0.3468 | 0.1532 | 2.20 |
| YOLO26m | 21.90 | 103.71 | 9.64 | 0.8249 | 0.4517 | 0.4038 | 0.1749 | 4.15 |

This profile uses a different evaluation setup from the validation table above (precision and recall differ noticeably), and its timing and memory describe the detector alone, so it should not be compared with the end-to-end tracker columns in the main table.

## Performance comparisons

![Tracking accuracy versus processing speed](docs/benchmark/figures/accuracy_vs_speed1.png)

The plot shows each measured detector and tracker combination. BoostTrack has the highest HOTA, while Deep-OC-SORT with YOLO26s has the highest recorded FPS. The [detailed benchmark report](docs/benchmark/tables.md) includes bar charts for FPS, latency, GPU memory, ID switches, and fragmentations, plus per-sequence speed.

![FPS comparison](docs/benchmark/figures/fps1.png)

![Mean latency comparison](docs/benchmark/figures/latency1.png)

![Peak GPU memory comparison](docs/benchmark/figures/gpu_memory1.png)

![ID switches comparison](docs/benchmark/figures/id_switches1.png)


| Tracker | ΔHOTA (pt) | ΔIDF1 (pt) | ΔMOTA (pt) | ΔFPS | ΔLatency (ms) | ΔGPU (MB) | ΔID switches | ΔFragmentations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ByteTrack | +2.30 | +4.80 | +5.00 | -5.22 | +31.77 | +154.94 | +318 | +755 |
| BoT-SORT | +2.80 | +5.30 | +5.20 | -5.80 | +35.10 | +222.80 | +235 | +571 |
| OC-SORT | +1.93 | +4.04 | +3.93 | -4.39 | +23.55 | +154.94 | +324 | +1,403 |
| Deep-OC-SORT | +1.74 | +3.77 | +3.47 | -9.04 | +39.94 | +170.10 | +312 | +2,032 |
| BoostTrack | +2.51 | +4.14 | +4.58 | -0.75 | +26.73 | +67.79 | +76 | +69 |
=======
- [Combined comparison](../../results/comparison.csv) · [ByteTrack tracking](../../results/tracking_metrics.csv) · [ByteTrack system](../../results/system_metrics.csv)
- [BoT-SORT tracking](../../results/botsorttrack/model_metrics.csv) · [BoT-SORT per-sequence results](../../results/botsorttrack/per_sequence/)
- [OC-SORT summaries](../../results/ocsort/) · [Deep-OC-SORT summaries](../../results/deepocsort/) · [BoostTrack summaries](../../results/boosttrack/)


Each change is **YOLO26m minus YOLO26s** for the same tracker. Positive tracking score changes are improvements; positive latency, GPU, ID switch, and fragmentation changes are increases in cost or error. Timing measurements came from separate runs, as described in [Setup](#setup).

## Interpretation

- **Highest tracking accuracy:** YOLO26m + BoostTrack leads the recorded runs in HOTA (29.72), IDF1 (41.17), and MOTA (32.71), with 4.92 FPS.
- **Best accuracy at practical speed:** YOLO26m + BoT-SORT is second in accuracy (HOTA 26.00, IDF1 34.40, MOTA 26.40) and runs at 10.2 FPS, about twice the speed of YOLO26m + BoostTrack. With YOLO26s, BoT-SORT (HOTA 23.20, 16.0 FPS) edges out ByteTrack (HOTA 22.70, 15.57 FPS).
- **Speed among YOLO26s runs:** Deep-OC-SORT records the highest speed at 20.22 FPS, but with the lowest HOTA (18.73). OC-SORT (16.02 FPS), BoT-SORT (16.00 FPS), and ByteTrack (15.57 FPS) are close to each other, and BoT-SORT has the best HOTA of the three.
- **Identity switches:** BoT-SORT has the lowest count for each detector (1,361 with YOLO26s, 1,596 with YOLO26m), followed by ByteTrack. Switch counts alone do not determine HOTA or IDF1: BoostTrack has the most switches among the compared runs yet the highest HOTA and IDF1.
- **Detector size:** YOLO26m improves HOTA for every evaluated tracker (+1.74 to +2.80 points), at the cost of lower FPS and higher latency and GPU memory in each corresponding run. BoT-SORT gains the most HOTA from the larger detector, and BoostTrack pays the smallest speed cost.
- **Association versus detection:** For BoT-SORT, YOLO26m's gain comes mostly from detection quality (DetA 17.0 → 21.5) rather than association (AssA 32.0 → 31.7).
- **GPU memory:** BoT-SORT uses more GPU memory than ByteTrack and OC-SORT with YOLO26m (435.0 MB versus 351.73 and 350.98 MB), similar to BoostTrack (442.81 MB), but with much higher throughput.

## Repository layout

```text
notebooks/
  bytetrack.ipynb                     # detection, ByteTrack, TrackEval, profiling, visuals
  ocsort_deepocsort_boosttrack.ipynb  # three tracker runs and evaluation
results/
  comparison.csv                      # all measured detector/tracker combinations
  detection_metrics.csv               # detector-only validation
  tracking_metrics.csv                # ByteTrack tracking metrics
  system_metrics.csv                  # ByteTrack timing and memory
  system_per_sequence.csv             # ByteTrack sequence timing
  botsort/
    model_metrics.csv                 # combined BoT-SORT tracking metrics, both detectors
    pedestrian_summary_yolos.csv      # BoT-SORT per-sequence, YOLO26s
    pedestrian_summary_yolom.csv      # BoT-SORT per-sequence, YOLO26m
    mot20_yolo26_comparison.csv       # detector-only GPU profile
  ocsort/                             # configuration, summaries, sequence detail
  deepocsort/                         # configuration, summaries, sequence detail
  boosttrack/                         # configuration, summaries, sequence detail
docs/benchmark/
  tables.md                           # detailed tables and charts
  figures/                            # generated PNG bar charts
scripts/
  plot_comparison.py                  # regenerate all figures from the CSVs
```

The notebooks are committed with cell outputs cleared to keep the repository manageable. Original ground-truth copies, prediction text files, videos, and checkpoint weights are omitted. Download MOT20 from its official source before rerunning.

## Reproduce the experiments

1. Obtain the MOT20 training images and labels from the [MOTChallenge dataset page](https://motchallenge.net/data/MOT20/), accepting its terms. The notebooks expect the Kaggle layout under `/kaggle/input`; adjust the path discovery cells if running elsewhere.
2. Open [the ByteTrack notebook](notebooks/bytetrack.ipynb) in order for detector validation, ByteTrack tracking, timing, and TrackEval. Its setup cells install the needed dependencies and locate the dataset.
3. Open [the other-trackers notebook](notebooks/ocsort_deepocsort_boosttrack.ipynb) for OC-SORT, Deep-OC-SORT, and BoostTrack. Run its sections in order; the notebook contains installation and TrackEval setup cells.
4. Compare outputs with [comparison.csv](results/comparison.csv) and the tracker-specific summary and per-sequence CSVs under `results/`. BoT-SORT's summaries are in `results/botsort/`.

To regenerate the report figures from the committed CSVs, install `matplotlib` and `numpy`, then run `python scripts/plot_comparison.py`.

The notebooks were written for Kaggle and contain environment-specific paths such as `/kaggle/working`. They are experiment records, not a single command-line benchmark runner. The BoostTrack OSNet checkpoint is not included, so reproducing that run requires obtaining the same checkpoint. The BoT-SORT run is recorded through its result CSVs; its notebook and configuration were not part of the supplied artifacts.

## Scope and limitations

These are results on the **MOT20 training split**, not MOT20 test leaderboard scores. The detectors use pretrained weights without MOT20 fine-tuning. Scores reflect the recorded single runs and tracker configurations, and some tracker implementations and ReID settings differ. FPS is taken from each run's own summary, so compare speed with that context. BoT-SORT's environment and configuration were not recorded in the supplied artifacts, so its timing is the least documented of the five trackers. `R@1` and `R@5` fields in some raw summaries are blank and are not reported here.

## Acknowledgments

[MOTChallenge / MOT20](https://motchallenge.net/data/MOT20/), [Ultralytics](https://github.com/ultralytics/ultralytics), [TrackEval](https://github.com/JonathonLuiten/TrackEval), and [BoxMOT](https://github.com/mikel-brostrom/boxmot).
