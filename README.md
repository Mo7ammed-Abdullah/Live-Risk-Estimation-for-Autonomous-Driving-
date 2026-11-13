# Vision-only driving risk estimation (SegFormer + UniDepthV2)

Camera-only pipeline on nuScenes front-camera images. SegFormer-B2 labels every pixel, UniDepthV2-small gives metric depth,
and a fixed formula fuses them into a risk map. The image is split into far / mid / near regions and the most urgent one is reported.
Based on "Risk Estimation via Semantic-Depth Fusion" (Chakraborty et al.), with metric depth and a max-per-region rule.

The score is a proximity-weighted risk score (class and distance only). It is not a collision probability.

## Risk formula (`code/risk.py`)
    R(x, y)  = w_class(x, y) * exp(-0.05 * distance_m(x, y))    # distance clipped to 80 m
    U_i      = alpha_i * max of R over region i                  # alpha: far 0.4, mid 0.7, near 1.0
    decision = region with the highest U_i
Class weights (assumed, not fitted): person/rider 1.0, motorcycle/bicycle 0.9, car/truck/bus/train 0.8, other 0.1.
Segmentation and depth are fine-tuned separately; the fusion has no trained parameters.

## Results (locked test set: 713 images, split by drive)
| | |
|---|---|
| Depth vs LiDAR | AbsRel 0.071, RMSE 4.54 m |
| Segmentation vs pseudo-labels | mIoU 0.568 |
| Real objects found (3D boxes) | 89.4% |
| Decision accuracy vs 3D boxes | 89.7% (max rule), 66.1% (base paper's mean rule), 68.9% (always "near") |
| Speed (RTX 4060 laptop, 640x320) | about 13 FPS |

In the ablation the decision depends mainly on class and image position; depth adds no accuracy.
The segmentation labels are pseudo-labels (Mask2Former) and the 3D-box truth uses the same weights and formula as the system.
Full numbers: `results/MAIN_RESULTS/`.

## Extra experiments (supervisor feedback; validation unless test is stated)
| Idea | Script / option | Result |
|---|---|---|
| Focal + class-balanced loss for SegFormer | `train_seg.py --loss focal --cb_beta 0.9999999` | rare-class recall up (bicycle acc 0.70 -> 0.85), but risk mIoU 0.630 -> 0.599 and decision accuracy 79.5% -> 68.5% (both with pretrained depth); not adopted |
| Distance-weighted depth loss | `train_depth.py --far_weight 3` | 50-80 m AbsRel 0.169 -> 0.156, 0-20 m 0.057 -> 0.066, decision accuracy unchanged (80.6% vs 80.3%); not adopted |
| Learned fusion trained on time-to-collision labels | `build_ttc_labels.py`, `cache_fusion_features.py`, `train_fusion.py` | better on validation (1 drive) but no better than chance on test (AUROC 0.47); too few danger examples (72 objects, 7 drives) and labels partly caused by our own turning; not adopted |
| Camera time-to-collision (two frames, existing models) | `build_motion_labels.py`, `estimate_ttc.py` | test: median TTC error 0.54 s within 20 m (1.16 s overall); finds objects reached within 4 s with AUROC 0.81 vs 0.61 for the fixed formula |
| Automatic TTC (system finds and follows objects itself; shown in `run.py`) | `ttc_tracker.py`, `eval_auto_ttc.py`, `diag_auto_ttc.py` | test: AUROC 0.48 (chance) because only ~1 in 5 approaching objects gets a TTC at nuScenes' 2 frames/s; objects that do get one are ranked well (val AUROC 0.85). Predicted matching (added after the test run) raises coverage 26% -> 33% on val. Bottleneck: linking objects 0.5 s apart, segmentation misses beyond ~35 m, merged objects |
| Automatic TTC at 12 Hz (also feeds the CAM_FRONT sweeps between keyframes) | `eval_auto_ttc_sweeps.py`, `replay_auto_ttc.py` | test: dangerous objects with a TTC 19% -> 43%, AP 0.37 -> 0.50, AUROC 0.49 -> 0.61, median error 1.23 -> 0.99 s (same tracker, 2 Hz vs 12 Hz; 0.8 s history chosen on val). Needs the sweeps in `dataset/sweeps/CAM_FRONT` (nuScenes camera blobs, part 1) |

Labels: `box_gt/motion.json` (from `build_motion_labels.py`) holds, per object, the true closing speed and TTC from real
nuScenes positions, and the real smallest gap to our car over the next 4 s. `box_gt/ttc.json` is the older constant-velocity
label used for the learned-fusion test. The camera TTC uses the nuScenes boxes to know which pixels belong to which object;
depth and size are measured by the models only.

    python code/build_motion_labels.py                    # CPU, ~1 min
    python code/estimate_ttc.py --split val               # GPU, ~2 min (test only with --split test --final)

## Web app (try it in the browser)
Live demo: **https://huggingface.co/spaces/Mohammed-Abdullah/Driving-risk-estimation** (free shared GPU). Upload a dashcam
photo or a short clip of the road ahead (no dashboard or bonnet in view); it shows the risk decision, each object's
distance and, for videos, the time to collision per object (CSV download).

Run the same app yourself (`app.py`, after the Setup below):

    python -m pip install gradio==6.31.0 imageio imageio-ffmpeg
    python app.py              # http://127.0.0.1:7860  (add --share for a temporary public link)

The trained weights (about 245 MB) are too large for GitHub. `app.py` uses `runs/` if you trained the models yourself,
otherwise it downloads them once from the Hugging Face Space into `weights/`.

## Setup
    python -m venv venv311
    venv311\Scripts\activate
    pip install -r requirements.txt
    pip install -e UniDepth --no-deps          # run again if you move the project folder
Pretrained models download from Hugging Face on first use. Datasets are not in this repository; the trained weights are on the Hugging Face Space (downloaded automatically by `app.py`).

## Run on a video
Set `VIDEO_PATH` (and optionally `EVERY`, `CROP_BOTTOM`, `SHOW_WINDOW`) at the top of `run.py`, then:

    python run.py

Output: `demo_out/demo.mp4` and `demo_out/risk_per_frame.csv`. For a 60 FPS video use `EVERY = 6`, `VIDEO_FPS = 10`.
Videos from other cameras are a visual demo only (unknown camera, approximate depth).

## Reproduce
Needs nuScenes in `dataset/CAM_FRONT`, `dataset/LIDAR_TOP` and `v1.0-trainval_meta/v1.0-trainval`. Run from the project folder:

    python code/generate_seg_gt.py            # pseudo-labels (add --out seg_gt_native for the training labels)
    python code/make_splits.py                # train/val/test split by drive
    python code/generate_depth_gt.py          # LiDAR depth
    python code/generate_box_gt.py            # 3D-box truth
    python code/train_seg.py --name seg_b2_native --seg_dir seg_gt_native --counts_csv splits_native/per_image_class_pixels.csv
    python code/train_depth.py --name depth_s
    python code/eval_risk.py --split test --final --depth_model runs/depth_s/best.pt     # also eval_depth, eval_seg, eval_box_risk, ablate_risk, bench_speed

Use `--split val` while developing; the test split only opens with `--final`.

## Layout
`code/` scripts (commented) | `run.py` video demo | `app.py` web app | `runs/` trained models | `results/` outputs | `UniDepth/` depth library | `demo_in/`, `demo_out/`

UniDepth is CC BY-NC 4.0 (non-commercial). Check the licenses of SegFormer, Mask2Former and nuScenes before sharing models or data.
