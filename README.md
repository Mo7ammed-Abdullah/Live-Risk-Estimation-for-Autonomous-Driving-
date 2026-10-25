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

## Setup
    python -m venv venv311
    venv311\Scripts\activate
    pip install -r requirements.txt
    pip install -e UniDepth --no-deps
Pretrained models download from Hugging Face on first use. Datasets and trained weights are not in this repository.

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
`code/` scripts (commented) | `run.py` video demo | `runs/` trained models | `results/` outputs | `UniDepth/` depth library | `demo_in/`, `demo_out/`

UniDepth is CC BY-NC 4.0 (non-commercial). Check the licenses of SegFormer, Mask2Former and nuScenes before sharing models or data.
