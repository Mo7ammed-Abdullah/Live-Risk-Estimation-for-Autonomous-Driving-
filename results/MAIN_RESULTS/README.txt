MAIN RESULTS OF THE MODEL (nuScenes front camera, locked TEST set, 713 images)
Models: SegFormer-B2 (runs\seg_b2_native\best) + UniDepthV2-small (runs\depth_s\best.pt)

1_depth_test.json                  Depth error vs LiDAR (AbsRel 0.071, RMSE 4.54 m)
2_segmentation_test.json           Segmentation vs pseudo-labels (mIoU 0.568, pixel acc 0.942)
3_segmentation_vs_3D_boxes_test.json   Real objects found, checked with 3D boxes (89.4%)
4_fusion_vs_oracle_test.json       Risk map vs the same formula with ground truth (decision agreement 97.2%)
5_fusion_vs_3D_boxes_test.json     Risk vs real objects; decision accuracy 89.7% (max rule), 66.1% (mean rule), baseline 68.9%
6_ablation_test.json               Which choices matter (semantic only 92.6%, depth only 68.9%)
7_speed.json                       Speed at several input sizes (about 13 FPS at 640x320)
8_object_table_test.csv            One row per real object (true class, distance, true and system risk)

Everything else is in ..\dev_validation_and_old\ (validation runs, other models, the older segmentation run).
Re-running a script writes a new file into ..\ (the results folder), not into this one.
