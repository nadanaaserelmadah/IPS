# Kaggle kernels behind the results

Exact copies of the kernel scripts and their `kernel-metadata.json`, as pushed to Kaggle. Before this they lived only in the session scratchpad. Each folder can be re-pushed with:

```bash
KAGGLE_CONFIG_DIR=~/.kaggle-accounts/<account> kaggle kernels push -p scripts/kaggle_kernels/<folder>
```

When pushing from another account, change the `id` and `dataset_sources` in `kernel-metadata.json` first. Copies of a script in sibling folders differ only in the seed, the checkpoint name or the time budget.

## TUM-VI (third dataset; RESULTS: "TUM-VI as the third gate dataset" onwards)

| Folder | Kernel | What it does |
|---|---|---|
| `kernel_tumft` | template | Fine-tunes the fixed-GT EuRoC PI-VIFT on room1–4 (15 epochs) and tests zero-shot and last on room5/6. Streams room3–6 from the TUM server, builds an EuRoC-layout shadow tree, and runs a fatal per-room physics/image check |
| `kernel_tumft_amrelmadah` | `amrelmadah/pivift-tumvi-ft` | seed 123, init `pivift_s123_last` |
| `kernel_tumft_moha70` | `moha70/pivift-tumvi-ft` | seed 42, init `pivift_s42_last`, training budget 6.5 h |
| `kernel_tumft_norhanelmadah` | `norhanelmadah/pivift-tumvi-ft` | seed 7, init `pivift_s7_best` |
| `kernel_tuminfer` | `amrelmadah/pivift-tumvi-infer14` | Zero-shot predictions on room1–4 from three EuRoC checkpoints (design data for gate v4) |
| `kernel_orbslam3_tumvi` | `mhaggag/orbslam3-tumvi` | ORB-SLAM3 v1.0 mono/stereo-inertial on room5/6, with and without loop closure (CPU) |

The local geometry for the TUM-VI rooms is `scripts/tum_vi/geo_rooms.py` (shadow tree, window timestamps, two-view direction).

## EuRoC

| Folder | Kernel | What it does |
|---|---|---|
| `pull_qfix` | `hagagovic/pivift-qfix-baseline` | PI-VIFT stage 1 on the quaternion-fixed GT (12 epochs; best, bestate and last tested) |
| `kernel_egeo` | `moha70/euroc-geo-direction` | Two-view geometric direction on the EuRoC test sequences (CPU) |
| `kernel_orbslam3` | `nadaelmadah/orbslam3-euroc` | ORB-SLAM3 v1.0 on MH_05, V2_02, V1_01 and V1_02 (CPU) |

## KITTI

| Folder | Kernel | What it does |
|---|---|---|
| `kno08val*` | `*/vift-kitti-no08-val08-mirror*` | Clean protocol: train without 08, select the checkpoint on 08 (three seeds) |
| `kernel_v08pred_*` | `*/vift-kitti-v08-pred08` | Predictions on held-out 08 from the no-08 models (gate v3/v4 design data) |
| `kernel_kgeo` | `mhaggag/kitti-geo-direction` | Two-view direction on KITTI, incl. 08 (CPU) |
| `pull_final`, `pull_mirror`, `pull_hassanhaggag`, `pull_dragonvic` | `*/vift-kitti-*` | KITTI training and dump kernels for the recipe and mirror runs (with and without 08) |

## Cross-model

| Folder | Kernel | What it does |
|---|---|---|
| `kernel_vsvio` | `hassanhaggag/vsvio-kitti-infer` | VS-VIO with the released weights on KITTI 05/07/10, policy and full-vision modes (CPU; too slow) |
| `kernel_vsvio_gpu` | `amrelmadah/vsvio-kitti-infer-gpu` | The same on GPU; this run produced the reported numbers |
