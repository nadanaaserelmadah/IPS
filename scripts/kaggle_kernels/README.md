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

## Audit paper (released models we did not train; see `docs/AUDIT_PAPER_PLAN.md`)

| Folder | Kernel | What it does |
|---|---|---|
| `kernel_ulvio` | `amrelmadah/ulvio-kitti-infer` | UL-VIO (ECCV'24) with the released `model_ulvio_v1.pth` on KITTI 05/07/10 and 08, the authors' test protocol, no test-time adaptation (GPU) |
| `kernel_nasvio` | `moha70/nasvio-kitti-infer` | NASVIO (ICASSP'23) with both released checkpoints (flops, latency) on KITTI 05/07/10 and 08, replicating the authors' `test.py` (GPU) |

| `kernel_kgeo_test` | `mhaggag/kitti-geo-direction-test` | `kernel_kgeo` unchanged except the sequence list (05/07/10), to rebuild the test-drive geometry that gate v3 needs (CPU) |

| `e4_lagfix_s42`, `e4_lagfix_s2`, `e4_lagfix_s3` | `hassanhaggag/vift-kitti-e4-lagfix-s42`, `-s2`, `dragonvic/vift-kitti-e4-lagfix-s3` | E4: the clean-protocol `kno08val*` kernels with the same seed, fed the lag-corrected `imus/*.mat` from `scripts/audit/build_fixed_imus.py` (dataset `*/kitti-imus-lagfix`). s2 v1 failed before training: Google Drive refused the encoder download ("many accesses") while s42 fetched it; v2 attached dataset `hassanhaggag/vsvio-encoder`, but its first version held the wrong file (FlowNet `flownetc_EPE1.766.tar`, Drive id `1x2zh…`), so v2 stopped at the weight check. Dataset version 2 holds the right file (Drive id `18tfw94asjStribqA8SLvaacrUwT86HsT`, 193,891,066 bytes, sha256 `3cd29a66…1ffd`), used from s2 v3 on |
| `e4_base_s4`/`e4_lagfix_s4`, `_s5`, `_s6` | `hassanhaggag/…-s4`, `dragonvic/…-s5`, `mhaggag/…-s6` (`vift-kitti-no08-val08-mirror-sN` and `vift-kitti-e4-lagfix-sN`) | E4 extension (2026-11-10), three more seed pairs because three seeds left the network-only effect unsettled. Each pair runs on one account with the same code dataset; the baseline is `kno08val_hassanhaggag-s2` with only the seed changed, the lag-fix is `e4_lagfix_s2` with only the seed changed. Every account has its own `*/vsvio-encoder` (same sha256) so nothing depends on Google Drive |
| `e5_vsvio` (master), `e5_vsvio_smoke`, `e5_vsvio_s0`/`_s1`/`_s2` | `hagagovic/vsvio-e5-smoke`, `hagagovic/vsvio-e5-s0`, `elsamadony/vsvio-e5-s1`, `abeerelsamadony/vsvio-e5-s2` | E5 (2026-11-10): VS-VIO (commit `27b06b3`) retrained end to end under the clean protocol (train 00 01 02 04 06 09, 08 held out), three seeds. The authors' `train.py` is patched as text into `train_e5.py` (the diff is saved in the output). Changes: (1) the split; (2) epochs 81–99 dump poses for 08/05/07/10, with seeded Gumbel sampling, so any selection rule can be applied afterwards (`scripts/audit/e5_select.py`); (3) one `resume.pth` plus a clean stop before the 12 h limit, continued by a new kernel with `kernel_sources`; (4) a float16 frame cache and both T4s via the authors' own `--gpu_ids` (speed only). Each account has `*/vsvio-e5-inputs` (repo `data/imus` + `flownets_bn_EPE2.459.pth.tar`, sha256 `77bbee0e…c9c8`). Smoke v3 timing: 1.2 s/iteration on 2×T4, about 17 min per epoch and 4 min per evaluation, so about 30 h per seed (three sessions). The copies differ only in the seed default (smoke: `E5_SMOKE` default 1). Later sessions are launched by `e5_vsvio/next_session.sh`: session 2 runs on the same account and mounts session 1's output (`kernel_sources`). Session 3 runs on another account because of the weekly GPU quota (s0 → `amrelmadah`, s1 → `moha70`, s2 → `norhanelmadah`). It reads session 2's `e5/` folder, uploaded as `<account>/vsvio-e5-s<seed>-resume`. Kernels are named `vsvio-e5-s<seed>-c<session>`. Notes from the run:
<ul>
<li>Session 2 of s1 was SIGKILLed (rc −9) at the start of epoch 79, 642 min in, with nothing in the log. The epoch-78 `resume.pth` was intact. Session 3 continued from it (`ALLOW_KILLED=1`). Since then the per-epoch line also logs host memory.</li>
<li>Kaggle unpacks a dataset holding a single folder at the dataset root, so the `e5/` name is lost. Version 1 of the three `-c3` kernels looked for a folder named `e5`, found none, and started from scratch. Version 2 matches the previous session on its files (`status.json` and `resume.pth`). It also refuses to start when `E5_EXPECT_RESUME=1` (set by `next_session.sh`) and no previous session is found.</li>
<li>`kaggle kernels logs -f <kernel>` streams the log of a running kernel.</li>
</ul> |
| `e5_bncheck` | `hagagovic/vsvio-e5-bn-recal` | E5 check (2026-11-12). Takes the best-on-08 checkpoints of seeds 0/1/2 and the released `vf_512_if_256_3e-05.model` (dataset `hagagovic/vsvio-e5-bncheck`). On one GPU and without changing any weight, it evaluates each one as saved, then again after re-estimating every BatchNorm statistic over 150 clean training batches of 16. Each evaluation is repeated for 3 gumbel seeds. Report: `scripts/audit/e5_bncheck_report.py` |
| `vsvio_e5_1gpu_s0_c*` | `nadaelmadah/vsvio-e5-1gpu-s0-c1`, then `-c2`… (nadaelmadah, nadanasserelmadah, sallyhaggag) | E5 on one GPU (2026-11-12). Seed 0 with the authors' default `--gpu_ids 0` (batch 16 on one T4), otherwise identical to E5. It separates training-time batch normalisation over 8 samples from the clean protocol, because the BatchNorm check ruled out the saved statistics. The master gained `E5_GPU_IDS` (the default `all` keeps E5 as it was, and the value is recorded in `status.json` and checked on resume). About 33 min per epoch, so six sessions. `e5_vsvio/chain.sh` launches each session when the previous one ends. Run it detached, e.g. `PREFIX=vsvio-e5-1gpu GPU_IDS=0 nohup chain.sh 0 nadaelmadah/vsvio-e5-1gpu-s0-c1 WORK nadaelmadah:2 nadanasserelmadah:3 nadanasserelmadah:4 sallyhaggag:5 sallyhaggag:6`. Finished 2026-11-14 in five sessions (c1–c5; at 30–36 min per epoch, c5 covered epochs 81–99). The result is the same as two GPUs (`results_cache/2026-11-14/`). Notes from the run:<ul><li>On one GPU the process memory stayed flat (about 6.4 GB). The leak that killed two-GPU sessions comes from DataParallel.</li><li>The Mac rebooted itself overnight twice, which kills `chain.sh` (sessions 3 and 5 were launched by hand). Wiping `/private/tmp` also lost the scoring inputs, which now live in `~/Library/Caches/ips_e5_chain/score`.</li><li>Downloading the 580 MB `resume.pth` broke off mid-file several times. `next_session.sh` now skips that download for a same-account session and retries it for a cross-account one.</li></ul> |
| `vsvio_e5_auth1gpu_s0_c*` | `sallyhaggag/vsvio-e5-auth1gpu-s0-c1`, then `-c2`… (sallyhaggag, hassanhaggag, dragonvic, mhaggag) | Control for the released VS-VIO checkpoint (2026-11-14). Same as the one-GPU run, but with the authors' training set including 08 (`E5_SPLIT=authors`; the master skips its train_seq patch, and evaluation is unchanged). Applying the authors' rule (best epoch on 05/07/10) tells whether their released code reproduces the 2.35 of their released checkpoint. `PREFIX=vsvio-e5-auth1gpu GPU_IDS=0 SPLIT=authors chain.sh …`. Seed 0 finished 2026-11-18 (`dragonvic/…-s0-c6`): with the authors' rule it gives 2.21, against 2.35 for the released checkpoint. Seeds 1 and 2 ran in parallel and finished 2026-11-21 (`hagagovic/vsvio-e5-auth1gpu-s{1,2}-c7`, seven sessions each); with the authors' rule the three seeds give 2.32±0.13 on the network and 1.31±0.04 after the gate (`results_cache/2026-11-21/`). Both LaunchAgents were removed on that day. They had started the same day in parallel: s1 on elsamadony → amrelmadah → norhanelmadah, s2 on abeerelsamadony → moha70 → mhaggag, with hagagovic as spare. Supervisor state files: `~/Library/Caches/ips_e5_chain/auth1gpu_s{1,2}.state` |

### Surviving reboots (LaunchAgent, 2026-11-14)

The Mac reboots itself overnight, which kills `chain.sh`. The LaunchAgent `~/Library/LaunchAgents/com.ips.e5chain.plist` runs `e5_vsvio/chain_supervisor.sh` at login and every 30 minutes. The supervisor:
- does nothing while the chain runs, or once the log says the run is done;
- otherwise finds the last planned kernel that exists on Kaggle and restarts `chain.sh` from it, with `caffeinate`.

The run is described by a state file (`~/Library/Caches/ips_e5_chain/auth1gpu_s0.state`: seed, prefix, GPU, split, first kernel, plan, log).

macOS (TCC) blocks launchd agents from `~/Documents`, so the agent runs copies of `chain.sh`, `next_session.sh`, `chain_supervisor.sh` and `vsvio_e5_train.py` in `~/Library/Caches/ips_e5_chain/bin`. Re-copy them after editing the originals. Kernel folders that a launchd-started chain creates land in `~/Library/Caches/ips_e5_chain/`, not here, so copy them back when the run ends.

`chain.sh` was also fixed on 2026-11-21: after the last planned session it now logs `run done after` once the run is complete. Before, the supervisor kept restarting a finished chain every 30 minutes.

To remove the agent:

```bash
launchctl bootout gui/$(id -u)/com.ips.e5chain && rm ~/Library/LaunchAgents/com.ips.e5chain.plist
```

Seed 0 of the authors' split finished 2026-11-18 and that agent was removed; seeds 1 and 2 finished 2026-11-21 and their agents (`com.ips.e5chain.s1` and `.s2`) were removed too. While they ran, each had one agent (`com.ips.e5chain.s1` and `.s2`, state files `auth1gpu_s{1,2}.state`). Remove both when the two runs are done:

```bash
for s in 1 2; do launchctl bootout gui/$(id -u)/com.ips.e5chain.s$s; rm ~/Library/LaunchAgents/com.ips.e5chain.s$s.plist; done
```
| `vift_auth08_s42`, `_s2`, `_s3` | `hassanhaggag/vift-kitti-auth08-testsel-s42`, `dragonvic/…-s2`, `mhaggag/…-s3` | VIFT under the protocol of its released code (2026-11-21): the clean-protocol `kno08val_hassanhaggag-s2` script with three defaults changed (training adds 08, `KR_VAL=05,07,10` so validation loss picks the checkpoint on the test drives, seed). Paired with the clean `base_s*` runs of E4. About 2 h per seed. After sensor substitution it gives 1.20±0.07 against 1.56±0.02 for the clean runs (`results_cache/2026-11-22/`) |

Scoring: `scripts/audit/score_released.py` (oracle swap, gyro, accelerometer length, frozen gate v3).
