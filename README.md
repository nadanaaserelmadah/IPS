# Observability-gated fusion for learned visual–inertial odometry

Code, scripts and cached results that accompany the manuscript

> **Observability-gated fusion of sensor measurements and a neural network improves learned visual–inertial odometry**
> Nada El Madah, Magdy Zakaria, Samir Elmougy, Shahenda Sarhan, Mohamed Handosa and Rasha Elsayed Sakr
> Department of Computer Science, Faculty of Computers and Information, Mansoura University, Egypt.

The work asks which part of the motion a learned visual–inertial odometry (VIO) network gets wrong, and then lets the inertial sensor measure what it can observe while a frozen network supplies the rest.

- **Oracle swap** — replace one predicted motion component (rotation, direction, length) at a time by ground truth to see which one fails.
- **Observability gate** — for each pair of frames and each component, use the sensor measurement where it is observable and the frozen network elsewhere. No network is retrained for the gate.
- **Datasets** — KITTI odometry, EuRoC MAV and TUM-VI. The base networks are VIFT (Kurt et al., [arXiv:2409.08769](https://arxiv.org/abs/2409.08769)), PI-VIFT (a physics-informed variant used here) and released VS-VIO, ULVIO and NASVIO models.

Every number in the manuscript comes from the scripts and cached results in this repository. Failures are reported with the successes.

## Repository layout

| Path | Content |
|---|---|
| `scripts/gate/` | The observability gate (versions 3 and 4), the final run, and the observability plots. |
| `scripts/audit/` | Oracle swap and scoring of released models, timing-lag and gyroscope-bias studies, condition-number studies, the E5/E6 runs. |
| `scripts/kitti/`, `scripts/euroc/`, `scripts/tum_vi/` | Per-dataset geometry, gyroscope and accelerometer estimators, scoring, and figures. |
| `scripts/pi_vift/`, `scripts/euroc_kernels/` | PI-VIFT utilities and its EuRoC fine-tuning scripts. |
| `scripts/phase0/` … `scripts/phase6/` | The experiments in the order they were run (environment and KITTI baseline, EuRoC data and training, fine-tuning, TUM-VI, later extensions). Kept as run; later folders build on earlier ones. |
| `scripts/scireports/` | Scripts that draw Figures 3 and 4 of the manuscript (see `RUN_FIGURES.md`). |
| `scripts/kaggle_kernels/` | Every Kaggle kernel used for training and inference, one folder each, with a README that says which result each kernel produced. |
| `results_cache/` | Cached intermediate results, by date: audit logs, pose dumps, geometry files. |
| `third_party/VIFT/` | Copy of the VIFT code (Kurt et al.) that the base networks and data loaders use. |
| `configs/`, `notebooks/` | Data configuration and the first Kaggle notebook. |

`scripts/` keeps its folder names on purpose: the Kaggle kernels and the notes in `scripts/kaggle_kernels/README.md` refer to these paths.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r third_party/VIFT/requirements.txt
cp .env.example third_party/VIFT/.env      # then set PROJECT_ROOT to the absolute path of third_party/VIFT
```

Training and inference of the networks ran on Kaggle (a T4 GPU). A GPU is not needed for the gate or the scoring scripts once the pose dumps exist.

## Data

All experiments use public datasets, which are not stored here:

- KITTI odometry — <https://www.cvlibs.net/datasets/kitti>
- EuRoC MAV — <https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>
- TUM-VI — <https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset>

The released model weights come from the cited papers. Download helpers are in `scripts/phase0/`, `scripts/phase1/`, `scripts/phase4/` and `scripts/kitti/fetch_oxts.py`.

## Reproducing the results

1. **Pose dumps.** Run the kernel that matches the result in `scripts/kaggle_kernels/` (its README lists the dataset and output of each). Outputs used in the manuscript are cached in `results_cache/`.
2. **Oracle swap and gate scores.** `scripts/audit/score_released.py` scores released models (oracle swap, gyroscope, accelerometer length, frozen gate). `scripts/gate/run_final.py` and `scripts/gate/observability_gate.py` run the gate; `scripts/gate/gate_v4.py` is the version with online trust.
3. **Figures.** Follow `scripts/scireports/RUN_FIGURES.md`.

## Notes

- The gyroscope-bias work is a separate study in `scripts/audit/` and is not part of the manuscript.
- Reported errors are means over three seeds; ± is the standard deviation over the seeds.

## Citation and licence

Please cite the manuscript and this repository (see `CITATION.cff`). The code in this repository is released under the MIT licence (`LICENSE`). `third_party/VIFT/` is a copy of code by Kurt et al. and keeps its original terms; check the upstream repository before reuse.
