# IPS — Indoor Positioning with Visual-Inertial Transformers

Research project extending **VIFT** ([arXiv:2409.08769](https://arxiv.org/abs/2409.08769)) toward **Indoor Positioning System (IPS)** using **camera + IMU**.

## Quick start (Phase 1 — EuRoC)

```powershell
.\scripts\phase1\download_euroc.ps1
.\scripts\phase1\verify_gt_ate.py
.\scripts\phase1\test_dataloader.ps1
```

See **[docs/PHASE1.md](docs/PHASE1.md)**.

## Quick start (Phase 0 on Kaggle)

You attached **IPS** + **EuRoC** datasets → open **[docs/PHASE0_KAGGLE.md](docs/PHASE0_KAGGLE.md)** or notebook **[notebooks/kaggle_phase0.ipynb](notebooks/kaggle_phase0.ipynb)**.

## Quick start (Phase 0 local)

```powershell
cd c:\Users\LORD\Desktop\IPS
.\scripts\phase0\setup_env.ps1
.\scripts\phase0\smoke_test.ps1
```

See **[docs/SETUP.md](docs/SETUP.md)** and **[docs/PHASE0.md](docs/PHASE0.md)** for full steps (KITTI, pretrained weights, paper training).

## Repository layout

```
IPS/
├── RESEARCH_PLAN.md      # Master research plan
├── docs/
│   ├── SETUP.md          # Environment & data setup
│   └── PHASE0.md         # Phase 0 checklist & status
├── scripts/phase0/       # Setup & verification scripts
└── third_party/VIFT/     # VIFT baseline (upstream)
```

## Master plan

[RESEARCH_PLAN.md](RESEARCH_PLAN.md)
