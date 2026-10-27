# Research Plan: Causal Visual-Inertial Transformer for Indoor IPS

**Project:** Indoor Positioning System (IPS) using Camera + IMU  
**Baseline reference:** VIFT — [arXiv:2409.08769](https://arxiv.org/abs/2409.08769) · [GitHub](https://github.com/ybkurt/VIFT)  
**Target contribution:** A stronger follow-up paper addressing VIFT limitations with indoor evaluation and IPS-oriented metrics  
**Last updated:** 2026-05-19

---

## 1. Executive Summary

### 1.1 Goal

Build and publish a method that:

1. Fuses **monocular RGB** and **IMU** (accelerometer + gyroscope) with a **causal Transformer**.
2. Estimates **egomotion** (relative 6-DoF pose between frames).
3. Integrates motion into a trajectory for **Indoor Positioning System (IPS)** on floor-level coordinates.
4. Outperforms or clearly extends **VIFT** on indoor-relevant benchmarks and IPS metrics.

### 1.2 Working title (draft)

**"Causal Visual-Inertial Transformer for Smartphone Indoor Positioning"**

Alternative titles:

- *Learned Visual-Inertial Odometry with Map-Aware Indoor Positioning*
- *End-to-End Causal Fusion of RGB and IMU for Indoor IPS*

### 1.3 One-line claim (for the paper)

> We propose an end-to-end (or hybrid) causal Transformer that fuses camera and IMU for metric indoor trajectories, evaluated on standard VIO datasets and indoor IPS benchmarks—with comparisons to classical VIO, learned baselines, and IMU-only navigation.

---

## 2. Problem Statement

### 2.1 Pipeline

```
RGB frames (10–30 Hz)  ──► Visual encoder
                              │
IMU (100–200 Hz)       ──► Inertial encoder
                              │
                              ▼
                    Causal Transformer (fusion)
                              │
                              ▼
                    Relative pose ΔT ∈ SE(3)
                              │
                              ▼
                    Trajectory integration
                              │
                              ▼
              IPS: (x, y, floor) [+ optional map matching]
```

### 2.2 Why not GPS indoors?

Indoor environments block or degrade GPS. **Camera + IMU** on smartphones is infrastructure-free and aligns with pedestrian IPS.

### 2.3 Relationship to VIFT

| Aspect | VIFT | This project |
|--------|------|--------------|
| Sensors | Camera + IMU | Camera + IMU |
| Fusion | Causal Transformer on latents | Causal Transformer (extend / replace) |
| Training data | KITTI (outdoor driving) | EuRoC, TUM-VI, indoor custom |
| Task framing | Relative VO | **Indoor IPS** (metric position on floor) |
| Encoders | Frozen FlowNet + 1D CNN | **Trainable** modern backbone (target) |
| Evaluation | KITTI trel / rrel | ATE, RPE + **IPS error (meters)** |

---

## 3. VIFT Limitations to Address (Gap Analysis)

Use these as motivation in the related work / introduction.

### 3.1 Methodological gaps

- [ ] **Frozen encoders** — only the fusion head is trained on pre-cached latents (Visual-Selective-VIO).
- [ ] **Not true end-to-end** — two-stage pipeline (latent caching → Transformer training).
- [ ] **Incremental novelty** — RNN → Transformer + RPMG on existing features.
- [ ] **Short temporal window** — best results at seq_len=11; seq_len=65 degrades performance.
- [ ] **Euler angles** in forward pass; manifold loss (RPMG) mainly in backward pass.
- [ ] **No uncertainty** — no adaptive trust between vision and IMU under failure.

### 3.2 Evaluation gaps

- [ ] **KITTI only** — outdoor, forward-dominated motion; not indoor pedestrian.
- [ ] **No EuRoC / TUM-VI** — standard indoor / aggressive VIO benchmarks missing.
- [ ] **No IPS metrics** — no floor plan, no meter-level positioning error in buildings.
- [ ] **Weak classical baselines** — limited fair comparison to VINS-Mono, ORB-SLAM3-VI, DM-VIO.
- [ ] **Workshop venue** — ECCV 2024 VCAD workshop, not main conference tier.

### 3.3 Application gaps

- [ ] **No smartphone / pedestrian** deployment story.
- [ ] **No real-time report** — causal design claimed but latency/FPS not measured.
- [ ] **No map matching / loop closure** — drift unbounded for long IPS sessions.
- [ ] **No cross-building generalization** experiment.

---

## 4. Proposed Contributions (pick 2–3 for the paper)

Check the boxes you will actually implement and emphasize in the abstract.

### Contribution A — Indoor IPS benchmark & task

- [ ] Define evaluation protocol: VIO (ATE, RPE) + IPS (2D floor error after integration).
- [ ] Report on **EuRoC**, **TUM-VI**, and at least one **indoor pedestrian** source (RoNIN-style or custom building).
- [ ] Release code + config to reproduce tables.

### Contribution B — End-to-end causal Visual-Inertial Transformer

- [ ] Replace frozen FlowNet with **trainable** backbone (ViT-Small, ResNet, or lightweight CNN for mobile).
- [ ] Full-rate or multi-rate IMU encoder (TCN / 1D Transformer) between frames.
- [ ] Causal masked Transformer for online inference.
- [ ] Continuous rotation representation (6D rotation / quaternion), not Euler-only.

### Contribution C — IPS layer (differentiator vs VIFT)

- [ ] Integrate Δpose → global trajectory.
- [ ] Optional **floor plan map matching** or particle filter to bound drift.
- [ ] Report IPS error (m) at 1 min, 5 min, 10 min walk durations.

### Contribution D — Uncertainty-aware fusion (optional, high impact)

- [ ] Predict per-step confidence (vision vs IMU).
- [ ] Down-weight vision under blur / occlusion / low texture (typical indoors).

### Contribution E — Hybrid learning + optimization (optional, for RA-L / IROS)

- [ ] Network predicts corrections / biases; **factor graph** or lightweight BA refines trajectory.

---

## 5. Technical Architecture (target design)

### 5.1 Module diagram

```
┌─────────────────────────────────────────────────────────────┐
│  INPUTS (synchronized window t−N … t)                       │
│  • Images I_{t−N}, …, I_t                                   │
│  • IMU: accel + gyro [t−N, t] at 100–200 Hz                 │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────┐                         ┌───────────────┐
│ Visual enc.   │                         │ Inertial enc. │
│ (CNN / ViT)   │                         │ (TCN / 1D-CNN)│
└───────┬───────┘                         └───────┬───────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              ▼
                ┌─────────────────────────┐
                │ Causal Transformer      │
                │ (self-attn + FFN)       │
                │ + positional encoding   │
                └────────────┬────────────┘
                             ▼
                ┌─────────────────────────┐
                │ Pose head → ΔT, σ²     │  (optional uncertainty)
                └────────────┬────────────┘
                             ▼
                ┌─────────────────────────┐
                │ Trajectory integrator   │
                └────────────┬────────────┘
                             ▼
                ┌─────────────────────────┐
                │ IPS: map match (opt.)   │
                └─────────────────────────┘
```

### 5.2 Design decisions (fill as you decide)

| Decision | Options | Choice (TBD) |
|----------|---------|--------------|
| Visual backbone | ResNet-18, ViT-S/16, MobileNet | |
| IMU representation | Raw windows, integration features, learned preintegration | |
| Sequence length N | 8, 11, 16, 32 | |
| Rotation output | 6D rot, quaternion, se(3) | |
| Loss | L1 pose, huber, RPMG, geodesic rotation | |
| Training | From scratch, ImageNet pretrain, VIFT latent warm-start | |

### 5.3 Baseline code strategy

| Component | Source |
|-----------|--------|
| Project scaffold | Fork [VIFT](https://github.com/ybkurt/VIFT) (Hydra + Lightning) |
| Encoders (interim) | [Visual-Selective-VIO](https://github.com/mingyuyng/Visual-Selective-VIO) weights |
| Classical VIO | VINS-Mono, ORB-SLAM3-VI, DM-VIO |
| IMU-only baseline | RoNIN (indoor pedestrian) |

---

## 6. Datasets & Metrics

### 6.1 Datasets

| Dataset | Role | Notes |
|---------|------|-------|
| **EuRoC MAV** | Train / val / test VIO | Indoor rooms, camera + IMU + GT |
| **TUM-VI** | Test generalization | Challenging VI sequences |
| **KITTI Odometry** | Optional pretrain | Same split as VIFT for fair comparison |
| **RoNIN** (or subset) | Indoor pedestrian IMU/GT | IPS-style trajectories in buildings |
| **Custom indoor** | IPS demo / map matching | Collect with phone + GT (Vicon / LiDAR / manual) |

### 6.2 Metrics

| Metric | Definition | Use |
|--------|------------|-----|
| **ATE** | Absolute Trajectory Error (RMSE) | Global consistency |
| **RPE** | Relative Pose Error (trans / rot) | Local odometry quality |
| **trel, rrel** | KITTI-style drift % | Compare directly to VIFT |
| **IPS error** | 2D distance to GT on floor (m) | Primary application metric |
| **Yaw error** | Heading RMSE | Navigation relevance |
| **Latency** | ms per frame | Real-time claim |
| **FPS** | On target GPU / phone | Deployment |

### 6.3 Splits (EuRoC example)

Document exact sequences in `configs/data/`:

- Train: MH_01, MH_02, V1_01, V1_02, … (define)
- Val: (define)
- Test: MH_05, V2_03, … (define)

---

## 7. Experiments Plan

### 7.1 Tier 1 — Required for publication

| ID | Experiment | Purpose |
|----|------------|---------|
| E1 | Reproduce VIFT on KITTI (same split) | Validate codebase |
| E2 | Train / evaluate on **EuRoC** | Indoor VIO standard |
| E3 | Test on **TUM-VI** (zero-shot or finetune) | Generalization |
| E4 | Ablation: vision-only, IMU-only, fusion | Show complementarity |
| E5 | Ablation: w/o causal mask, w/o RPMG/manifold loss | Architecture justification |
| E6 | Compare: VINS-Mono, ORB-SLAM3-VI, VIFT, Ours | SOTA table |

### 7.2 Tier 2 — IPS differentiation

| ID | Experiment | Purpose |
|----|------------|---------|
| E7 | Trajectory integration → **IPS 2D error** | Application metric |
| E8 | Long-sequence drift (5–10 min) | Practical IPS |
| E9 | Map matching / particle filter post-processing | Drift reduction |
| E10 | Cross-session / cross-building test | Robustness |

### 7.3 Tier 3 — Strong venue (optional)

| ID | Experiment | Purpose |
|----|------------|---------|
| E11 | Cross-dataset: train EuRoC, test custom building | Generalization story |
| E12 | Real-time profiling (GPU / Jetson / phone) | Deployment |
| E13 | Failure cases: dark corridor, pocket, fast turn | Qualitative figures |
| E14 | Uncertainty calibration (if Contribution D) | Reliability |
| E15 | 3+ random seeds, mean ± std | Statistical rigor |

### 7.4 Figures & tables checklist

- [ ] Architecture diagram (Fig. 1)
- [ ] Qualitative trajectories: GT vs Ours vs VINS (Fig. 2–3)
- [ ] Main results table: EuRoC + TUM-VI + KITTI (Table 1)
- [ ] IPS error vs walk duration (Fig. 4)
- [ ] Ablation table (Table 2)
- [ ] Attention / confidence visualization (Fig. 5, optional)
- [ ] Runtime table (Table 3, optional)

---

## 8. Implementation Roadmap

### Phase 0 — Environment & reproduction (Weeks 1–2)

- [x] Clone VIFT into `third_party/VIFT` (see `docs/PHASE0.md`)
- [ ] Install dependencies — run `scripts/phase0/setup_env.ps1`
- [x] Smoke test — `dummy_vio` 2 epochs on **Kaggle GPU** (local: `scripts/phase0/smoke_test.ps1` pending PyTorch fix)
- [ ] Download KITTI + pretrained encoders — see `scripts/phase0/`
- [ ] Run `experiment=latent_kitti_vio_paper` — `scripts/phase0/run_paper_train.ps1`
- [ ] Run evaluation — match published trel/rrel approximately
- [x] Document setup — `docs/SETUP.md`, `docs/PHASE0.md`

### Phase 1 — EuRoC data pipeline (Weeks 3–5)

- [x] Implement EuRoC dataloader (`third_party/VIFT/src/data/components/EuRoC_dataset.py`)
- [x] Add `configs/data/euroc_vio.yaml` (+ latent + smoke experiment)
- [x] Time-sync camera ↔ IMU; latent cache scripts (`latent_euroc_*.py`)
- [x] GT sanity check (`scripts/phase1/verify_gt_ate.py`) — see `docs/PHASE1.md`
- [x] EuRoC pipeline code: verify, latent cache, `euroc_latent_minimal` train + ATE test — see `docs/PHASE1.md`
- [ ] User: run on Kaggle/local with real EuRoC download

### Phase 2 — Model v1 (Weeks 6–9)

- [ ] Start from VIFT `PoseTransformer` + `WeightedVIOLitModule`
- [ ] Adapt input dims for EuRoC resolution
- [ ] Train fusion head on EuRoC latents (fair comparison to VIFT-style)
- [ ] First EuRoC results table vs VINS-Mono

### Phase 3 — End-to-end encoders (Weeks 10–14)

- [ ] Integrate trainable visual encoder (replace frozen FlowNet path)
- [ ] Integrate IMU encoder at full rate between frames
- [ ] Joint training with reduced LR on backbone
- [ ] Ablation: frozen vs finetuned encoders

### Phase 4 — IPS layer (Weeks 15–18)

- [ ] Trajectory integrator module (SE(3) chain product)
- [ ] 2D floor projection + IPS metrics script
- [ ] Optional map matching (occupancy grid / corridor graph)
- [ ] Collect or adopt indoor pedestrian test sequence

### Phase 5 — Paper & release (Weeks 19–24)

- [ ] Final SOTA tables + ablations
- [ ] Write manuscript (see Section 10)
- [ ] Clean repository, LICENSE, README, pretrained weights
- [ ] Submit to target venue (Section 9)

---

## 9. Publication Strategy

### 9.1 Recommended venues (ranked)

| Venue | Fit | Notes |
|-------|-----|-------|
| **IEEE RA-L** + ICRA | Strong | Needs solid EuRoC + classical baselines |
| **IROS / ICRA** | Strong | Robotics + indoor navigation angle |
| **WACV / 3DV** | Good | Learning-heavy VIO |
| **IEEE Sensors** | Moderate | Applied IPS, smartphone focus |
| **CVPR / ICCV main** | Stretch | Needs clear SOTA + novelty (B or E) |

### 9.2 Timeline (6-month aggressive / 12-month safe)

| Month | Milestone |
|-------|-----------|
| 1–2 | VIFT reproduced; EuRoC pipeline |
| 3–4 | EuRoC trained model v1; VINS comparison |
| 5–6 | End-to-end training; TUM-VI tests |
| 7–8 | IPS metrics + map matching; indoor collection |
| 9–10 | Ablations + figures |
| 11–12 | Paper writing + submission |

### 9.3 What reviewers will ask (prepare answers)

1. How is this different from VIFT? → Indoor IPS, trainable encoders, datasets, IPS metrics.
2. Why not use ORB-SLAM3-VI only? → Learning handles calibration / textureless regions; show hybrid or complementary.
3. Does it run in real time? → Report FPS and model size.
4. Scale ambiguity? → IMU provides metric scale; show ablation without IMU.
5. Generalization across buildings? → Cross-dataset experiment.

---

## 10. Paper Outline (draft)

### Abstract (structure)

- Problem: IPS indoors without infrastructure.
- Method: Causal Visual-Inertial Transformer (+ IPS integration).
- Results: Best/competitive on EuRoC/TUM-VI; IPS error X m on indoor walk.
- Code: public repository.

### Sections

1. **Introduction** — GPS failure indoors; camera+IMU; gaps in VIFT.
2. **Related Work** — VIO classical, deep VIO (VINet, VIFT), indoor localization, RoNIN.
3. **Method** — encoders, causal Transformer, loss, integration, optional map match.
4. **Experiments** — datasets, metrics, baselines, main table, ablations, IPS results.
5. **Conclusion** — limitations, future work (semantic map, WiFi fusion).

### Key references to cite

- VIFT (Kurt et al., 2024)
- Visual-Selective-VIO (Yang et al., ECCV 2022)
- VINet, DeepVIO, ATVIO
- VINS-Mono, ORB-SLAM3, DM-VIO
- RPMG (rotation loss)
- EuRoC, TUM-VI datasets
- RoNIN (pedestrian inertial)
- Map matching indoor navigation papers

---

## 11. Repository Structure (target)

```
IPS/
├── RESEARCH_PLAN.md          ← this file
├── README.md
├── docs/
│   ├── SETUP.md
│   └── EXPERIMENTS.md
├── configs/
│   ├── data/
│   │   ├── euroc_vio.yaml
│   │   └── tum_vi.yaml
│   ├── model/
│   │   └── indoor_vio_tf.yaml
│   └── experiment/
│       └── euroc_ips.yaml
├── src/
│   ├── data/
│   ├── models/
│   │   └── components/
│   ├── metrics/
│   │   ├── vio_metrics.py
│   │   └── ips_metrics.py
│   ├── integration/
│   │   └── trajectory.py
│   └── train.py
├── scripts/
│   ├── download_euroc.sh
│   └── eval_ips.py
└── third_party/
    └── VIFT/                 ← fork or submodule
```

---

## 12. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| EuRoC training unstable | High | Start with VIFT-style frozen encoders; finetune gradually |
| Cannot match VINS accuracy | Medium | Hybrid BA; tune longer; better IMU noise model |
| No indoor GT for IPS | High | Use EuRoC + RoNIN; plan small custom capture |
| Reviewers say “just VIFT on EuRoC” | High | IPS layer + end-to-end + uncertainty + smartphone |
| GPU memory limits | Medium | Smaller backbone, gradient checkpointing, shorter N |
| IMU-camera time offset | Medium | Estimate offset; augmentation during training |

---

## 13. Success Criteria (definition of done)

### Minimum (workshop / Sensors)

- [ ] EuRoC ATE competitive with learned VIO baselines
- [ ] Clear improvement over VIFT-style reimplementation on at least one indoor metric
- [ ] Public code + configs

### Strong (RA-L / IROS)

- [ ] EuRoC + TUM-VI tables with VINS-Mono and ORB-SLAM3-VI
- [ ] IPS error in meters on indoor sequence(s)
- [ ] Ablations + runtime
- [ ] End-to-end or hybrid with clear novelty

### Stretch (main conference)

- [ ] Cross-building generalization
- [ ] Mobile real-time demo
- [ ] Uncertainty-aware fusion with interpretable results

---

## 14. Notes & Decisions Log

Use this section to record decisions as the project progresses.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-19 | Plan created | Baseline VIFT; target Indoor IPS |
| | | |
| | | |

---

## 15. Quick Command Reference (VIFT baseline)

```bash
# Clone baseline
git clone https://github.com/ybkurt/VIFT.git third_party/VIFT
cd third_party/VIFT
pip install -r requirements.txt

# KITTI data + latent cache (see VIFT README)
cd data && sh data_prep.sh
python latent_caching.py

# Train paper config
python src/train.py experiment=latent_kitti_vio_paper trainer=gpu
```

---

*This document is the master plan for the IPS research project. Update checkboxes and tables as experiments complete.*
