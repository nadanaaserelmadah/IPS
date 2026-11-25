# Running the Scientific Reports figure scripts

Scripts (this folder): `sr_style.py` (shared style), `fig_heading_error.py`, `fig_measure_vs_learn.py`.
Both import the numerical code of `scripts/kitti/plot_figures.py` and `scripts/gate/plot_observability.py` and change only the styling.

## Python

The system `python3` has matplotlib and numpy but the data code also imports scipy and VIFT's `src` (hydra), so use the scoring venv that already holds all of them:

```bash
PY=/Users/finetch/Library/Caches/ips_e5_chain/score/venv/bin/python
```

## Input data (found / missing)

All of this lives in `IN=/Users/finetch/Library/Caches/ips_e5_chain/score/inputs` (built by `scripts/audit/build_score_inputs.sh`):

| Argument | Path | Status |
|---|---|---|
| GYRO_DIR | `$IN/gyro` (`<seq>_gyro_R.npy`, `<seq>_gyro_maxgap_s.npy`, `calib_2011_09_30.json`) | found |
| NPZ_DIR | `$IN/npz` (`05.npz 07.npz 10.npz`) | found |
| RAW_CALIB | `$IN/raw_calib` (contains `2011_09_30/`) | found |
| KITTI_GEO | `$IN/geo_test/kgeo/lag60` (`05/07/10_geo_dir.npz`) | found |
| EUROC_GEO | `/Users/finetch/Documents/IPS/results_cache/2026-10-28/euroc_geo` (`<seq>_geo.npz`, 4 sequences) | found |
| KITTI DUMP_DIR (July reproduction, epoch 144, with-08) | none | **MISSING** |
| EUROC_DUMPS/pivift_s123_last | none | **MISSING** (substitute below) |

### Missing 1: the KITTI dump behind the existing PNGs

The existing PNGs use the July reproduction checkpoint (`latent_kitti_vio_paper`, epoch 144, trained with 08; docs/RESULTS.md line 255, kernel `sallyhaggag/vift-kitti-dump`). Its `05/07/10_{estimated,gt}_poses.npy` are not on this Mac. The only with-08 dumps present locally are other models:

- `/Users/finetch/Documents/IPS/results_cache/2026-10-28/kitti/out_vift-kitti-gyrocond-08-mirror` (and `-s2`, `-s3`): gyro-conditioned mirror runs.
- `/Users/finetch/Library/Caches/ips_e5_chain/vift_auth/auth_s{42,2,3}/kitti_rpmg/poses_{best,last}`: VIFT authors-protocol reruns.

The numbers match the current PNGs only with the July dump. To fetch it (the output layout of that kernel was not checked; look for the folder holding `05_estimated_poses.npy`):

```bash
mkdir -p /Users/finetch/Documents/IPS/results_cache/july_dump
KAGGLE_CONFIG_DIR=~/.kaggle-accounts/sallyhaggag kaggle kernels output sallyhaggag/vift-kitti-dump -p /Users/finetch/Documents/IPS/results_cache/july_dump
find /Users/finetch/Documents/IPS/results_cache/july_dump -name 05_estimated_poses.npy
```

Then set `KD` below to the folder printed by `find` (without the file name). To test the scripts before the download, `KD` can point at one of the local dumps above.

### Missing 2: EuRoC PI-VIFT dump `pivift_s123_last`

`plot_observability.py` reads `EUROC_DUMPS/pivift_s123_last/<seq>_{estimated,gt}_poses.npy` for MH_05_difficult, V2_02_medium, V1_01_easy, V1_02_medium. That set was not preserved. The only 4-sequence PI-VIFT dump kept is `results_cache/2026-10-28/euroc_ft/pose_dumps_best` (a different checkpoint, network ATE 3.60 m; see docs/ONLINE_GYRO_BIAS_DESIGN.md line 96). Using it as a stand-in keeps the figure's shape but the orange curve in panel a will NOT be numerically identical to the old PNG:

```bash
mkdir -p /Users/finetch/Documents/IPS/results_cache/euroc_dumps_stub
ln -sfn /Users/finetch/Documents/IPS/results_cache/2026-10-28/euroc_ft/pose_dumps_best /Users/finetch/Documents/IPS/results_cache/euroc_dumps_stub/pivift_s123_last
```

The original checkpoint would have to be regenerated from Kaggle (kernels `scripts/kaggle_kernels/kernel_v203_infer` and `kernel_tumft*` use `pivift_s123_last.ckpt`) to get identical numbers.

## Commands

```bash
cd /Users/finetch/Documents/IPS
PY=/Users/finetch/Library/Caches/ips_e5_chain/score/venv/bin/python
IN=/Users/finetch/Library/Caches/ips_e5_chain/score/inputs
KD=/Users/finetch/Documents/IPS/results_cache/july_dump        # <- folder containing 05_estimated_poses.npy (see Missing 1)
OUT=/Users/finetch/Documents/IPS/paper/scireports/figures

# Figure 1: heading error (writes $OUT/kitti_heading_error.pdf)
$PY scripts/scireports/fig_heading_error.py "$KD" "$IN/gyro" "$OUT"

# Figure 2: measure vs learn (writes the PDF given as the last argument)
$PY scripts/scireports/fig_measure_vs_learn.py "$KD" "$IN/geo_test/kgeo/lag60" "$IN/npz" "$IN/gyro" "$IN/raw_calib" \
    /Users/finetch/Documents/IPS/results_cache/euroc_dumps_stub \
    /Users/finetch/Documents/IPS/results_cache/2026-10-28/euroc_geo \
    "$OUT/observability_measure_vs_learn.pdf"
```

Checks after running: open both PDFs and look at label overlaps (legend vs the two-line note in panel a of figure 2, "Gate threshold" and the lower note in panel b of figure 2; I could not render them), and `pdffonts` should list only Arial (Type 42/TrueType), no Type 3.

The original commands for the PNGs were the docstring forms (`python scripts/kitti/plot_figures.py DUMP_DIR GYRO_DIR OUT_DIR` and `python scripts/gate/plot_observability.py KITTI_DUMP KITTI_GEO NPZ_DIR GYRO_DIR RAW_CALIB EUROC_DUMPS EUROC_GEO OUT_PNG`); no log of the exact paths used exists in docs/RESULTS.md, docs/SUBMISSION_PLAN.md, scripts/audit or results_cache.
