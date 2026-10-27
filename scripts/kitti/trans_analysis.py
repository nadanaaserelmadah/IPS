import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_hybrid as h
D, C = sys.argv[1], sys.argv[2]
seqs = ["08", "05", "07", "10"]
E = {s: np.load(f"{D}/{s}_estimated_poses.npy").astype(float) for s in seqs}
G = {s: np.load(f"{D}/{s}_gt_poses.npy").astype(float) for s in seqs}
GR = {s: np.load(f"{C}/{s}_gyro_R.npy") for s in seqs}
stat = {}
for s in seqs:
    te, tg = E[s][:, 3:], G[s][:, 3:]
    mv = np.linalg.norm(tg, axis=1) > 0.3  # moving frames
    ratio = np.linalg.norm(te[mv], axis=1).sum() / np.linalg.norm(tg[mv], axis=1).sum()
    cosang = np.sum(te[mv] * tg[mv], 1) / (np.linalg.norm(te[mv], axis=1) * np.linalg.norm(tg[mv], axis=1))
    dirdeg = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    bias = (te - tg).mean(0)
    # direction bias: mean of normalized lateral/vertical components relative to forward
    lat = np.mean(te[mv, 0] / te[mv, 2] - tg[mv, 0] / tg[mv, 2]); ver = np.mean(te[mv, 1] / te[mv, 2] - tg[mv, 1] / tg[mv, 2])
    stat[s] = dict(ratio=ratio, lat=lat, ver=ver)
    print(f"{s}: scale ratio {ratio:.4f} | dir err median {np.median(dirdeg):.3f} deg | mean err x/y/z {np.round(bias, 4)} m | slope err lat {np.degrees(lat):+.3f} deg ver {np.degrees(ver):+.3f} deg | per-frame |t| err std {np.std(np.linalg.norm(te,axis=1)-np.linalg.norm(tg,axis=1)):.4f} m")


def fix(s, scale=1.0, lat=0.0, ver=0.0):
    e = E[s].copy(); t = e[:, 3:]
    t[:, 0] -= lat * t[:, 2]; t[:, 1] -= ver * t[:, 2]
    e[:, 3:] = t / scale
    return h.hybrid(e, GR[s])


def run(name, f):
    r = np.array([h.score(f(s), G[s]) for s in seqs])
    print(f"| {name} | " + " | ".join(f"{a:.2f}/{b:.2f}" for a, b in r) + f" | **{r[1:,0].mean():.2f}** |")


print("| gyro rotation + translation fix | 08 (calib) | 05 | 07 | 10 | mean 05/07/10 |\n|---|---|---|---|---|---|")
run("none", lambda s: fix(s))
run("oracle own scale", lambda s: fix(s, stat[s]["ratio"]))
run("scale from 08", lambda s: fix(s, stat["08"]["ratio"]))
run("oracle own lat+ver slope", lambda s: fix(s, 1, stat[s]["lat"], stat[s]["ver"]))
run("lat+ver slope from 08", lambda s: fix(s, 1, stat["08"]["lat"], stat["08"]["ver"]))
run("scale+slope from 08", lambda s: fix(s, stat["08"]["ratio"], stat["08"]["lat"], stat["08"]["ver"]))
run("oracle own scale+slope", lambda s: fix(s, stat[s]["ratio"], stat[s]["lat"], stat[s]["ver"]))
