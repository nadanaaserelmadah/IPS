"""AUDIT PAPER E5 (2026-11-10): retrain Visual-Selective-VIO (Yang et al., ECCV 2022) from scratch with the
clean protocol, to measure the seed spread of a method trained end to end (E6 measured it only for VIFT).

The authors' train.py at commit 27b06b38 (github.com/mingyuyng/Visual-Selective-VIO), run unchanged except:
  1. train_seq 00 01 02 04 06 09 (the authors' default adds 08);
  2. val_seq 08 05 07 10 (the authors' default is 05 07 10, the test drives, and their train.py keeps
     best_<t_rel>.pth by t_rel on them, i.e. checkpoint selection on the test set);
  3. the end-of-epoch evaluation (same epochs as the authors: ep > warmup + joint, i.e. 81-99) is replaced by
     one that scores every val drive and dumps <seq>_estimated_poses.npy / <seq>_gt_poses.npy per epoch, with
     the gumbel-softmax sampling seeded (1000 + epoch) and the training RNG state restored afterwards. The
     choice of epoch (best on 08, best on 05/07/10 like the authors, or the last) is made offline from the dumps;
  4. instead of one 190 MB checkpoint per epoch, one resume file (weights, Adam state, RNG states) is kept, and
     a session stops cleanly before Kaggle's 12 h limit. The next session is a kernel that attaches this one's
     output (kernel_sources) and continues from the same epoch;
  5. torch.load(..., weights_only=False) for the FlowNet weights (newer torch refuses the old pickle otherwise).
  6. speed only: the training frames are decoded and transformed once into a float16 cache (the transform acts
     on each frame alone; checked against the original __getitem__, max difference <= 3e-4), and evaluation
     reads frames with 4 DataLoader workers in order. Without this a T4 session is CPU bound at ~2.8 s/iteration.
  7. --gpu_ids over every visible GPU (Kaggle T4 x2): the authors' own DataParallel path, batch 16 split 8 + 8
     (BatchNorm statistics per GPU). On one T4 the run is GPU bound at ~2.4 s/iteration, ~55 h for 100 epochs.
Everything else (100 epochs: 40 warm-up, 40 joint, 20 fine-tune; batch 16; 11-frame windows; lr schedule; loss;
FlowNetS-BN init; released imus/*.mat) is the authors' default.

Inputs: hocop1/kitti-odometry (images, poses), <acct>/vsvio-e5-inputs (the authors' data/imus/*.mat and
flownets_bn_EPE2.459.pth.tar from their Drive folder, sha256 77bbee0e...c9c8).
Env: E5_SEED (default below), E5_SMOKE=1 (4 short epochs, 30 iterations each), E5_BUDGET_H (default 11.0).
Outputs in /kaggle/working/e5/: evals.jsonl, epochs/<ep>/<seq>_*.npy, best08.pth, resume.pth (while
unfinished), status.json, train_log.txt.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
SEED = int(os.environ.get("E5_SEED", "0"))
SMOKE = os.environ.get("E5_SMOKE", "0") == "1"
BUDGET_H = float(os.environ.get("E5_BUDGET_H", "11.0"))
EXPECT_RESUME = os.environ.get("E5_EXPECT_RESUME", "1") == "1"  # set to 1 by next_session.sh: never restart from scratch
COMMIT = "27b06b38dcf07a015fba22038050113462387e7d"
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
OUT = WORK / "e5"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{(time.time() - T0) / 60:7.1f} min] {msg}"
    print(line, flush=True)
    with open(OUT / "driver_log.txt", "a") as f:
        f.write(line + "\n")


def walk_find(pred, base=INPUT, max_depth=7):
    hits = []
    for root, dirs, files in os.walk(base, followlinks=True):
        if root[len(str(base)):].count(os.sep) >= max_depth or root.endswith(("image_2", "image_3", "velodyne")):
            dirs[:] = []
        if pred(Path(root), files):
            hits.append(Path(root))
    return hits


def sh(cmd, **kw):
    log("$ " + " ".join(map(str, cmd)))
    return subprocess.call(list(map(str, cmd)), **kw)


# ---------------------------------------------------------------- inputs
kitti = walk_find(lambda r, f: (r / "poses/05.txt").is_file() and (r / "sequences").is_dir())
imus = walk_find(lambda r, f: "05.mat" in f and "08.mat" in f)
flow = walk_find(lambda r, f: "flownets_bn_EPE2.459.pth.tar" in f)
# a previous session: its e5/ folder, mounted from the previous kernel (kernel_sources) or uploaded as a dataset
# (Kaggle unpacks a lone folder at the dataset root, so match on the files, not on the folder name)
prev = walk_find(lambda r, f: "status.json" in f and "resume.pth" in f)
assert kitti and imus and flow, f"inputs missing: kitti={kitti} imus={imus} flownet={flow}"
assert len(prev) <= 1 and (prev or not EXPECT_RESUME), f"continuation kernel, previous session found: {prev}"
KITTI_ROOT, IMUS, FLOW = kitti[0], imus[0], flow[0] / "flownets_bn_EPE2.459.pth.tar"
log(f"kitti {KITTI_ROOT}\nimus {IMUS}\nflownet {FLOW}\nprevious session {prev}")
sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
sh(["bash", "-c", "nproc; free -g | head -2; df -h /tmp /kaggle/working | cat"])
CACHE = Path("/tmp/e5cache") if shutil.disk_usage("/tmp").free > 14e9 else WORK / "e5cache"
log(f"frame cache in {CACHE} ({shutil.disk_usage(CACHE.parent if CACHE.parent.exists() else '/').free / 1e9:.0f} GB free)")

status = {"seed": SEED, "smoke": SMOKE, "commit": COMMIT, "budget_h": BUDGET_H, "sessions": []}
resume_in = None
if prev:  # continue a previous session: carry its history, resume from its state
    p = prev[0]
    old = json.loads((p / "status.json").read_text())
    assert old["seed"] == SEED and old["smoke"] == SMOKE, (old, SEED, SMOKE)
    assert not old.get("done"), "previous session already finished"
    status["sessions"] = old["sessions"]
    for item in p.iterdir():
        if item.name in ("resume.pth", "status.json"):
            continue
        dst = OUT / item.name
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, dst, **({"dirs_exist_ok": True} if item.is_dir() else {}))
    resume_in = p / "resume.pth"
    assert resume_in.is_file(), resume_in
session = {"n": len(status["sessions"]) + 1, "resume_from": str(resume_in) if resume_in else None, "start": time.time()}
status["sessions"].append(session)
(OUT / "status.json").write_text(json.dumps(status, indent=2))

# ---------------------------------------------------------------- code
repo = WORK / "vsvio"
if not repo.is_dir():
    assert sh(["git", "clone", "-q", "https://github.com/mingyuyng/Visual-Selective-VIO.git", repo]) == 0
assert sh(["git", "-C", repo, "checkout", "-q", COMMIT]) == 0
sh([sys.executable, "-m", "pip", "install", "-q", "path"])

data = WORK / "kitti"
data.mkdir(exist_ok=True)
for name, src in (("sequences", KITTI_ROOT / "sequences"), ("poses", KITTI_ROOT / "poses"), ("imus", IMUS)):
    if not (data / name).exists():
        (data / name).symlink_to(src)
assert (data / "sequences/00/image_2").is_dir(), "expected sequences/XX/image_2"

src = (repo / "train.py").read_text()
EVAL_START = "        if ep > args.epochs_warmup+args.epochs_joint:"
EVAL_END = "    message = f'Training finished"
i0, i1 = src.index(EVAL_START), src.index(EVAL_END)
src = src[:i0] + "        e5_eval(model, tester, ep, args)\n    \n" + src[i1:]
PATCHES = [
    ("default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training'",
     "default=['00', '01', '02', '04', '06', '09'], help='sequences for training'"),
    ("default=['05', '07', '10'], help='sequences for validation'",
     "default=['08', '05', '07', '10'], help='sequences for validation'"),
    ("import math\n", "import math\nfrom e5_hooks import e5_save, e5_eval, e5_resume, e5_subset, e5_budget\n"),
    ("torch.load(args.pretrain_flownet, map_location='cpu')",
     "torch.load(args.pretrain_flownet, map_location='cpu', weights_only=False)"),
    ("        torch.save(model.module.state_dict(), f'{checkpoints_dir}/{ep:003}.pth')\n",
     "        e5_save(model, optimizer, ep)\n"),
    ("    best = 10000\n", "    init_epoch = e5_resume(model, optimizer, init_epoch)\n    best = 10000\n"),
    ("        lr, selection, temp = update_status(ep, args, model)\n",
     "        e5_budget(ep, init_epoch, args)\n        lr, selection, temp = update_status(ep, args, model)\n"),
    ("    train_loader = torch.utils.data.DataLoader(\n",
     "    train_dataset = e5_subset(train_dataset)\n    train_loader = torch.utils.data.DataLoader(\n"),
]
for a, b in PATCHES:
    assert src.count(a) == 1, f"patch anchor not found exactly once: {a!r}"
    src = src.replace(a, b)
(repo / "train_e5.py").write_text(src)
sh(["git", "-C", repo, "diff", "--no-index", "--stat", repo / "train.py", repo / "train_e5.py"])
(OUT / "train_e5.diff").write_text(subprocess.run(["git", "-C", str(repo), "diff", "--no-index", str(repo / "train.py"),
                                                   str(repo / "train_e5.py")], capture_output=True, text=True).stdout)

HOOKS = r'''
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from utils.kitti_eval import kitti_eval

OUT = Path(os.environ["E5_OUT"])
T0 = float(os.environ["E5_T0"])
BUDGET = float(os.environ["E5_BUDGET_H"]) * 3600
SMOKE = os.environ.get("E5_SMOKE") == "1"
RESUME_IN = os.environ.get("E5_RESUME_IN")
_clock = {}


def _mem():
    """Host memory for the log (session 2 of seed 1 was SIGKILLed between epochs with nothing in the log)."""
    info = dict(l.split(":", 1) for l in open("/proc/meminfo"))
    rss = int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1])
    cg = next((Path(f) for f in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
               if Path(f).is_file()), None)
    return (f"mem: available {int(info['MemAvailable'].split()[0]) / 2**20:.1f} GB, rss {rss / 2**20:.1f} GB"
            + (f", cgroup {int(cg.read_text()) / 2**30:.1f} GB" if cg else ""))


def _note(msg):
    line = f"[{(time.time() - T0) / 60:7.1f} min] {msg}"
    print(line, flush=True)
    with open(OUT / "driver_log.txt", "a") as f:
        f.write(line + "\n")


class _Cached(torch.utils.data.Dataset):
    """The authors' KITTI dataset with each frame's transform output read from a float16 cache.
    The training transform (ToTensor - 0.5, then TF.resize to 256x512, no flip/colour) acts on each frame
    alone, so the cache holds exactly what __getitem__ would compute, rounded to float16 (<= 2.4e-4)."""

    def __init__(self, ds, mm, row):
        self.ds, self.mm, self.row = ds, mm, row

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        s = self.ds.samples[index]
        imgs = torch.from_numpy(self.mm[[self.row[p] for p in s["imgs"]]].astype(np.float32))
        return imgs, np.copy(s["imus"]), np.copy(s["gts"]), s["rot"].astype(np.float32), self.ds.weights[index]


_POOL = {}


def _fill(args):
    from PIL import Image
    lo, hi = args
    mm = np.memmap(_POOL["file"], dtype=np.float16, mode="r+", shape=_POOL["shape"])
    for k in range(lo, hi):
        t = _POOL["ds"].transform([np.asarray(Image.open(_POOL["paths"][k]))], None, None)[0][0]
        mm[k] = t.numpy().astype(np.float16)
    mm.flush()
    return hi - lo


def _cache(ds):
    import multiprocessing as mp
    t = time.time()
    paths = sorted({p for s in ds.samples for p in s["imgs"]})
    shape = (len(paths), 3, 256, 512)
    f = Path(os.environ["E5_CACHE"]) / "frames.f16"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.memmap(f, dtype=np.float16, mode="w+", shape=shape).flush()
    _POOL.update(file=str(f), shape=shape, ds=ds, paths=paths)
    chunks = [(i, min(i + 200, len(paths))) for i in range(0, len(paths), 200)]
    with mp.get_context("fork").Pool(os.cpu_count()) as pool:
        assert sum(pool.map(_fill, chunks)) == len(paths)
    mm = np.memmap(f, dtype=np.float16, mode="r", shape=shape)
    cached = _Cached(ds, mm, {p: i for i, p in enumerate(paths)})
    rng = np.random.RandomState(123)
    err = 0.0
    for idx in rng.choice(len(ds), 4, replace=False):
        a, b = ds[idx], cached[idx]
        err = max(err, float((a[0] - b[0]).abs().max()))
        assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) and a[3] == b[3] and a[4] == b[4]
    assert err <= 3e-4, err
    _note(f"frame cache: {len(paths)} frames, {f.stat().st_size / 1e9:.1f} GB in {(time.time() - t) / 60:.1f} min, "
          f"max |cached - original| on 4 windows {err:.2e}")
    return cached


def e5_subset(ds):
    _note(f"train windows: {len(ds)}")
    ds = _cache(ds)
    if SMOKE:
        step = max(1, len(ds) // 480)
        ds = torch.utils.data.Subset(ds, list(range(0, len(ds), step))[:480])
    _clock["iters_per_epoch"] = -(-len(ds) // 16)
    return ds


def _rng():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "np": np.random.get_state()}


def _set_rng(s):
    torch.set_rng_state(s["torch"]); torch.cuda.set_rng_state_all(s["cuda"]); np.random.set_state(s["np"])


def e5_resume(model, optimizer, init_epoch):
    if not RESUME_IN:
        return init_epoch
    ck = torch.load(RESUME_IN, map_location="cpu", weights_only=False)
    model.module.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optim"])
    _set_rng(ck["rng"])
    _note(f"resumed from {RESUME_IN}: epoch {ck['epoch']} done, continuing at {ck['epoch'] + 1}")
    return ck["epoch"] + 1


def e5_budget(ep, init_epoch, args):
    now = time.time()
    if "ep_start" in _clock:
        _clock["last"] = now - _clock["ep_start"]
        _note(f"epoch {ep - 1}: {_clock['last'] / 60:.1f} min (train {_clock['train'] / 60:.1f}, iters {_clock['iters']}); {_mem()}")
        nxt = _clock["last"] * (1.3 if ep == args.epochs_warmup + args.epochs_joint + 1 else 1.0)  # first eval epoch
        if now - T0 + 1.15 * nxt > BUDGET:
            st = json.loads((OUT / "status.json").read_text())
            st["sessions"][-1].update(stopped_before_epoch=ep, end=now)
            (OUT / "status.json").write_text(json.dumps(st, indent=2))
            _note(f"time budget: stopping before epoch {ep}; resume.pth holds epoch {ep - 1}")
            sys.exit(0)
    _clock["ep_start"] = now


def e5_save(model, optimizer, ep):
    _clock["train"] = time.time() - _clock["ep_start"]
    _clock["iters"] = _clock.get("iters_per_epoch")
    ck = {"model": model.module.state_dict(), "optim": optimizer.state_dict(), "epoch": ep, "rng": _rng()}
    tmp = OUT / "resume.pth.tmp"
    torch.save(ck, tmp)
    os.replace(tmp, OUT / "resume.pth")


def _test_one_path(net, df, num_gpu):
    """KITTI_tester.test_one_path (utils/kitti_eval.py) with the frames read by 4 workers, in order."""
    hc, pose_list, decision_list = None, [], []
    loader = torch.utils.data.DataLoader(df, batch_size=None, shuffle=False, num_workers=4)
    for i, (image_seq, imu_seq, gt_seq) in enumerate(loader):
        x_in = image_seq.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1).cuda()
        i_in = imu_seq.unsqueeze(0).repeat(num_gpu, 1, 1).cuda()
        with torch.no_grad():
            pose, decision, probs, hc = net(x_in, i_in, is_first=(i == 0), hc=hc, selection="gumbel-softmax", p=0.5)
        pose_list.append(pose[0, :, :].detach().cpu().numpy())
        decision_list.append(decision[0, :, :].detach().cpu().numpy()[:, 0])
    return np.vstack(pose_list), np.hstack(decision_list)


def e5_eval(model, tester, ep, args):
    if ep <= args.epochs_warmup + args.epochs_joint:  # the authors evaluate only after warm-up + joint
        return
    t = time.time()
    keep = _rng()
    torch.manual_seed(1000 + ep)
    model.eval()
    d = OUT / "epochs" / f"{ep:03d}"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"epoch": ep}
    with torch.no_grad():
        for i, seq in enumerate(args.val_seq):
            pe, de = _test_one_path(model, tester.dataloader[i], len(args.gpu_ids.split(",")))
            gt = tester.dataloader[i].poses_rel
            _, _, t_rel, r_rel, t_rmse, r_rmse, usage, _ = kitti_eval(pe, de, gt)
            np.save(d / f"{seq}_estimated_poses.npy", pe)
            np.save(d / f"{seq}_gt_poses.npy", gt)
            rec[seq] = {"t_rel": float(t_rel), "r_rel": float(r_rel), "usage": float(usage)}
    rec["test_mean_t_rel"] = float(np.mean([rec[s]["t_rel"] for s in ("05", "07", "10")]))
    _set_rng(keep)
    hist = [json.loads(l) for l in open(OUT / "evals.jsonl")] if (OUT / "evals.jsonl").is_file() else []
    if all(rec["08"]["t_rel"] < h["08"]["t_rel"] for h in hist):
        torch.save(model.module.state_dict(), OUT / "best08.pth")
        rec["best08_saved"] = True
    with open(OUT / "evals.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    _note(f"eval epoch {ep} ({(time.time() - t) / 60:.1f} min): 08 {rec['08']['t_rel']:.2f} | 05/07/10 "
          f"{rec['05']['t_rel']:.2f} {rec['07']['t_rel']:.2f} {rec['10']['t_rel']:.2f} = {rec['test_mean_t_rel']:.2f}")
'''
(repo / "e5_hooks.py").write_text(HOOKS)

# ---------------------------------------------------------------- train
env = dict(os.environ, E5_CACHE=str(CACHE), E5_OUT=str(OUT), E5_T0=str(T0), E5_BUDGET_H=str(BUDGET_H), E5_SMOKE="1" if SMOKE else "0",
           PYTHONUNBUFFERED="1")
if resume_in:
    env["E5_RESUME_IN"] = str(resume_in)
import torch  # noqa: E402

GPUS = ",".join(str(g) for g in range(torch.cuda.device_count()))
log(f"gpu_ids {GPUS}")
cmd = [sys.executable, "train_e5.py", "--data_dir", data, "--seed", SEED, "--gpu_ids", GPUS, "--workers", "4",
       "--experiment_name", f"e5_s{SEED}", "--pretrain_flownet", FLOW]
if SMOKE:
    cmd += ["--epochs_warmup", "1", "--epochs_joint", "1", "--epochs_fine", "2"]
with open(OUT / "train_log.txt", "a") as f:
    rc = sh(cmd, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT)
tail = (OUT / "train_log.txt").read_text().splitlines()[-25:]
print("\n".join(tail))

status = json.loads((OUT / "status.json").read_text())
cur = status["sessions"][-1]
cur.update(rc=rc, end=cur.get("end", time.time()))
done = rc == 0 and "stopped_before_epoch" not in cur and any("Training finished" in l for l in tail)
status["done"] = done
if done:
    (OUT / "resume.pth").unlink(missing_ok=True)
(OUT / "status.json").write_text(json.dumps(status, indent=2))
shutil.rmtree(repo / "results", ignore_errors=True)
shutil.rmtree(CACHE, ignore_errors=True)
log(f"rc {rc}, done {done}, epochs evaluated: "
    f"{sum(1 for _ in open(OUT / 'evals.jsonl')) if (OUT / 'evals.jsonl').is_file() else 0}")
assert rc == 0, f"train_e5.py failed with code {rc}"
