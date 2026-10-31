"""Classical baseline run by us: ORB-SLAM3 (v1.0) mono-inertial and stereo-inertial on EuRoC MH_05,
V2_02, V1_01, V1_02, with and without loop closure (env ORB_NO_LC patched into LoopClosing), CPU only.
Builds Pangolin v0.6 + ORB-SLAM3 from source (viewer disabled), runs the official EuRoC examples and
keeps the frame trajectories (TUM format, ns timestamps) for scoring with our own protocol.
Outputs land in /kaggle/working/orbslam3/.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

T0 = time.time()
WORK, INPUT = Path("/kaggle/working"), Path("/kaggle/input")
OUT = WORK / "orbslam3"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("/kaggle/tmp_build")
SRC.mkdir(parents=True, exist_ok=True)
LOG = OUT / "build_run_log.txt"


def log(msg):
    line = f"[{(time.time() - T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd, cwd=None, check=True, tail=60):
    log("$ " + cmd)
    p = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout.splitlines()
    with LOG.open("a") as f:
        f.write("\n".join(out[-400:]) + "\n")
    if p.returncode != 0:
        log("\n".join(out[-tail:]))
        if check:
            raise SystemExit(f"failed ({p.returncode}): {cmd}")
    return p.returncode


# ------------------------------------------------------------------ dependencies
sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake git "
   "libeigen3-dev libglew-dev libgl1-mesa-dev libboost-serialization-dev libboost-filesystem-dev libssl-dev "
   "libopencv-dev libepoxy-dev pkg-config > /dev/null")
sh("pkg-config --modversion opencv4 || pkg-config --modversion opencv", check=False)
nproc = os.cpu_count() or 4
# newer gcc: old Pangolin / ORB-SLAM3 sources miss <cstdint> / <limits>; force-include them everywhere
CXXF = '-DCMAKE_CXX_FLAGS="-include cstdint -include limits -w"'

# ------------------------------------------------------------------ Pangolin v0.6
if not (SRC / "Pangolin/build/src/libpangolin.so").exists():
    sh("git clone -q --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git", cwd=SRC)
    p = SRC / "Pangolin"
    # gcc >= 11: missing <limits>
    for f in p.rglob("*.h"):
        t = f.read_text(errors="ignore")
        if "std::numeric_limits" in t and "#include <limits>" not in t:
            f.write_text("#include <limits>\n" + t)
    sh(f"mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release {CXXF} -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF "
       f"-DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBJPEG=OFF > /dev/null && make -j{nproc} > /dev/null && make install > /dev/null", cwd=p)
    sh("ldconfig")

# ------------------------------------------------------------------ ORB-SLAM3 v1.0
orb = SRC / "ORB_SLAM3"
if not (orb / "lib/libORB_SLAM3.so").exists():
    sh("git clone -q --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git", cwd=SRC)
    # C++14 everywhere (newer compilers reject the C++11 flag combination used upstream)
    for cm in [orb / "CMakeLists.txt", orb / "Thirdparty/g2o/CMakeLists.txt", orb / "Thirdparty/DBoW2/CMakeLists.txt",
               orb / "Thirdparty/Sophus/CMakeLists.txt"]:
        if cm.exists():
            t = cm.read_text().replace("-std=c++11", "-std=c++14").replace("CXX_STANDARD 11", "CXX_STANDARD 14")
            t = t.replace('CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)',
                          'CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)')
            t = t.replace("find_package(OpenCV 4.4)", "find_package(OpenCV 4)")
            cm.write_text(t)
    # loop closure switch: ORB_NO_LC=1 disables it (odometry mode, comparable to VIO without loop closure)
    lc = orb / "src/LoopClosing.cc"
    t = lc.read_text()
    t = t.replace("mbActiveLC(true)", 'mbActiveLC(std::getenv("ORB_NO_LC") == nullptr)')
    if "#include <cstdlib>" not in t:
        t = "#include <cstdlib>\n" + t
    lc.write_text(t)
    # no display on Kaggle: disable the viewer in the EuRoC inertial examples
    for ex in list((orb / "Examples").rglob("*_euroc.cc")):
        t = ex.read_text()
        t = t.replace("ORB_SLAM3::System::IMU_MONOCULAR, true", "ORB_SLAM3::System::IMU_MONOCULAR, false")
        t = t.replace("ORB_SLAM3::System::IMU_STEREO, true", "ORB_SLAM3::System::IMU_STEREO, false")
        t = t.replace("ORB_SLAM3::System::MONOCULAR, true", "ORB_SLAM3::System::MONOCULAR, false")
        t = t.replace("ORB_SLAM3::System::STEREO, true", "ORB_SLAM3::System::STEREO, false")
        ex.write_text(t)
    for d in ["Thirdparty/DBoW2", "Thirdparty/g2o", "Thirdparty/Sophus"]:
        sh(f"mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release {CXXF} > /dev/null && make -j{nproc} > /dev/null", cwd=orb / d)
    sh("cd Vocabulary && tar -xf ORBvoc.txt.tar.gz", cwd=orb)
    sh(f"mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release {CXXF} > /dev/null && make -j{nproc} 2>&1 | grep -E 'error|Error|Built target' | tail -40", cwd=orb)
assert (orb / "Examples/Monocular-Inertial/mono_inertial_euroc").exists(), "ORB-SLAM3 examples not built"
log("build done")

# ------------------------------------------------------------------ data + runs
euroc = None
for root, dirs, files in os.walk(INPUT, followlinks=True):
    if "MH_05_difficult" in dirs and "V1_01_easy" in dirs:
        euroc = Path(root)
        break
    if root.endswith(("/data", "/cam1")):
        dirs[:] = []
assert euroc, "EuRoC not found"
log(f"euroc {euroc}")
SEQS = {"MH_05_difficult": "MH05", "V2_02_medium": "V202", "V1_01_easy": "V101", "V1_02_medium": "V102"}
CFG = {"monoi": ("Examples/Monocular-Inertial/mono_inertial_euroc", "Examples/Monocular-Inertial/EuRoC.yaml",
                 "Examples/Monocular-Inertial/EuRoC_TimeStamps"),
       "stereoi": ("Examples/Stereo-Inertial/stereo_inertial_euroc", "Examples/Stereo-Inertial/EuRoC.yaml",
                   "Examples/Stereo-Inertial/EuRoC_TimeStamps")}
for mode, (exe, cfg, tsd) in CFG.items():
    for lc in ("lc", "nolc"):
        for seq, short in SEQS.items():
            tag = f"{mode}_{lc}_{seq}"
            run = WORK / "runs" / tag
            run.mkdir(parents=True, exist_ok=True)
            env = "ORB_NO_LC=1 " if lc == "nolc" else ""
            t = time.time()
            rc = sh(f"{env}timeout 3600 {orb / exe} {orb / 'Vocabulary/ORBvoc.txt'} {orb / cfg} {euroc / seq} "
                    f"{orb / tsd / (short + '.txt')} traj_{tag} > run.log 2>&1", cwd=run, check=False)
            for f in run.glob("f_traj_*.txt"):
                shutil.copy2(f, OUT / f"{tag}_frames.txt")
            for f in run.glob("kf_traj_*.txt"):
                shutil.copy2(f, OUT / f"{tag}_keyframes.txt")
            shutil.copy2(run / "run.log", OUT / f"{tag}_run.log") if (run / "run.log").exists() else None
            log(f"{tag}: rc {rc}, {(time.time() - t) / 60:.1f} min, outputs {[p.name for p in run.glob('*traj*')]}")
log("all done")
