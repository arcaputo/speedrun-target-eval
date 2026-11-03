"""Kupanga benchmark: the modded-nanogpt speedrun (Track 1) on 8×H100, timed against a same-node baseline.

The subject is a checkout of https://github.com/KellerJordan/modded-nanogpt (a record, or a candidate) in
subjects/<name>/. On one 8×H100 SXM machine the eval:

  1. Downloads the speedrun's data (kjj0/fineweb10B-gpt2: the val shard and 9 training shards of 100M
     tokens) into a folder it owns, read-only, given to the run as DATA_PATH.
  2. Rebuilds the parent's code by reverse-applying $HARNESS_CHANGES_FILE (what the candidate changed
     since its experiment started) to a copy of the subject. The base node has no parent: it is timed
     on its own.
  3. Runs `torchrun --standalone --nproc_per_node=8 train_gpt.py` KX_RUNS times for the candidate and
     KX_BASELINE_RUNS times for the parent, interleaved so drift on the node hits both, and reads each
     run's last `step:N/N val_loss:X train_time:Yms` line. train_time leaves out compile and warmup, as
     the leaderboard's does.
  4. Reports the candidate's mean train_time (the primary metric) and its standard error, the mean
     val_loss with the speedrun's one-sided t-test against 3.28 (scipy.stats.ttest_1samp(losses, 3.28,
     alternative="less")), and the time difference from the parent measured on this node.

A candidate whose mean val_loss isn't below 3.28 with p < 0.01 is reported with meets_target = 0: a time only
counts at the target loss. The rules a candidate must keep (val_tokens, the data, no compiler flags) are checked by the
speedrun-rules QA eval before this one runs.

Environment variables: KX_RUNS (8), KX_BASELINE_RUNS (3), KX_TRAIN_SHARDS (9), KX_RUN_TIMEOUT_S (1500,
the first run of each code compiles for several minutes), KX_NPROC (8).

Local check of the parsing and statistics, no GPU: python bench.py --parse <log.txt> [<log.txt> ...]
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESULT_FILE = os.environ.pop("HARNESS_RESULT_FILE", None)  # before anything runs the code under test


def report(result: dict) -> None:
    """Print the result, and write it to the file Kupanga reads it from (docs/evals.md)."""
    line = json.dumps(result)
    print(line, flush=True)
    if RESULT_FILE:
        with open(RESULT_FILE, "w") as f:
            f.write(line + "\n")


DATA_URL = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main"  # cached_fineweb10B.py's source
TARGET = 3.28  # the speedrun's val_loss target
ALPHA = 0.01  # the speedrun's significance level
RUNS = int(os.environ.get("KX_RUNS", "8"))
BASELINE_RUNS = int(os.environ.get("KX_BASELINE_RUNS", "3"))
MIN_RUNS = 4  # fewer candidate runs than this (out of time) can't support the t-test
TRAIN_SHARDS = int(os.environ.get("KX_TRAIN_SHARDS", "9"))
RUN_TIMEOUT_S = int(os.environ.get("KX_RUN_TIMEOUT_S", "1500"))
NPROC = int(os.environ.get("KX_NPROC", "8"))
EVAL_TIMEOUT_S = int(os.environ.get("HARNESS_EVAL_TIMEOUT_S", "5400"))
# get_kernel("<repo>", version=N) calls in the subject: Hugging Face hub kernels it downloads at import time
HUB_KERNEL = re.compile(r"""get_kernel\(\s*['"]([^'"]+)['"](?:\s*,\s*version\s*=\s*(\d+))?""")
TRACEBACK = re.compile(r"^(\[rank\d+\]:\s?)?Traceback \(most recent call last\):", re.M)
FINAL = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms", re.M)


def find_subject() -> Path:
    root = Path(os.environ.get("HARNESS_SUBJECTS_DIR") or Path(__file__).resolve().parent.parent.parent / "subjects")
    found = sorted({p.parent for p in root.glob("*/train_gpt.py")} | {p.parent.parent for p in root.glob("*/*/train_gpt.py")})
    if not found:
        raise SystemExit(f"no modded-nanogpt checkout (train_gpt.py) under {root}/<subject>/ or {root}/<subject>/<record>/")
    return found[0]


def record_dir(subject: Path) -> Path:
    """Where train_gpt.py runs, relative to the subject: its root, or, for a subject that holds several records in
    folders of their own (subject/<record>/train_gpt.py), the one folder the candidate changed."""
    if (subject / "train_gpt.py").is_file():
        return Path(".")
    records = sorted(p.parent.relative_to(subject) for p in subject.glob("*/train_gpt.py"))
    changes = os.environ.get("HARNESS_CHANGES_FILE")
    diff = Path(changes).read_text(errors="replace") if changes and Path(changes).is_file() else ""
    prefix = f"subjects/{subject.name}/"
    touched = {line.split(" ", 1)[1].strip()[2:].removeprefix(prefix).split("/", 1)[0]
               for line in diff.splitlines() if line.startswith(("--- a/", "+++ b/"))}
    changed = [r for r in records if str(r) in touched]
    names = ", ".join(str(r) for r in records)
    if not changed:
        raise SystemExit(f"this subject holds several records ({names}) and the candidate changed none of them, so "
                         "there is nothing to time (the base node of such a project has no run of its own)")
    if len(changed) > 1:
        raise SystemExit(f"the candidate changed {', '.join(map(str, changed))}; an experiment changes one record folder")
    return changed[0]


def fetch(name: str, dest: Path) -> Path:
    path = dest / name
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{DATA_URL}/{name}", timeout=300) as r, open(path, "wb") as f:  # noqa: S310
                while chunk := r.read(1 << 24):
                    f.write(chunk)
            return path
        except OSError as e:
            print(f"download {name} failed ({e}), attempt {attempt + 1}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"could not download {name}")


def prepare_data(root: Path) -> None:
    """root/data/fineweb10B/ as `python data/cached_fineweb10B.py 9` leaves it, read-only."""
    d = root / "data" / "fineweb10B"
    d.mkdir(parents=True)
    names = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, TRAIN_SHARDS + 1)]
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(lambda n: fetch(n, d), names))
    for p in paths:
        p.chmod(0o444)
    d.chmod(0o555)
    print(f"downloaded {len(paths)} shards in {time.time() - t0:.0f}s", flush=True)


def parent_code(subject: Path, work: Path) -> Path | None:
    """A copy of the subject with the experiment's changes reverse-applied (the parent's code), or None on
    the base node."""
    changes = os.environ.get("HARNESS_CHANGES_FILE")
    if not changes or not Path(changes).is_file() or not Path(changes).read_text().strip():
        return None
    diff = Path(changes).read_text(errors="replace")
    if "[harness: the diff is cut here" in diff:
        raise SystemExit("the candidate's changes are too large for Kupanga's diff, so the parent can't be rebuilt "
                         "for the same-node baseline")
    dest = work / "subjects" / subject.name
    shutil.copytree(subject, dest, symlinks=True, ignore=shutil.ignore_patterns("logs", "__pycache__"))
    if shutil.which("git") is None:  # the PyTorch image has none; the job runs as root
        print("installing git to rebuild the parent", flush=True)
        subprocess.run("apt-get update -qq && apt-get install -y -qq --no-install-recommends git", shell=True,
                       check=True, capture_output=True, env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
    p = subprocess.run(["git", "apply", "-R", "--whitespace=nowarn", changes], cwd=work, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"could not rebuild the parent from the candidate's changes: {p.stderr.strip()[:500]}")
    return dest


def prefetch_kernels(code: Path) -> None:
    """Download the subject's hub kernels once, before torchrun: otherwise every rank downloads them into the
    same cache at import time, and one rank losing that race fails the run in its first seconds."""
    # train_gpt_medium.py is the other track's script (its kernel has no build for this torch); records/ is logs
    files = [f for f in code.rglob("*.py") if f.name != "train_gpt_medium.py" and "records" not in f.parts]
    found = sorted({(m.group(1), m.group(2)) for f in files for m in HUB_KERNEL.finditer(f.read_text(errors="replace"))})
    for name, version in found:
        arg = f", version={int(version)}" if version else ""
        for attempt in range(3):
            # in a temp dir, so the subject's own modules aren't on the path
            p = subprocess.run([sys.executable, "-c", f"from kernels import get_kernel; get_kernel({name!r}{arg})"],
                               cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=600)
            if p.returncode == 0:
                print(f"kernel {name}{arg}: cached", flush=True)
                break
            print(f"kernel {name}{arg}: download failed (attempt {attempt + 1})", flush=True)
            time.sleep(10 * (attempt + 1))
        else:
            # not fatal: the regex can match a call the run never makes; a real failure shows in the run's traceback
            print(f"kernel {name}{arg}: could not prefetch, the run will fetch it itself: {p.stderr.strip()[-1500:]}",
                  flush=True)


def failure_excerpt(out: str) -> str:
    """The first Python traceback in a torchrun's output (any rank), then its last lines: torchrun's summary of
    an 8-rank failure alone is longer than the tail, so the cause would otherwise be cut."""
    lines = out.splitlines()
    tail = "\n".join(lines[-25:])
    m = TRACEBACK.search(out)
    if m is None:
        return tail
    start = out[:m.start()].count("\n")
    prefix = m.group(1) or ""
    block = [lines[start]]
    for line in lines[start + 1:]:
        if prefix and not line.startswith(prefix.rstrip()):
            break
        block.append(line)
        # the exception line ends a traceback: unindented, after the frames
        if not prefix and line and not line[0].isspace():
            break
    # Inductor's tracebacks run to hundreds of lines and end in the error that matters, so keep both ends
    if len(block) > 60:
        block = block[:25] + [f"... ({len(block) - 50} lines of frames) ..."] + block[-25:]
    return "First traceback:\n" + "\n".join(block) + "\n...\nLast output:\n" + tail


def run_once(code: Path, data: Path, label: str, timeout: float) -> dict:
    """One torchrun of train_gpt.py in `code`; returns its final line's numbers."""
    shutil.rmtree(code / "logs", ignore_errors=True)
    env = {**os.environ, "DATA_PATH": str(data), "PYTHONUNBUFFERED": "1"}
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={NPROC}", "train_gpt.py"]
    t0 = time.time()
    out = code.parent / f"{label}.out"
    with open(out, "w") as f:
        p = subprocess.Popen(cmd, cwd=code, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            p.wait()
            rc = -9
    wall = time.time() - t0
    logs = sorted((code / "logs").glob("*.txt"), key=lambda q: q.stat().st_mtime)
    text = logs[-1].read_text(errors="replace") if logs else ""
    final = parse(text)
    if rc != 0 or final is None:
        why = "timed out" if rc == -9 else f"exit {rc}" if rc else "no final val_loss line"
        raise SystemExit(f"{label}: train_gpt.py failed ({why}) after {wall:.0f}s; no score. "
                         f"{failure_excerpt(out.read_text(errors='replace'))}")
    print(f"{label}: val_loss {final['val_loss']:.4f}  train_time {final['train_time_s']:.3f}s  "
          f"steps {final['steps']}  wall {wall:.0f}s", flush=True)
    return {**final, "wall_s": wall}


def parse(text: str) -> dict | None:
    """The last validation line of a finished run (step == total), or None."""
    done = [m for m in FINAL.finditer(text) if m.group(1) == m.group(2)]
    if not done:
        return None
    m = done[-1]
    return {"steps": int(m.group(2)), "val_loss": float(m.group(3)), "train_time_s": int(m.group(4)) / 1000}


def mean_se(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def p_value(losses: list[float]) -> float:
    from scipy import stats

    if len(losses) < 2 or len(set(losses)) == 1:
        return 0.0 if max(losses) < TARGET else 1.0
    return float(stats.ttest_1samp(losses, TARGET, alternative="less").pvalue)


def summarize(cand: list[dict], base: list[dict]) -> dict:
    losses = [r["val_loss"] for r in cand]
    times = [r["train_time_s"] for r in cand]
    loss, loss_se = mean_se(losses)
    t, t_se = mean_se(times)
    out = {"train_time_s": t, "train_time_s_se": t_se, "val_loss": loss, "val_loss_se": loss_se,
           "p_value": p_value(losses), "runs": len(cand), "steps": cand[-1]["steps"], "target": TARGET,
           "baseline_runs": len(base)}
    if base:
        bt, bt_se = mean_se([r["train_time_s"] for r in base])
        bl, _ = mean_se([r["val_loss"] for r in base])
        out |= {"baseline_train_time_s": bt, "baseline_val_loss": bl, "baseline_steps": base[-1]["steps"],
                "time_delta_s": t - bt, "time_delta_s_se": math.sqrt(t_se ** 2 + bt_se ** 2)}
    return out


def cuda_home(root: Path) -> str:
    """A CUDA root with the headers NVRTC needs, for kernels the subject compiles at import (torch's
    _compile_kernel includes $CUDA_HOME/include; #87's kernel also names /usr/local/cuda/include). python:3.12
    has no CUDA toolkit, so merge the include folders of NVIDIA's pip wheels (runtime, nvcc's crt/, cccl)."""
    if (Path("/usr/local/cuda") / "include" / "cuda_bf16.h").is_file():
        return "/usr/local/cuda"
    import importlib.util

    inc = root / "include"
    inc.mkdir(parents=True)
    for pkg in ("cuda_runtime", "cuda_nvcc", "cuda_cccl"):
        spec = importlib.util.find_spec(f"nvidia.{pkg}")
        src = Path(next(iter(spec.submodule_search_locations))) / "include" if spec else None
        if src is None or not src.is_dir():
            raise SystemExit(f"no CUDA headers from nvidia-{pkg.replace('_', '-')}-cu12; is it in requirements.txt?")
        shutil.copytree(src, inc, dirs_exist_ok=True)
    try:
        Path("/usr/local/cuda").symlink_to(root)
    except OSError as e:
        print(f"could not link /usr/local/cuda to the headers ({e}); CUDA_HOME still points at them", flush=True)
    return str(root)


def gpu_info() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main() -> None:
    t0 = time.time()
    deadline = t0 + EVAL_TIMEOUT_S - 300  # leave time to report
    subject = find_subject()
    rel = record_dir(subject)
    gpus = gpu_info()
    print(f"subject: {subject / rel}\nGPUs:\n{gpus}", flush=True)
    names = gpus.splitlines()
    if len(names) < NPROC or not all("H100" in n for n in names):
        raise SystemExit(f"needs {NPROC}×H100, found: {gpus or 'no GPUs'}")
    with tempfile.TemporaryDirectory(prefix="speedrun-") as tmp:
        tmp = Path(tmp)
        prepare_data(tmp / "dataroot")
        if "CUDA_HOME" not in os.environ:
            os.environ["CUDA_HOME"] = cuda_home(tmp / "cuda")
        print(f"CUDA_HOME={os.environ['CUDA_HOME']}", flush=True)
        work = tmp / "work"
        cand_code = work / "candidate" / subject.name
        shutil.copytree(subject, cand_code, symlinks=True, ignore=shutil.ignore_patterns("logs", "__pycache__"))
        cand_code = cand_code / rel
        base_code = parent_code(subject, work / "parent")
        base_code = base_code / rel if base_code is not None else None
        for code in (cand_code, base_code):
            if code is not None:
                prefetch_kernels(code)
        # Candidate and parent alternate (c, p, c, c, p, ...) so both see the node's drift; the parent goes
        # second so the candidate's first run (which compiles) can't make the parent look faster.
        every = max(1, RUNS // max(BASELINE_RUNS, 1))
        order = []
        for i in range(RUNS):
            order.append("candidate")
            if base_code is not None and i % every == 0 and order.count("parent") < BASELINE_RUNS:
                order.append("parent")
        cand: list[dict] = []
        base: list[dict] = []
        longest = 0.0
        for i, who in enumerate(order):
            if time.time() + longest * 1.2 > deadline:
                print(f"stopping early: not enough time for another run ({len(cand)} candidate, {len(base)} parent)",
                      flush=True)
                break
            first = not (cand if who == "candidate" else base)
            timeout = min(RUN_TIMEOUT_S if first else RUN_TIMEOUT_S / 2, max(60.0, deadline - time.time()))
            r = run_once(cand_code if who == "candidate" else base_code, tmp / "dataroot", f"{who}-{i}", timeout)
            (cand if who == "candidate" else base).append(r)
            longest = max(longest, r["wall_s"] / 2 if first else r["wall_s"])  # a first run includes compiling
    if len(cand) < MIN_RUNS:
        raise SystemExit(f"only {len(cand)} candidate runs finished in the time limit (need {MIN_RUNS}); no score")
    res = summarize(cand, base)
    res |= {"gpus": len(names), "gpu": names[0].split(",")[0].strip(), "driver": names[0].split(",")[-1].strip(),
            "wall_s": time.time() - t0}
    print(f"candidate: train_time {res['train_time_s']:.3f} ± {res['train_time_s_se']:.3f}s, val_loss "
          f"{res['val_loss']:.4f} ± {res['val_loss_se']:.4f} (p = {res['p_value']:.2g}, n = {res['runs']})", flush=True)
    if base:
        print(f"parent on this node: train_time {res['baseline_train_time_s']:.3f}s, val_loss "
              f"{res['baseline_val_loss']:.4f}; delta {res['time_delta_s']:+.3f} ± {res['time_delta_s_se']:.3f}s",
              flush=True)
    # A miss is reported, not raised: the times and losses still show in the tree, and meets_target says whether the
    # time counts. (Raised, it would read as a crash, which hides the numbers and can't be told from a real one.)
    res["meets_target"] = int(res["val_loss"] < TARGET and res["p_value"] < ALPHA)
    if not res["meets_target"]:
        print(f"val_loss {res['val_loss']:.4f} over {res['runs']} runs is not below {TARGET} with p < {ALPHA} "
              f"(p = {res['p_value']:.3g}): the time does not count (meets_target = 0)", flush=True)
    report(res)


def check_parse(paths: list[str]) -> None:
    """Parse saved logs as if each were a candidate run (a local check of parse and summarize)."""
    runs = []
    for p in paths:
        r = parse(Path(p).read_text(errors="replace"))
        if r is None:
            raise SystemExit(f"{p}: no final val_loss line")
        runs.append(r)
        print(f"{Path(p).name}: {r}")
    half = len(runs) // 2
    print(json.dumps(summarize(runs, runs[:half]), indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--parse"]:
        check_parse(sys.argv[2:])
    else:
        main()
