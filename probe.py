"""Probes for the modded-nanogpt speedrun on one H100: bounded experiments the agent runs during its turn.

Kupanga starts this with PROBE_KIND (an experiment proc under [eval.procs] in eval.toml), PROBE_<PARAM> for each parameter
(PROBE_STEPS, PROBE_NUM_ITERATIONS) and HARNESS_PROBE=1, plus the usual HARNESS_* variables. Parameters, bounds,
defaults and max_wall_s are read from eval.toml, so they are stated once.

  loss-vs-parent   the parent's code (rebuilt with bench.parent_code) and then the candidate's, each trained for
                   PROBE_STEPS steps of its own LR schedule (num_iterations isn't touched: the run is stopped after
                   the step, as speedrun-perf does), validating every 100 steps. Outputs val loss at each 100th step
                   for both, ms/step for both (slope over steps 50..N), and the loss delta at the last one.
  full-run-1gpu    the candidate's whole run, optionally with num_iterations = PROBE_NUM_ITERATIONS, validating only
                   at the end. Outputs the final val loss, planned steps, 1-GPU train time and ms/step.

Both run a patched temporary copy of the code: `val_loss_every` (and for full-run-1gpu `num_iterations`) in the
record's Hyperparameters is set to the probe's value. If that assignment isn't found exactly once, the probe ends
with no result (a broken eval for the researcher), never silently running unpatched code.

A crash of the candidate's code (or a candidate run that doesn't finish) is a result with a `critique` holding its
first traceback. A failure that isn't the candidate's (no H100, a download, a kernel fetch, rebuilding the parent,
the parent's own run, a pattern not found, out of time) prints why and exits 0 with no result, as speedrun-perf does.

Environment variables (besides Kupanga's): SPEEDRUN_PROBE_TRAIN_SHARDS (3 for loss-vs-parent, 10 for full-run-1gpu),
SPEEDRUN_PROBE_SAVE_DIR (unset: where to keep each run's output, for --parse).

Local check, no GPU:
  python probe.py --parse loss-vs-parent <parent.out> <candidate.out> [steps]
  python probe.py --parse full-run-1gpu <candidate.out>
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

RESULT_FILE = os.environ.pop("HARNESS_RESULT_FILE", None)  # before anything runs the code under test


def report(result: dict) -> None:
    """Print the result, and write it to the file Kupanga reads it from (docs/evals.md)."""
    line = json.dumps(result, allow_nan=False)
    print(line, flush=True)
    if RESULT_FILE:
        with open(RESULT_FILE, "w") as f:
            f.write(line + "\n")


import bench  # noqa: E402  (speedrun-8xh100's bench.py, shipped with this eval: the same setup it uses)

HERE = Path(__file__).resolve().parent
LOSS = "loss-vs-parent"
FULL = "full-run-1gpu"
VAL_EVERY = 100  # loss-vs-parent's validation interval
SLOPE_START = 50  # ms/step is fitted from here (skips warmup and the first compiles), as speedrun-perf
MIN_POINTS = 10
GPU_FREE_MIB = 2048  # GPU memory in use above this before a run means something else still holds the GPU
CRITIQUE_MAX = 8000
REPORT_RESERVE_S = 60  # time kept back from the eval's limit to stop the run and report
TRAIN_SHARDS = {LOSS: 3, FULL: 10}  # 500 steps of ≤ 393k tokens is < 200M; 2640 steps of 262k is 692M
SAVE_DIR = os.environ.get("SPEEDRUN_PROBE_SAVE_DIR")
# train_gpt.py's two step lines (modded-nanogpt records and modded-nanogpt-jumps print the same):
#   step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:...        (training)
#   step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms ...  (validation)
TRAIN_STEP = re.compile(r"^step:(\d+)/(\d+) train_time:(\d+)ms", re.M)
VAL_STEP = re.compile(r"^step:(\d+)/(\d+) val_loss:(-?[\d.]+|nan|-?inf) train_time:(\d+)ms", re.M)
SEED_NOTE = ("train_gpt.py seeds nothing itself, so both codes start from torch's default seed, but GPU kernels "
             "aren't bit-deterministic")


NL = "\n"


class InfraError(Exception):
    """A failure that isn't the candidate's: the probe ends with no result."""


def stage(name: str) -> None:
    """A stage Kupanga may show the agent (eval.toml lists each kind's stages)."""
    print(f"stage: {name}", flush=True)


# ---------------------------------------------------------------- kinds and parameters

def load_kinds(path: Path = HERE / "eval.toml") -> dict:
    """The experiment procs of eval.toml ([eval.procs.<kind>]; [eval.probe.<kind>] before procs), by kind."""
    body = tomllib.loads(path.read_text())["eval"]
    if "procs" not in body:
        return body["probe"]
    return {k: v for k, v in body["procs"].items() if v.get("class") == "experiment"}


def parse_params(kind: str, env: dict | None = None, kinds: dict | None = None) -> dict:
    """The kind's parameters from PROBE_<NAME>, checked against eval.toml: a default (or None, when optional) for an
    absent one. Kupanga checks them before the job starts; anything wrong here is the eval's setup, not the
    candidate's, so it raises InfraError."""
    env = os.environ if env is None else env
    kinds = load_kinds() if kinds is None else kinds
    if kind not in kinds:
        raise InfraError(f"PROBE_KIND={kind!r} is not one of this eval's probes ({', '.join(sorted(kinds))})")
    out: dict = {}
    for name, spec in kinds[kind].get("params", {}).items():
        var = f"PROBE_{name.upper()}"
        raw = (env.get(var) or "").strip()
        if not raw:
            if "default" in spec:
                out[name] = spec["default"]
            elif spec.get("optional"):
                out[name] = None
            else:
                raise InfraError(f"{var} is required for the {kind} probe")
            continue
        typ = spec["type"]
        try:
            value = int(raw) if typ == "int" else float(raw) if typ == "float" else raw
        except ValueError:
            raise InfraError(f"{var}={raw!r} is not an {typ}") from None
        if typ == "enum" and value not in spec.get("values", []):
            raise InfraError(f"{var}={raw!r} is not one of {spec.get('values')}")
        if "min" in spec and value < spec["min"] or "max" in spec and value > spec["max"]:
            raise InfraError(f"{var}={value} is outside {spec.get('min')}–{spec.get('max')}")
        out[name] = value
    return out


# ---------------------------------------------------------------- patching the temporary copy

def assignment(name: str) -> re.Pattern:
    """`name = …` or `name: type = …` at the start of a line (a dataclass field or a plain assignment), not
    `args.name = …` and not `name == …`; the right-hand side up to a trailing comment."""
    return re.compile(rf"^(?P<head>[ \t]*{name}(?:[ \t]*:[^=\n]+?)?[ \t]*=)(?!=)[ \t]*(?P<rhs>[^#\n]*?)"
                      rf"(?P<tail>[ \t]*#.*)?$", re.M)


def set_hyperparameter(code: Path, name: str, value: int) -> tuple[Path, str]:
    """Set `name` to `value` in the code's single assignment of it: train_gpt.py first, then the record's other
    top-level modules (newer records keep Hyperparameters in one). Returns the file and the old right-hand side.
    Raises InfraError when there isn't exactly one assignment, so the probe never runs code it didn't patch."""
    pat = assignment(name)
    files = [code / "train_gpt.py"] + sorted(p for p in code.glob("*.py")
                                             if p.name not in ("train_gpt.py", "train_gpt_medium.py"))
    hits = [(f, m) for f in files if f.is_file() for m in pat.finditer(f.read_text(errors="replace"))]
    where = ", ".join(f"{f.name} line {f.read_text(errors='replace').count(NL, 0, m.start()) + 1}" for f, m in hits)
    if len(hits) != 1:
        found = f"{len(hits)} assignments ({where})" if hits else "no assignment"
        raise InfraError(f"can't set {name} = {value} for the probe: found {found} of `{name} = …` in "
                         f"{', '.join(f.name for f in files if f.is_file())}; the probe needs exactly one, in the "
                         "record's Hyperparameters")
    f, m = hits[0]
    text = f.read_text()
    new = text[:m.start()] + f"{m.group('head')} {value}{m.group('tail') or ''}" + text[m.end():]
    f.write_text(new)
    check = [x.group("rhs").strip() for x in pat.finditer(new)]
    if check != [str(value)]:  # can't happen with the regex above; a guard against silently running unpatched code
        raise InfraError(f"patching {name} in {f.name} didn't take: now reads {check}")
    old = m.group("rhs").strip()
    print(f"probe copy: {f.name} {name} = {old} -> {value}", flush=True)
    return f, old


# ---------------------------------------------------------------- reading a run's output (no GPU)

def train_lines(text: str) -> list[tuple[int, int, float]]:
    return [(int(m.group(1)), int(m.group(2)), float(m.group(3))) for m in TRAIN_STEP.finditer(text)]


def val_lines(text: str) -> dict[int, tuple[int, float, float]]:
    """step -> (total, val_loss, train_time ms), the last line for each step."""
    return {int(m.group(1)): (int(m.group(2)), float(m.group(3)), float(m.group(4))) for m in VAL_STEP.finditer(text)}


def reached(text: str) -> int:
    return max((k for k, _, _ in train_lines(text)), default=0)


def slope(points: list[tuple[int, float]]) -> float:
    """Least-squares slope of train_time (ms) over step."""
    n = len(points)
    mx = sum(k for k, _ in points) / n
    my = sum(t for _, t in points) / n
    sxx = sum((k - mx) ** 2 for k, _ in points)
    if sxx == 0:
        raise ValueError("every point is at the same step")
    return sum((k - mx) * (t - my) for k, t in points) / sxx


def ms_per_step(text: str, end: int, start: int = SLOPE_START) -> float:
    """The slope of train_time over the training lines from `start` to `end`. train_time leaves out validation."""
    points = sorted({k: t for k, _, t in train_lines(text) if start <= k <= end}.items())
    if len(points) < MIN_POINTS:
        raise ValueError(f"only {len(points)} training step lines between steps {start} and {end} (need {MIN_POINTS})")
    return slope(points)


def val_schedule(steps: int) -> list[int]:
    return list(range(VAL_EVERY, steps + 1, VAL_EVERY))


def measure_loss_run(text: str, steps: int) -> dict:
    """val loss at every 100th step up to `steps`, ms/step over steps 50..steps, planned total. Raises ValueError
    when a validation line or the step lines are missing."""
    vals = val_lines(text)
    want = val_schedule(steps)
    missing = [k for k in want if k not in vals]
    if missing:
        raise ValueError(f"no val_loss line at step(s) {', '.join(map(str, missing))} (validation every {VAL_EVERY} "
                         "steps was patched in, but the run didn't print it)")
    lines = train_lines(text)
    return {"val_steps": want, "val_loss": [vals[k][1] for k in want], "ms_per_step": ms_per_step(text, steps),
            "steps": lines[-1][1] if lines else vals[want[-1]][0]}


def loss_run_done(steps: int) -> Callable[[str], bool]:
    """When a loss-vs-parent run can be stopped: training step `steps` printed and the last validation line too
    (validation at step k runs after training line k). Also once it is two steps past without that line, so a run
    whose validation didn't happen ends quickly and is reported as such."""
    last_val = val_schedule(steps)[-1]

    def done(text: str) -> bool:
        k = reached(text)
        return k >= steps and (last_val in val_lines(text) or k >= steps + 2)
    return done


def final_line(text: str) -> tuple[int, float, float] | None:
    """(steps, val_loss, train_time ms) of the validation line after the last step, or None."""
    done = [(k, loss, t) for k, (total, loss, t) in val_lines(text).items() if k == total]
    return done[-1] if done else None


def full_run_done(text: str) -> bool:
    return final_line(text) is not None


def finite(x: float) -> float | None:
    return round(x, 4) if math.isfinite(x) else None


# ---------------------------------------------------------------- running (one H100)

def descendants(pid: int) -> list[int]:
    """Every process under `pid`, children before grandchildren."""
    parents: dict[int, list[int]] = {}
    if Path("/proc/self/stat").exists():
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()  # the name, in parentheses, may hold spaces
            except (OSError, IndexError):
                continue
            parents.setdefault(int(fields[1]), []).append(int(stat.parent.name))
    else:  # no /proc (macOS, where the tests run)
        out = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid="], capture_output=True, text=True).stdout
        for line in out.splitlines():
            child, parent = map(int, line.split())
            parents.setdefault(parent, []).append(child)
    out, todo = [], [pid]
    while todo:
        kids = parents.get(todo.pop(0), [])
        out += kids
        todo += kids
    return out


def stop(p: subprocess.Popen) -> None:
    """End a torchrun and everything under it (speedrun-perf's stop): torchrun starts each worker in a session of
    its own, so killing torchrun's process group would leave the worker training and holding the GPU."""
    tree = descendants(p.pid)  # before torchrun exits and its children move to init
    try:
        os.kill(p.pid, signal.SIGTERM)
        p.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    for pid in [p.pid, *tree, *(q for t in tree for q in descendants(t))]:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    p.wait()


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def wait_gpu_free(limit_s: float = 120) -> None:
    """The next run must have the GPU to itself: a run sharing it would be measured slower than it is."""
    t0 = time.time()
    while (used := gpu_used_mib()) is not None and used > GPU_FREE_MIB:
        if time.time() - t0 > limit_s:
            raise InfraError(f"{used} MiB of GPU memory still in use {limit_s:.0f} s after the previous run ended")
        time.sleep(2)


def run(code: Path, data: Path, cache: Path, label: str, timeout: float, done: Callable[[str], bool], *,
        final_stage: bool = False) -> dict:
    """train_gpt.py in `code` with torchrun on one GPU until done(output) or the run exits or `timeout` passes;
    then stopped with everything it started. final_stage: show a stage for the validation after the last step."""
    shutil.rmtree(code / "logs", ignore_errors=True)
    out = code.parent / f"{label}.out"
    # One Inductor and Triton cache for both codes, keyed by each graph's content: only what changed compiles twice.
    env = {**os.environ, "DATA_PATH": str(data), "PYTHONUNBUFFERED": "1",
           "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "TRITON_CACHE_DIR": str(cache / "triton")}
    wait_gpu_free()
    t0 = time.time()
    finished = timed_out = False
    with open(out, "w") as f:
        p = subprocess.Popen(["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"], cwd=code, env=env,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        stage(f"compiling {label}")
        training = validating = False
        while True:
            time.sleep(1)
            text = out.read_text(errors="replace")
            lines = train_lines(text)
            if lines and not training:
                training = True
                stage(f"training {label}")
            if final_stage and lines and lines[-1][0] >= lines[-1][1] and not validating:
                validating = True
                stage(f"validating {label}")
            if done(text):
                break
            if p.poll() is not None:
                finished = True
                break
            if time.time() - t0 > timeout:
                timed_out = True
                break
        rc = p.returncode if finished else None
        stop(p)
    text = out.read_text(errors="replace")
    if SAVE_DIR:
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy(out, Path(SAVE_DIR) / out.name)
    wall = time.time() - t0
    k = reached(text)
    print(f"{label}: reached step {k} in {wall:.0f}s" + (" (timed out)" if timed_out else
                                                          f" (exited {rc})" if finished else ""), flush=True)
    return {"text": text, "rc": rc, "finished": finished, "timed_out": timed_out, "reached": k, "wall_s": wall,
            "timeout": timeout}


def failure(r: dict, rel: Path, goal: str, note: str = "") -> str:
    """The critique for a candidate run that didn't get to `goal`: why, and its first traceback."""
    if r["timed_out"]:
        why = f"didn't get to {goal} within {r['timeout']:.0f} s (it reached training step {r['reached']})"
        try:
            why += f", at {ms_per_step(r['text'], r['reached']):.0f} ms/step on one H100"
        except ValueError:
            pass
    else:
        why = f"exited with code {r['rc']} at training step {r['reached']}, before {goal}"
    msg = f"[FAIL] on one H100, {rel}/train_gpt.py {why}.{note} {bench.failure_excerpt(r['text'])}"
    return msg if len(msg) <= CRITIQUE_MAX else msg[:3000] + "\n...\n" + msg[-(CRITIQUE_MAX - 3000):]


# ---------------------------------------------------------------- results

def loss_caveats(steps: int) -> str:
    return (f"One H100 and one seed ({SEED_NOTE}), so val loss differences of a few thousandths can be noise. Loss "
            f"at step {steps} doesn't fix the final value: most of each LR schedule, its cooldown included, is still "
            "ahead, so a lead can shrink or grow. A method whose effect changes over training (an auxiliary loss "
            "that fades, a change that switches on in a later phase) can trail far behind early and still finish "
            "level: multi-token prediction trailed its parent by 0.37 at step 300 and finished within about 0.01. "
            "For such a method, decide from full-run-1gpu's final val loss. ms/step is one GPU with 8-way gradient accumulation: no multi-GPU "
            "communication or optimizer sharding, so it isn't the benchmark's step time.")


FULL_CAVEATS = (f"One H100 and one seed ({SEED_NOTE}), so the final val loss can differ by a few thousandths from "
                "another run of the same code. train_time is one GPU with 8-way gradient accumulation, about 8× the "
                "benchmark's and without its communication: compare it only with other 1-GPU numbers. "
                "Run-to-run SD is 0.0015, and cutting a step costs about 5.9e-5 val_loss near 2380 steps, so one run "
                "can't place a step count to better than about ±25 steps. The benchmark counts the time only if 8 runs "
                "average below 3.28 with p < 0.01, about 3.2784 or lower (3.2779 to pass 80% of the time); "
                "margin_to_target is 3.28 minus this run's loss. To choose a step count, use the speedrun-target gate, "
                "which runs 8 seeds and says which num_iterations passes.")


def loss_result(parent: dict, cand: dict | None, steps: int) -> dict:
    """The loss-vs-parent result from the measured runs (cand is None when the candidate's run failed)."""
    out: dict = {"val_steps": [float(k) for k in parent["val_steps"]],
                 "parent_val_loss": [finite(x) for x in parent["val_loss"]],
                 "parent_ms_per_step": round(parent["ms_per_step"], 2), "caveats": loss_caveats(steps)}
    if cand is None:
        return out
    out |= {"candidate_val_loss": [finite(x) for x in cand["val_loss"]],
            "ms_per_step": round(cand["ms_per_step"], 2),
            "step_delta_pct": round((cand["ms_per_step"] / parent["ms_per_step"] - 1) * 100, 2)}
    a, b = cand["val_loss"][-1], parent["val_loss"][-1]
    if math.isfinite(a) and math.isfinite(b):
        out["val_loss_delta"] = round(a - b, 4)
    bad = [k for k, x in zip(cand["val_steps"], cand["val_loss"]) if not math.isfinite(x)]
    if bad:
        out["critique"] = [f"The candidate's val loss isn't finite at step(s) {', '.join(map(str, bad))}: training "
                           "diverged (null in candidate_val_loss)."]
    return out


def full_result(text: str, num_iterations: int | None) -> dict:
    """The full-run-1gpu result from a finished run's output. Raises ValueError without a final val line."""
    fin = final_line(text)
    if fin is None:
        raise ValueError("no validation line after the last step")
    steps, loss, t = fin
    out: dict = {"steps": steps, "train_time_s": round(t / 1000, 2), "caveats": FULL_CAVEATS}
    if num_iterations is not None:
        out["num_iterations"] = num_iterations
    if math.isfinite(loss):
        out["val_loss"] = round(loss, 4)
        out["margin_to_target"] = round(bench.TARGET - loss, 4)
    else:
        out["critique"] = [f"The final val loss is {loss}: training diverged."]
    try:
        out["ms_per_step"] = round(ms_per_step(text, steps), 2)
    except ValueError:
        pass
    return out


# ---------------------------------------------------------------- the probe

def download(root: Path, shards: int) -> Path:
    d = root / "data" / "fineweb10B"
    d.mkdir(parents=True)
    names = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, shards + 1)]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda n: bench.fetch(n, d), names))
    return root


def read_int(path: Path, name: str) -> int | None:
    """The integer a single literal `name = N` assignment sets, or None (absent, repeated, or not a literal)."""
    hits = list(assignment(name).finditer(path.read_text(errors="replace"))) if path.is_file() else []
    try:
        return int(hits[0].group("rhs").replace("_", "")) if len(hits) == 1 else None
    except ValueError:
        return None


def main() -> None:
    t0 = time.time()
    kinds = load_kinds()
    kind = os.environ.get("PROBE_KIND", "")
    params = parse_params(kind, kinds=kinds)
    # The eval's own limit is the kind's max_wall_s, even if the job was given longer.
    limit = float(kinds[kind]["max_wall_s"])
    if (env_limit := os.environ.get("HARNESS_EVAL_TIMEOUT_S", "").strip()):
        limit = min(limit, float(env_limit))
    deadline = t0 + limit - REPORT_RESERVE_S
    shards = int(os.environ.get("SPEEDRUN_PROBE_TRAIN_SHARDS") or TRAIN_SHARDS[kind])
    print(f"probe {kind} {params}, limit {limit:.0f} s (HARNESS_PROBE={os.environ.get('HARNESS_PROBE', '')})",
          flush=True)
    changes = os.environ.get("HARNESS_CHANGES_FILE")
    has_parent = bool(changes and Path(changes).is_file() and Path(changes).read_text().strip())
    if kind == LOSS and not has_parent:
        print("no changes: the base node has no parent to compare against", flush=True)
        report({"critique": ["No parent to compare against (the base node): nothing was measured. full-run-1gpu "
                             "measures a code on its own."], "wall_s": round(time.time() - t0, 1)})
        return
    try:
        subject = bench.find_subject()
        rel = bench.record_dir(subject)
        with tempfile.TemporaryDirectory(prefix="speedrun-probe-") as tmp:
            tmp = Path(tmp)
            work = tmp / "work"
            cand_code = work / "candidate" / subject.name
            shutil.copytree(subject, cand_code, symlinks=True, ignore=shutil.ignore_patterns("logs", "__pycache__"))
            cand_code = cand_code / rel
            base_code = None
            # Patch first: a pattern that isn't found costs no download and no GPU time.
            if kind == LOSS:
                stage("rebuilding the parent")
                base_code = bench.parent_code(subject, work / "parent") / rel
                for code in (base_code, cand_code):
                    set_hyperparameter(code, "val_loss_every", VAL_EVERY)
            else:
                set_hyperparameter(cand_code, "val_loss_every", 0)  # only the final validation
                if params["num_iterations"] is not None:
                    set_hyperparameter(cand_code, "num_iterations", params["num_iterations"])
                    num_iterations = params["num_iterations"]
                else:
                    num_iterations = read_int(cand_code / "train_gpt.py", "num_iterations")
            gpus = bench.gpu_info()
            print(f"subject: {subject / rel}\nGPUs:\n{gpus}", flush=True)
            if "H100" not in gpus:
                raise InfraError(f"needs an H100, found: {gpus or 'no GPUs'}")
            stage("downloading data")
            data = download(tmp / "dataroot", shards)
            os.environ.setdefault("CUDA_HOME", bench.cuda_home(tmp / "cuda"))
            for code in (base_code, cand_code):
                if code is not None:
                    bench.prefetch_kernels(code)
            cache = tmp / "cache"
            if kind == LOSS:
                steps = params["steps"]
                goal = f"training step {steps} and its validation"
                parent = run(base_code, data, cache, "parent", deadline - time.time(), loss_run_done(steps))
                try:
                    p = measure_loss_run(parent["text"], steps)
                except ValueError as e:
                    how = "timed out" if parent["timed_out"] else f"exit {parent['rc']}" if parent["finished"] else ""
                    raise InfraError(f"the parent's code didn't get to {goal} on this machine ({how}): {e}\n"
                                     f"{bench.failure_excerpt(parent['text'])}") from None
                cand_limit = deadline - time.time()
                cand = run(cand_code, data, cache, "candidate", cand_limit, loss_run_done(steps))
                try:
                    c = measure_loss_run(cand["text"], steps)
                except ValueError as e:
                    # The candidate's own failure: it crashed, or ran out of a fair share of time (at least what the
                    # parent took, with room for compiling what it changed). Otherwise the probe ran out of time.
                    if cand["timed_out"] and cand_limit < parent["wall_s"] * 1.25 + 60:
                        raise InfraError(f"out of time: the candidate had {cand_limit:.0f} s after the parent's "
                                         f"{parent['wall_s']:.0f} s run") from None
                    if not cand["timed_out"] and cand["reached"] >= steps:  # it ran, but validation didn't print
                        raise InfraError(f"the candidate's run: {e}") from None
                    res = loss_result(p, None, steps)
                    res["critique"] = [failure(cand, rel, goal, " The parent's code got there on the same machine.")]
                    report(res | {"wall_s": round(time.time() - t0, 1)})
                    return
                res = loss_result(p, c, steps)
                print(f"val loss at steps {p['val_steps']}: parent {p['val_loss']}, candidate {c['val_loss']}; "
                      f"ms/step parent {p['ms_per_step']:.1f}, candidate {c['ms_per_step']:.1f}", flush=True)
            else:
                cand = run(cand_code, data, cache, "candidate", deadline - time.time(), full_run_done,
                           final_stage=True)
                try:
                    res = full_result(cand["text"], num_iterations)
                except ValueError:
                    lines = train_lines(cand["text"])
                    res = {"reached_step": cand["reached"], "caveats": FULL_CAVEATS,
                           "critique": [failure(cand, rel, "the end of its run and the final validation")]}
                    if lines:
                        res["steps"] = lines[-1][1]
                        try:
                            res["ms_per_step"] = round(ms_per_step(cand["text"], cand["reached"]), 2)
                        except ValueError:
                            pass
                    if num_iterations is not None:
                        res["num_iterations"] = num_iterations
                else:
                    print(f"final val loss {res.get('val_loss')} after {res['steps']} steps, train_time "
                          f"{res['train_time_s']} s on one H100", flush=True)
    except SystemExit as e:  # bench's helpers end with SystemExit on a download, kernel or parent-rebuild failure
        raise InfraError(str(e.code)) from None
    report(res | {"wall_s": round(time.time() - t0, 1)})


def check_parse(argv: list[str]) -> None:
    """A result from saved run outputs (a local check, no GPU)."""
    kind = argv[0] if argv else ""
    if kind == LOSS and len(argv) in (3, 4):
        steps = int(argv[3]) if len(argv) == 4 else 300
        p = measure_loss_run(Path(argv[1]).read_text(errors="replace"), steps)
        c = measure_loss_run(Path(argv[2]).read_text(errors="replace"), steps)
        print(json.dumps(loss_result(p, c, steps), indent=1))
    elif kind == FULL and len(argv) == 2:
        print(json.dumps(full_result(Path(argv[1]).read_text(errors="replace"), None), indent=1))
    else:
        raise SystemExit("usage: python probe.py --parse loss-vs-parent <parent.out> <candidate.out> [steps]\n"
                         "       python probe.py --parse full-run-1gpu <candidate.out>")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--parse"]:
        check_parse(sys.argv[2:])
    else:
        try:
            main()
        except InfraError as e:
            # No result and exit 0: Kupanga marks the run broken (an eval error for the researcher to re-run)
            # instead of a crash, which it would treat as the candidate's.
            print(f"speedrun-probe: not the candidate's failure, no result: {e}", flush=True)
            sys.exit(0)
        except Exception:
            # A bug in this script is never the candidate's: a non-zero exit would be charged to the agent's
            # GPU budget as its crash. The candidate's own failures are results with a critique, not exceptions.
            import traceback
            traceback.print_exc()
            print("speedrun-probe: the probe itself failed, no result", flush=True)
            sys.exit(0)
