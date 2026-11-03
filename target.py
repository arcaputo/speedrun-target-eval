"""speedrun-target: will the benchmark count this candidate's time, and how many steps does its loss margin pay for?

A Kupanga QA eval, run after the agent's turn: its critique is the next turn's prompt. During the turn the agent
measures the same thing with speedrun-probe's full-run-1gpu experiment proc (repeats, a num_iterations parameter, a
parent baseline), which reports num_iterations_for_target by the same rule; QA jobs get no repeats, so this eval
runs its 8 copies itself. It runs the candidate's whole training run once on each H100 of an 8×H100
machine, in parallel. One GPU takes 8 gradient-accumulation micro-steps per step, the same math as 8×H100: over 24
paired runs its final val_loss was unbiased against the benchmark's (+0.0002, SD 0.0016), so these are 8 draws of
the benchmark's own loss.

From them it predicts speedrun-8xh100's verdict. The benchmark times 8 runs and counts the time only if their mean
val_loss is below 3.28 with a one-sided t-test p < 0.01. The prediction is simulated: the true mean is drawn from
what the 8 runs say, and 8 benchmark runs are drawn around it. It also finds the step count at which that chance is
PASS_GOAL, using the loss cost of a step measured near this record's 2380 steps (SLOPE, with its uncertainty).

Two blocking gates:
  target_chance   the predicted chance the benchmark counts the time is below PASS_GOAL (0.8): the candidate will
                  probably miss the target. The critique says how many steps to add.
  unspent_margin  the loss margin would pay for cutting UNSPENT_STEPS (30) or more steps and still pass with
                  PASS_GOAL: the candidate leaves about 1.8 s or more on the table. The critique says how many to cut.

Why: in nanogpt-jumps-probes, 11 agents chose step cuts from single-seed probes whose noise (σ = 0.0015) is worth
about 25 steps, and no QA gate checked the target, so cuts 10–20 steps too deep passed QA and missed it.
notes/evals-v2/hypotheses.md has the evidence.

The candidate's failures (a crash, a run that doesn't finish) are results with a critique. A failure that isn't the
candidate's (not enough H100s, a download, fewer than MIN_RUNS runs for a reason of the machine's) exits 0 with no
result, which Kupanga marks broken, as speedrun-probe does.

Local check, no GPU:
  python target.py --predict 2380 3.2779 3.2801 3.2765 ...    (num_iterations, then the runs' final val losses)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

RESULT_FILE = os.environ.pop("HARNESS_RESULT_FILE", None)  # before anything runs the code under test


def report(result: dict) -> None:
    line = json.dumps(result, allow_nan=False)
    print(line, flush=True)
    if RESULT_FILE:
        with open(RESULT_FILE, "w") as f:
            f.write(line + "\n")


import subprocess  # noqa: E402

import bench  # noqa: E402  (speedrun-8xh100's bench.py: data, CUDA headers, hub kernels, failure_excerpt)
import probe  # noqa: E402  (speedrun-probe's probe.py: patching, reading run output, stopping a torchrun)
from probe import InfraError  # noqa: E402

TARGET = bench.TARGET  # 3.28
ALPHA = bench.ALPHA  # 0.01
BENCH_RUNS = 8  # candidate runs speedrun-8xh100 scores
PASS_GOAL = 0.8
UNSPENT_STEPS = 30  # ≈ 1.8 s on 8×H100 at ~62 ms/step, and > 2 SE of the recommendation (about ±9 steps)
MIN_RUNS = 6
# Run-to-run SD of the final val_loss: 0.0015 over 216 candidate and 72 parent 8×H100 runs of this record family
# (no machine effect, ANOVA p = 0.77). The runs' own SD is pooled with it as SIGMA_PRIOR_DF runs' worth.
SIGMA_PRIOR = 0.0015
SIGMA_PRIOR_DF = 24
# val_loss cost of cutting one step near 2380 steps: 5.9e-5 ± 1.8e-5 on 8×H100 (5.7e-5 from 1-GPU full runs).
SLOPE = 5.9e-5
SLOPE_SE = 1.8e-5
# A run this far from the others' median (5 run SDs) is set aside when it is the only one: about 1 run in 200 of this
# record family lands that far out, and one such run would otherwise set the SD and the step advice for all eight
# (backout-target-5: one run at 3.2900 among seven near 3.278 asked for +67 steps; full-run-1gpu then said -92).
OUTLIER = 5 * SIGMA_PRIOR
BIAS = 0.0002  # 1-GPU minus 8×H100 final val_loss, mean over 24 pairs: subtracted before predicting
SIMS = 20000
SEED = 1234  # fixed, so the same runs always give the same verdict
SEARCH = 400  # steps either way the recommendation searches
REPORT_RESERVE_S = 90
TRAIN_SHARDS = 10  # 2640 steps of 262k tokens is 692M
CRITIQUE_MAX = 8000
STAGES = ["downloading data", "compiling candidate", "training candidate", "validating candidate",
          "predicting the benchmark"]


def stage(name: str) -> None:
    print(f"stage: {name}", flush=True)


# ---------------------------------------------------------------- the prediction (no GPU)

def t_crit(df: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(1 - ALPHA, df))


class Predictor:
    """The chance speedrun-8xh100 counts the time, for the measured runs shifted by a step change. Common random
    numbers across shifts, so the chance is monotone in the shift and the search is exact."""

    def __init__(self, losses: list[float]):
        import numpy as np

        n = len(losses)
        self.n = n
        self.mean = sum(losses) / n - BIAS
        s2 = sum((x - self.mean - BIAS) ** 2 for x in losses) / (n - 1) if n > 1 else 0.0
        self.sd = math.sqrt(((n - 1) * s2 + SIGMA_PRIOR_DF * SIGMA_PRIOR ** 2) / (n - 1 + SIGMA_PRIOR_DF))
        self.se = self.sd / math.sqrt(n)
        rng = np.random.default_rng(SEED)
        z_mu = rng.standard_normal(SIMS)
        z_runs = rng.standard_normal((SIMS, BENCH_RUNS))
        self.z_slope = rng.standard_normal(SIMS)
        self.mu = self.mean + self.se * z_mu  # where the true mean may be, given the runs
        self.run_mean = z_runs.mean(axis=1)
        self.run_sd = z_runs.std(axis=1, ddof=1)
        self.tc = t_crit(BENCH_RUNS - 1)
        self.np = np

    def chance(self, cut: int = 0) -> float:
        """P(benchmark passes) with `cut` fewer steps (negative: more steps)."""
        np = self.np
        slope = np.maximum(SLOPE + SLOPE_SE * self.z_slope, 0.0)
        mu = self.mu + slope * cut
        m = mu + self.sd * self.run_mean
        s = self.sd * self.run_sd
        t = (TARGET - m) / (s / math.sqrt(BENCH_RUNS))
        return float(np.mean((m < TARGET) & (t > self.tc)))

    def best_cut(self) -> int:
        """The most steps that can be cut (negative: must be added) with chance ≥ PASS_GOAL."""
        lo, hi = -SEARCH, SEARCH
        if self.chance(lo) < PASS_GOAL:
            return lo
        if self.chance(hi) >= PASS_GOAL:
            return hi
        while hi - lo > 1:  # chance(lo) ≥ goal > chance(hi)
            mid = (lo + hi) // 2
            if self.chance(mid) >= PASS_GOAL:
                lo = mid
            else:
                hi = mid
        return lo

    def cut_se(self) -> float:
        """Rough SE of best_cut in steps: the mean's SE through the slope, and the slope's own uncertainty."""
        c = abs(self.best_cut())
        return math.hypot(self.se / SLOPE, c * SLOPE_SE / SLOPE)


def required_mean(sd: float) -> float:
    """The mean val_loss at which 8 runs with this SD just reach p = 0.01."""
    return TARGET - t_crit(BENCH_RUNS - 1) * sd / math.sqrt(BENCH_RUNS)


def verdict(losses: list[float], num_iterations: int | None, s_per_step_8gpu: float | None) -> dict:
    """The eval's result from the runs' final val losses."""
    all_losses = losses
    med = sorted(losses)[len(losses) // 2]
    far = [x for x in losses if abs(x - med) > OUTLIER]
    if len(far) == 1:
        losses = [x for x in losses if x not in far]
    pr = Predictor(losses)
    chance = pr.chance(0)
    cut = pr.best_cut()
    cut_se = pr.cut_se()
    req = required_mean(pr.sd)
    sec = (lambda k, word: f", about {abs(k) * s_per_step_8gpu:.1f} s {word} on 8×H100" if s_per_step_8gpu else "")
    where = (lambda k: f"num_iterations {num_iterations} → {num_iterations - k}" if num_iterations is not None
             else f"num_iterations {'−' if k > 0 else '+'}{abs(k)}")
    rule = (f"The benchmark (speedrun-8xh100) times 8 runs on 8×H100 and counts the time only if their mean "
            f"val_loss is below {TARGET} with a one-sided t-test p < {ALPHA}; with run-to-run SD {pr.sd:.4f} that "
            f"needs a mean of about {req:.4f} or lower, and about {req - 0.84 * pr.sd / math.sqrt(BENCH_RUNS):.4f} "
            f"to pass {PASS_GOAL:.0%} of the time.")
    measured = (f"{len(losses)} whole runs of the candidate (one H100 each, the benchmark's math) averaged val_loss "
                f"{pr.mean + BIAS:.4f} ± {pr.se:.4f} (SE; runs: {', '.join(f'{x:.4f}' for x in losses)}).")
    if len(far) == 1:
        measured += (f" One more run finished at {far[0]:.4f}, more than {OUTLIER:.4f} from the others' median, and is left "
                     "out of this prediction: about 1 run in 200 lands that far out. The benchmark can draw one too, which "
                     "this chance doesn't include.")
    elif far:
        measured += (f" {len(far)} runs finished more than {OUTLIER:.4f} from the median ({', '.join(f'{x:.4f}' for x in far)}): "
                     "the candidate's training is unstable, which the benchmark will see too.")
    lever = (f"Each step cut costs about {SLOPE:.1e} val_loss here (±{SLOPE_SE:.0e}), measured on this record's "
             "runs near 2380 steps.")
    check = (lambda k: "Before committing, measure the new count during the turn with speedrun-probe's full-run-1gpu "
             f"(num_iterations = {num_iterations - k if num_iterations is not None else 'the new count'}, repeats = 4: "
             "about ±13 steps; its num_iterations_for_target uses this rule). One run can't resolve it.")
    gates, critique, warnings = [], [], []
    ok_chance = chance >= PASS_GOAL
    gates.append({"name": "target_chance", "kind": "gate", "blocking": True, "passed": ok_chance,
                  "value": round(chance, 3), "required": PASS_GOAL, "unit": "probability",
                  "margin": round((chance - PASS_GOAL) / (1 - PASS_GOAL), 3)})
    if not ok_chance:
        fix = (f"to reach {PASS_GOAL:.0%}: {where(cut)} ({-cut:+d} steps, ±{cut_se:.0f}{sec(cut, 'slower')})"
               if cut > -SEARCH else f"no step count within {SEARCH} steps reaches {PASS_GOAL:.0%}")
        critique.append(f"[FAIL] target_chance: predicted chance the benchmark counts this candidate's time is "
                        f"{chance:.0%} (needs ≥ {PASS_GOAL:.0%}). {measured} {rule} {lever} Change the step count "
                        f"{fix}, or make the change itself reach a lower loss at the same steps. {check(cut)}")
    ok_unspent = cut < UNSPENT_STEPS
    gates.append({"name": "unspent_margin", "kind": "gate", "blocking": True, "passed": ok_unspent,
                  "value": cut, "required": UNSPENT_STEPS, "unit": "steps",
                  "margin": round((UNSPENT_STEPS - cut) / UNSPENT_STEPS, 3)})
    if not ok_unspent:
        critique.append(f"[FAIL] unspent_margin: the loss margin pays for cutting about {cut} steps (±{cut_se:.0f}) "
                        f"and still passing {PASS_GOAL:.0%} of the time: {where(cut)}{sec(cut, 'faster')}. The "
                        f"speedrun scores time, so margin left under the target is time not taken. {measured} "
                        f"{rule} {lever} Spend it (fewer steps, or anything else that is faster at the same loss). "
                        f"{check(cut)}")
    elif ok_chance and cut >= UNSPENT_STEPS // 2:
        warnings.append(f"unspent_margin: about {cut} steps (±{cut_se:.0f}) of loss margin are unspent ({where(cut)} "
                        f"would still pass {PASS_GOAL:.0%} of the time{sec(cut, 'faster')}).")
    caveats = (f"Predicted chance {chance:.0%} that the benchmark counts the time; the step count for "
               f"{PASS_GOAL:.0%} is {where(cut)} (±{cut_se:.0f}). {measured} The step cost is this record family's, "
               "measured near 2380 steps: a change that alters how loss falls late in training can differ. The "
               "benchmark's own 8 runs are new draws, so even at 80% one node in five misses.")
    out = {"problems": sum(1 for g in gates if g["blocking"] and not g["passed"]),
           "critique": critique + ([caveats] if critique else []), "gates": gates, "caveats": caveats,
           "val_loss": round(pr.mean + BIAS, 5), "val_loss_se": round(pr.se, 5), "val_loss_sd": round(pr.sd, 5),
           "runs": len(all_losses), "run_losses": ", ".join(f"{x:.4f}" for x in all_losses),
           "pass_chance": round(chance, 4), "margin": round(TARGET - pr.mean, 5), "required_mean": round(req, 5),
           "recommended_cut": cut, "recommended_cut_se": round(cut_se, 1)}
    if warnings:
        out["warnings"] = warnings
    if num_iterations is not None:
        out |= {"num_iterations": num_iterations, "recommended_num_iterations": num_iterations - cut}
    return out


# ---------------------------------------------------------------- running 8 at once

def h100s() -> int:
    return sum(1 for line in bench.gpu_info().splitlines() if "H100" in line)


def launch(code: Path, data: Path, cache: Path, gpu: int) -> tuple[subprocess.Popen, Path]:
    shutil.rmtree(code / "logs", ignore_errors=True)
    out = code.parent / f"run{gpu}.out"
    env = {**os.environ, "DATA_PATH": str(data), "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu),
           "TORCHINDUCTOR_CACHE_DIR": str(cache / f"inductor{gpu}"), "TRITON_CACHE_DIR": str(cache / f"triton{gpu}")}
    cmd = ["torchrun", "--nnodes=1", "--nproc_per_node=1", f"--master_port={29600 + gpu}", "train_gpt.py"]
    with open(out, "w") as f:
        p = subprocess.Popen(cmd, cwd=code, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return p, out


def run_all(codes: list[Path], data: Path, cache: Path, deadline: float) -> list[dict]:
    """Each code on its own GPU, until every run has printed its final validation line, exited, or the deadline."""
    procs = [launch(c, data, cache, i) for i, c in enumerate(codes)]
    state = [{"done": False, "rc": None, "timed_out": False} for _ in procs]
    stage("compiling candidate")
    shown = set()
    t0 = time.time()
    while not all(s["done"] for s in state):
        time.sleep(2)
        for (p, out), s in zip(procs, state):
            if s["done"]:
                continue
            text = out.read_text(errors="replace")
            lines = probe.train_lines(text)
            if lines and "training" not in shown:
                shown.add("training")
                stage("training candidate")
            if lines and lines[-1][0] >= lines[-1][1] and "validating" not in shown:
                shown.add("validating")
                stage("validating candidate")
            if probe.final_line(text) is not None:
                s["done"] = True
            elif p.poll() is not None:
                s.update(done=True, rc=p.returncode)
            elif time.time() > deadline:
                s.update(done=True, timed_out=True)
    results = []
    for (p, out), s in zip(procs, state):
        probe.stop(p)
        text = out.read_text(errors="replace")
        results.append({**s, "text": text, "reached": probe.reached(text), "wall_s": time.time() - t0,
                        "finished": s["rc"] is not None, "timeout": deadline - t0})
    return results


def main() -> None:
    t0 = time.time()
    limit = float(os.environ.get("HARNESS_EVAL_TIMEOUT_S") or 3300)
    deadline = t0 + limit - REPORT_RESERVE_S
    print(f"speedrun-target, limit {limit:.0f} s (HARNESS_SELFCHECK={os.environ.get('HARNESS_SELFCHECK', '')})",
          flush=True)
    try:
        subject = bench.find_subject()
        rel = bench.record_dir(subject)
        gpus = h100s()
        print(f"subject: {subject / rel}\nGPUs:\n{bench.gpu_info()}", flush=True)
        if gpus < MIN_RUNS:
            raise InfraError(f"needs at least {MIN_RUNS} H100s, found {gpus}")
        with tempfile.TemporaryDirectory(prefix="speedrun-target-") as tmp:
            tmp = Path(tmp)
            first = tmp / "work0" / subject.name
            shutil.copytree(subject, first, symlinks=True, ignore=shutil.ignore_patterns("logs", "__pycache__"))
            probe.set_hyperparameter(first / rel, "val_loss_every", 0)  # only the final validation
            num_iterations = probe.read_int(first / rel / "train_gpt.py", "num_iterations")
            codes = [first / rel]
            for i in range(1, gpus):
                d = tmp / f"work{i}" / subject.name
                shutil.copytree(first, d, symlinks=True)
                codes.append(d / rel)
            stage("downloading data")
            data = probe.download(tmp / "dataroot", TRAIN_SHARDS)
            os.environ.setdefault("CUDA_HOME", bench.cuda_home(tmp / "cuda"))
            bench.prefetch_kernels(codes[0])
            runs = run_all(codes, data, tmp / "cache", deadline)
    except SystemExit as e:  # bench's helpers end with SystemExit on a download or kernel failure
        raise InfraError(str(e.code)) from None
    finals = [probe.final_line(r["text"]) for r in runs]
    losses = [f[1] for f in finals if f is not None and math.isfinite(f[1])]
    for i, (r, f) in enumerate(zip(runs, finals)):
        print(f"run {i}: " + (f"val_loss {f[1]:.4f} after {f[0]} steps, train_time {f[2] / 1000:.1f} s" if f else
                              f"no final val_loss (reached step {r['reached']})"), flush=True)
    diverged = [f for f in finals if f is not None and not math.isfinite(f[1])]
    failed = [r for r, f in zip(runs, finals) if f is None]
    crashed = [r for r in failed if r["finished"]]
    if diverged or crashed or (failed and len(losses) < MIN_RUNS):
        why = []
        if diverged:
            why.append(f"[FAIL] training diverged: the final val loss is not finite in {len(diverged)} of {len(runs)} "
                       "whole runs on one H100 each.")
        bad = crashed or failed
        if bad:
            why.append(probe.failure(bad[0], rel, "the end of its run and the final validation",
                                     f" ({len(bad)} of {len(runs)} runs)"))
        report({"problems": 1, "critique": why, "runs": len(losses), "wall_s": round(time.time() - t0, 1)})
        return
    if len(losses) < MIN_RUNS:
        raise InfraError(f"only {len(losses)} runs finished")
    stage("predicting the benchmark")
    done = [f for f in finals if f is not None]
    s_per_step = sum(f[2] for f in done) / len(done) / 1000 / done[0][0] / 8  # 1 GPU ≈ 8× the benchmark's time
    res = verdict(losses, num_iterations, s_per_step)
    print("\n".join(res["critique"]) or res["caveats"], flush=True)
    report(res | {"wall_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    if sys.argv[1:2] == ["--predict"]:
        print(json.dumps(verdict([float(x) for x in sys.argv[3:]], int(sys.argv[2]), 0.0617), indent=1))
    else:
        try:
            main()
        except InfraError as e:
            print(f"speedrun-target: not the candidate's failure, no result: {e}", flush=True)
            sys.exit(0)
        except Exception:
            import traceback
            traceback.print_exc()
            print("speedrun-target: the eval itself failed, no result", flush=True)
            sys.exit(0)
