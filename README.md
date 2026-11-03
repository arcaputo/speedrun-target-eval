# speedrun-target

A Kupanga QA eval for the modded-nanogpt speedrun. It answers the question the benchmark asks: **will this
candidate's time count, and how many steps does its loss margin pay for?** It answers before the 8×H100 benchmark
runs, and while the agent can still act on the answer.

It runs the candidate's whole training run once on each H100 of an 8×H100 machine, in parallel. One GPU takes 8
gradient-accumulation micro-steps per step, the same math as 8×H100. Over 24 paired runs, a 1-GPU run's final
val_loss was unbiased against the benchmark's (+0.0002, SD 0.0016). So the eval gets 8 draws of the benchmark's own
loss in about 35 minutes.

## Why it exists

In nanogpt-jumps-probes, 11 agents implemented the backout jump and chose how many training steps to cut. They did
it from single-seed probes, and no QA gate checked the target. Two facts made that fail:
- **The decision is finer than a single run.** Run-to-run SD of val_loss is 0.0015, and a step costs about 5.9e-5
  of val_loss, so one run's noise is worth about 25 steps. Backout's whole gain is worth about 28 steps.
- **The loop was open.** The benchmark's verdict never reaches the agent, and every QA gate passed a candidate that
  would miss. Cuts 10–20 steps too deep went straight to the benchmark and missed on p.

`notes/evals-v2/hypotheses.md` has the audit.

## What it reports

The benchmark (speedrun-8xh100) counts the time only if its 8 runs' mean val_loss is below 3.28, with a one-sided
t-test p < 0.01. This eval simulates that rule. It draws the true mean from the 8 runs, with their SD pooled with
the record family's 0.0015, then draws 8 benchmark runs around that mean and t-tests them.

| Output | |
|---|---|
| `pass_chance` | Predicted chance the benchmark counts the time |
| `margin`, `required_mean` | 3.28 − the mean, and the mean that just reaches p = 0.01 |
| `recommended_cut`, `recommended_num_iterations` | The most steps that can be cut while passing 80% of the time (negative means add), using the step cost 5.9e-5 ± 1.8e-5; `recommended_cut_se` is its rough SE (about ±9–15 steps) |
| `val_loss`, `val_loss_se`, `val_loss_sd`, `run_losses`, `runs` | The measurement itself |

Two blocking gates:
- **`target_chance`:** `pass_chance` < 0.8, so the candidate will probably miss. The critique gives the
  `num_iterations` to use.
- **`unspent_margin`:** `recommended_cut` ≥ 30, so the loss margin pays for 30 or more steps (≈ 1.8 s) and the
  candidate leaves them unspent. The speedrun scores time, so the critique says how many steps to cut. From 15 to
  29 steps unspent is only a warning.

A crash, divergence or unfinished run is the candidate's failure, with its first traceback. A failure that isn't
the candidate's makes the run broken: fewer than 6 H100s, a download failure, or fewer than 6 runs finishing for
the machine's reasons.

It has a gate proc (`[eval.procs.target]`), so the agent can run it on its working tree during its turn as part of
its self-check. The gate takes about 35 minutes. A turn is about 55 minutes and gates don't pause the clock, so the
agent has to start it early.

## Files

| File | |
|---|---|
| `target.py` | The eval |
| `bench.py`, `probe.py` | Copies from speedrun-8xh100 and speedrun-probe (data, CUDA headers, hub kernels, patching, run parsing) |
| `requirements.txt` | speedrun-8xh100's pinned stack |
| `tests/` | `python -m pytest tests`, no GPU (a fake torchrun prints a run per GPU) |

## Local check

```bash
python target.py --predict 2380 3.2779 3.2801 3.2765 3.2790 3.2783 3.2771 3.2795 3.2788
```

## What it doesn't cover

- **The step cost** is this record family's, measured near 2380 steps. A change that alters how loss falls late
  in training can differ.
- **Step time** is speedrun-perf's job; this eval reads only loss.
- **Misses still happen.** The benchmark's 8 runs are new draws, so even at 80% one candidate in five misses.
- **The prediction trusts the run's printed val_loss**, as the benchmark does.
