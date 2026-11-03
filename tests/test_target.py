"""target.py without a GPU: python -m pytest tests

The eval.toml contract, the prediction against the benchmark's rule, and main() end to end with a fake torchrun
that prints a whole run's lines, one loss per GPU (CUDA_VISIBLE_DEVICES), as speedrun-probe's tests do.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))

import bench  # noqa: E402
import probe  # noqa: E402
import target  # noqa: E402

CFG = tomllib.loads((EVAL / "eval.toml").read_text())["eval"]


# ---------------------------------------------------------------- eval.toml

def test_eval_toml_is_a_qa_eval_without_procs():
    # in-turn measuring is speedrun-probe's full-run-1gpu experiment with repeats; this is the after-turn check
    assert (CFG["type"], CFG["evaluator"], CFG["entrypoint"]) == ("qa", "deterministic", "python target.py")
    assert CFG["pass"]["condition"] == "problems == 0" and CFG["qa"]["qa_cap"] >= 1
    assert "procs" not in CFG
    job = tomllib.loads((EVAL / "job.toml").read_text())
    assert job["timeout_s"] <= 3600
    assert job["resources"]["accelerators"] == "H100:8"


def schema_ok(r: dict) -> None:
    schema = CFG["schema"]
    for k, v in r.items():
        if k == "gates":
            continue
        assert k in schema, k
        want = {"int": int, "float": (int, float), "string": str, "list[str]": list}[schema[k]["type"]]
        assert isinstance(v, want), (k, v)
    for k, spec in schema.items():
        assert spec.get("optional") or k in r, k
    json.dumps(r, allow_nan=False)


# ---------------------------------------------------------------- the prediction

def runs(mean: float, spread=(-0.0018, -0.0011, -0.0005, -0.0001, 0.0001, 0.0005, 0.0011, 0.0018)) -> list[float]:
    return [round(mean + d, 4) for d in spread]


def test_required_mean_matches_the_t_test():
    # 8 runs with SD 0.0015: t(0.99, 7) = 2.998, so the mean must be 3.28 - 2.998 * 0.0015 / sqrt(8)
    assert target.required_mean(0.0015) == pytest.approx(3.27841, abs=1e-5)


def test_a_candidate_at_the_parents_loss_is_a_coin_flip_and_fails():
    # the base code averages 3.2782 over 38 runs on 8×H100 and missed the rule in 2 of 2 scorings
    r = target.verdict(runs(3.2784), 2380, 0.0617)
    schema_ok(r | {"wall_s": 1.0})  # main() adds wall_s
    assert 0.2 < r["pass_chance"] < 0.7 and r["problems"] == 1
    assert [g["name"] for g in r["gates"] if not g["passed"]] == ["target_chance"]
    assert r["recommended_cut"] < 0 and r["recommended_num_iterations"] > 2380
    assert "num_iterations 2380 → " in r["critique"][0] and "p < 0.01" in r["critique"][0]


def test_a_comfortable_candidate_with_a_small_margin_passes():
    r = target.verdict(runs(3.2770), 2380, 0.0617)
    assert r["problems"] == 0 and r["critique"] == [] and r["pass_chance"] >= 0.8
    assert 0 <= r["recommended_cut"] < target.UNSPENT_STEPS


def test_a_big_unspent_margin_fails_and_says_how_many_steps_to_cut():
    r = target.verdict(runs(3.2740), 2380, 0.0617)
    assert r["pass_chance"] > 0.99 and r["problems"] == 1
    assert [g["name"] for g in r["gates"] if not g["passed"]] == ["unspent_margin"]
    cut = r["recommended_cut"]
    assert cut >= target.UNSPENT_STEPS and r["recommended_num_iterations"] == 2380 - cut
    assert f"num_iterations 2380 → {2380 - cut}" in r["critique"][0] and "faster" in r["critique"][0]


def test_following_the_recommendation_lands_near_the_goal():
    # shift the runs by what the recommended cut costs at the prior slope: the chance should come out near 0.8
    for mean in (3.2740, 3.2790):
        r = target.verdict(runs(mean), 2380, 0.0617)
        after = target.verdict(runs(mean + target.SLOPE * r["recommended_cut"]), 2380 - r["recommended_cut"], 0.0617)
        assert 0.6 < after["pass_chance"] < 0.95, (mean, r["recommended_cut"], after["pass_chance"])
        assert abs(after["recommended_cut"]) < target.UNSPENT_STEPS


def test_the_verdict_is_deterministic():
    a = target.verdict(runs(3.2779), 2380, None)
    b = target.verdict(runs(3.2779), 2380, None)
    assert a == b


def test_chance_falls_as_steps_are_cut():
    pr = target.Predictor(runs(3.2770))
    cs = [pr.chance(c) for c in (-40, 0, 40, 80)]
    assert cs == sorted(cs, reverse=True)


def test_without_num_iterations_the_advice_is_relative():
    r = target.verdict(runs(3.2790), None, None)
    assert "recommended_num_iterations" not in r and "num_iterations +" in r["critique"][0]


# ---------------------------------------------------------------- main() with a fake torchrun

FAKE_TORCHRUN = textwrap.dedent("""\
    #!{python}
    import os, re, sys
    assert "HARNESS_RESULT_FILE" not in os.environ
    gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])
    assert any(a == f"--master_port={{29600 + gpu}}" for a in sys.argv), sys.argv
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].endswith(f"inductor{{gpu}}")
    src = open("train_gpt.py").read()
    knob = lambda k: re.search(r"^ *" + k + r"(?: *: *int)? = (.+?)(?:  #.*)?$", src, re.M).group(1)
    losses, crash = eval(knob("LOSSES")), eval(knob("CRASH_GPU"))
    every, total = int(knob("val_loss_every")), int(knob("num_iterations"))
    assert every == 0
    open(f"gpu{{gpu}}.ran", "w").write("1")
    t = 0
    for step in range(1, total + 1):
        t += 440
        if gpu == crash and step == 100:
            print("[rank0]: Traceback (most recent call last):\\n[rank0]:   File \\"train_gpt.py\\", line 9, in <module>\\n"
                  "[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered", flush=True)
            sys.exit(1)
        if step % 200 == 0 or step == total:
            print(f"step:{{step}}/{{total}} train_time:{{t}}ms step_avg:440.00ms", flush=True)
    print(f"step:{{total}}/{{total}} val_loss:{{losses[gpu]}} train_time:{{t}}ms step_avg:440.00ms", flush=True)
    """)

CODE = textwrap.dedent("""\
    LOSSES = {losses}
    CRASH_GPU = {crash}
    class Hyperparameters:
        num_iterations: int = {iters}  # number of iterations to run
        val_loss_every: int = 250  # every how many steps to evaluate val loss? 0 for only at the end
    """)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    tr = bin_ / "torchrun"
    tr.write_text(FAKE_TORCHRUN.format(python=sys.executable))
    tr.chmod(tr.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(bench, "fetch", lambda name, dest: (dest / name).write_bytes(b"") or dest / name)
    monkeypatch.setattr(bench, "cuda_home", lambda root: str(root))
    monkeypatch.setattr(bench, "prefetch_kernels", lambda code: None)
    monkeypatch.delenv("HARNESS_EVAL_TIMEOUT_S", raising=False)
    out = tmp_path / "result.json"
    monkeypatch.setattr(target, "RESULT_FILE", str(out))

    def go(losses, crash=-1, iters=2380, gpus=8) -> dict | None:
        monkeypatch.setattr(bench, "gpu_info", lambda: "\n".join(["NVIDIA H100 80GB HBM3, 580.159.04"] * gpus))
        subj = tmp_path / "subjects" / "modded-nanogpt"
        subj.mkdir(parents=True, exist_ok=True)
        cand = CODE.format(losses=list(losses), crash=crash, iters=iters)
        (subj / "train_gpt.py").write_text(cand)
        monkeypatch.setenv("HARNESS_SUBJECTS_DIR", str(tmp_path / "subjects"))
        monkeypatch.setenv("HARNESS_CHANGES_FILE", str(tmp_path / "none.diff"))
        target.main()
        assert (subj / "train_gpt.py").read_text() == cand  # only the temporary copies are patched
        return json.loads(out.read_text()) if out.exists() else None

    return go


def test_main_end_to_end(setup, capsys):
    r = setup(runs(3.2790))
    schema_ok(r)
    assert r["runs"] == 8 and r["num_iterations"] == 2380 and r["problems"] == 1
    assert r["val_loss"] == pytest.approx(3.2790 + 0.0, abs=1e-4)
    assert r["recommended_num_iterations"] > 2380
    stages = [line[7:] for line in capsys.readouterr().out.splitlines() if line.startswith("stage: ")]
    assert stages == target.STAGES


def test_a_crash_in_one_run_is_the_candidates_failure(setup):
    r = setup(runs(3.2770), crash=3)
    schema_ok(r)
    assert r["problems"] == 1 and "illegal memory access" in r["critique"][0] and "(1 of 8 runs)" in r["critique"][0]


def test_too_few_gpus_is_not_the_candidates_failure(setup):
    with pytest.raises(probe.InfraError, match="at least 6 H100s"):
        setup(runs(3.2770), gpus=4)


def test_one_far_run_is_set_aside_but_two_are_kept():
    # backout-target-5's runs: one at 3.2900 among seven near 3.278 asked for +67 steps with it in
    five = [3.2822, 3.2798, 3.2756, 3.2900, 3.2769, 3.2777, 3.2798, 3.2776]
    r = target.verdict(five, 2370, 0.0617)
    assert r["runs"] == 8 and "3.2900" in r["run_losses"] and "left out" in r["caveats"]
    assert r["val_loss_sd"] < 0.002 and abs(r["recommended_cut"]) < 30  # not +67
    two = five[:3] + [3.2900, 3.2905] + five[5:]
    r2 = target.verdict(two, 2370, 0.0617)
    assert "unstable" in r2["caveats"] and r2["val_loss_sd"] > 0.002
