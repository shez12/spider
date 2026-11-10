"""Verify SPIDER sampling updates stop when candidate count reaches zero.

Uses the actual regular and fast SPIDER update functions with real, deterministic
MuJoCo slider rollouts. This isolates sampling from a favorable precomputed grasp.
No production optimizer behavior is replaced or mocked.
"""

import argparse
import json
from pathlib import Path

import mujoco
import mujoco.rollout
import numpy as np
import torch

from spider.config import Config, get_noise_scale
from spider.optimizers.sampling import make_optimize_once_fn
from spider.optimizers.sampling_fast import make_optimize_once_fn_fast


def run(schedule, factory):
    """Run real sampling updates and record the first zero-sample failure."""
    torch.manual_seed(0)
    model = mujoco.MjModel.from_xml_string(
        """
    <mujoco><option timestep="0.01" integrator="implicitfast" gravity="0 0 0"/>
      <worldbody><body><joint name="slider" type="slide" axis="1 0 0" damping="1"/>
        <geom type="sphere" size="0.03" mass="0.1"/></body></worldbody>
      <actuator><position joint="slider" kp="100" kv="5"/></actuator>
    </mujoco>"""
    )
    data = mujoco.MjData(model)
    config = Config(
        device="cpu",
        embodiment_type="humanoid",
        nu=1,
        horizon=0.2,
        knot_dt=0.04,
        horizon_steps=20,
        knot_steps=4,
        use_torch_compile=False,
        first_ctrl_noise_scale=1,
        last_ctrl_noise_scale=1,
        joint_noise_scale=1,
        exploit_noise_scale=1,
        temperature=1,
        num_trace_uniform_samples=0,
        num_trace_topk_samples=0,
    )
    controls = torch.zeros((20, 1), dtype=torch.float64)
    initial = np.zeros(3)  # FULLPHYSICS: time, position, velocity
    rollout_batches = []

    def cost(commands):
        states, _ = mujoco.rollout.rollout(model, data, initial, commands)
        return ((states[:, :, 1] - 1) ** 2).mean(axis=1), states

    def rollout(cfg, env, candidates, ref_slice, env_param, **kwargs):
        del env, ref_slice, env_param
        rollout_batches.append(len(candidates))
        # Empty candidates contain no simulator evaluations. Let the repository
        # itself decide whether a sampling update can consume empty rewards.
        if len(candidates):
            costs, states = cost(candidates.numpy())
        else:
            costs, states = np.empty(0), np.empty((0, 20, 3))
        info = {}
        if kwargs.get("record_states"):
            info.update(
                recorded_qpos=torch.from_numpy(states[:, :, 1:2]),
                recorded_qvel=torch.from_numpy(states[:, :, 2:3]),
                state_snapshot={},
            )
        return (
            candidates,
            torch.from_numpy(-costs),
            torch.zeros(cfg.num_samples, dtype=torch.bool),
            info,
        )

    update = factory(rollout)
    rows = []
    baseline = float(cost(controls.numpy()[None])[0][0])
    for iteration, samples in enumerate(schedule):
        config.num_samples = samples
        # Rebuild the candidate dimension on every iteration, including zero.
        # get_noise_scale itself rejects zero, so slice the correctly configured
        # previous noise tensor to also exercise the inner update's zero case.
        config.noise_scale = (
            get_noise_scale(config) if samples else config.noise_scale[:0]
        )
        row = {"iteration": iteration, "samples": samples}
        previous = controls.clone()
        try:
            controls, _, _ = update(
                config,
                None,
                controls,
                (),
                sample_params={"global_noise_scale": 0.8**iteration},
            )
            if not torch.isfinite(controls).all():
                raise FloatingPointError("Sampling update returned non-finite controls")
            row.update(
                status="updated",
                cost=float(cost(controls.numpy()[None])[0][0]),
                control_change=float(torch.linalg.vector_norm(controls - previous)),
            )
        except (RuntimeError, IndexError, ValueError, FloatingPointError) as error:
            row.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                retained_previous_controls=bool(torch.equal(controls, previous)),
            )
            rows.append(row)
            break
        rows.append(row)
    return {
        "initial_cost": baseline,
        "iterations": rows,
        "rollout_batch_sizes": rollout_batches,
    }


def main():
    """Compare fixed samples against a schedule ending in zero candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/sampling_ablation")
    )
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    if args.samples < 32:
        parser.error("--samples must be >= 32 for this positive-control test")
    schedule = [
        args.samples,
        max(32, args.samples // 4),
        max(32, args.samples // 16),
        0,
    ]
    report = {}
    for name, factory in [
        ("sampling", make_optimize_once_fn),
        ("sampling_fast", make_optimize_once_fn_fast),
    ]:
        positive = run([args.samples] * len(schedule), factory)
        decreasing = run(schedule, factory)
        assert all(row["status"] == "updated" for row in positive["iterations"])
        assert positive["iterations"][-1]["cost"] < positive["initial_cost"]
        assert decreasing["iterations"][-1]["samples"] == 0
        assert decreasing["iterations"][-1]["status"] == "failed"
        assert decreasing["rollout_batch_sizes"][-1] == 0
        report[name] = {"fixed_samples": positive, "decreasing_samples": decreasing}
    try:
        get_noise_scale(Config(device="cpu", num_samples=0, nu=1))
    except IndexError as error:
        report["zero_sample_configuration"] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    else:
        raise AssertionError("Expected zero-sample configuration to fail")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ["sampling", "sampling_fast"]:
        fixed = report[name]["fixed_samples"]
        zero = report[name]["decreasing_samples"]["iterations"][-1]
        print(
            f"{name}: fixed {args.samples} samples cost {fixed['initial_cost']:.6f} -> {fixed['iterations'][-1]['cost']:.6f}"
        )
        print(
            f"{name}: schedule {schedule} fails at zero: {zero['error_type']}: {zero['error']}"
        )
    print(
        f"PASS: zero candidates cannot perform a sampling update. Report: {args.output_dir / 'report.json'}"
    )


if __name__ == "__main__":
    main()
