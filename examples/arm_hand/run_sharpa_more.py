"""Ten additional frozen-source Sharpa experiments; existing methods imported unchanged."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from examples.arm_hand.run_sharpa_complex import ROOT, build, fit, reference, validate
from examples.arm_hand.test_active_distal import fingerprint
from examples.arm_hand.try_sharpa import optimize

OUT = ROOT / "outputs/sharpa_more"
DATA = ROOT / "example_datasets/retarget_full"


def hashes():
    """Fingerprint all imported method files."""
    result = fingerprint()
    for name in ["try_sharpa.py", "run_sharpa_complex.py"]:
        key = "examples/arm_hand/" + name
        result[key] = hashlib.sha256((ROOT / key).read_bytes()).hexdigest()
    return result


def select():
    """Freeze ten diverse source cases before running targets."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "selected.json"
    if path.exists():
        return json.loads(path.read_text())
    runs = json.loads((DATA / "manifest.json").read_text())["runs"]
    objects = [
        "011_banana",
        "024_bowl",
        "004_sugar_box",
        "006_mustard_bottle",
        "021_bleach_cleanser",
        "019_pitcher_base",
        "036_wood_block",
        "008_pudding_box",
        "009_gelatin_box",
        "052_extra_large_clamp",
    ]
    chosen = []
    for obj in objects:
        candidates = []
        for row in runs:
            if (
                row["dataset"] != "dexycb_lifted"
                or row["robot_type"] != "allegro"
                or obj not in row["task"]
            ):
                continue
            with np.load(DATA / row["path"] / "trajectory_mjwp.npz") as f:
                q = f["qpos"].reshape(-1, f["qpos"].shape[-1])[:, -7:]
            rot = Rotation.from_quat(q[:, [4, 5, 6, 3]])
            angle = float(np.rad2deg((rot[0].inv() * rot[-1]).magnitude()))
            candidates.append(
                dict(
                    row,
                    label=obj[4:],
                    source_net_rotation_deg=angle,
                    source_lift_m=float(q[-1, 2] - q[0, 2]),
                    source_frames=len(q),
                )
            )
        chosen.append(max(candidates, key=lambda r: r["source_net_rotation_deg"]))
    path.write_text(json.dumps(chosen, indent=2) + "\n")
    return chosen


def run(row):
    """Execute and validate one case with unchanged settings."""
    output = OUT / row["label"]
    output.mkdir(exist_ok=True)
    if (output / "report.json").exists():
        return
    before = hashes()
    start = time.perf_counter()
    scene = DATA / row["scene"]
    model, source = build(scene, output)
    ref, frames = reference(
        source, DATA / row["path"] / "trajectory_mjwp.npz", output, 0.5
    )
    ref = fit(model, ref, output, 2)
    optimize(
        SimpleNamespace(output_dir=output, samples=512, iterations=20, threads=8),
        model,
        ref,
    )
    report = validate(output, model, row)
    assert before == hashes()
    report.update(
        original_frames=frames,
        source_scene=str(scene.resolve()),
        elapsed_s=time.perf_counter() - start,
        method_files_unchanged=True,
        method_sha256=before,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def launch(row):
    """Run a case in its own process and retain its log."""
    print("START " + row["label"], flush=True)
    with (OUT / (row["label"] + ".log")).open("w") as log:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.arm_hand.run_sharpa_more",
                "--case",
                row["label"],
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    print(f"FINISH {row['label']} exit={proc.returncode}", flush=True)
    return proc.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    args = parser.parse_args()
    rows = select()
    if args.case:
        run(next(r for r in rows if r["label"] == args.case))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(launch, rows))
        reports = [
            json.loads((OUT / r["label"] / "report.json").read_text())
            for r in rows
            if (OUT / r["label"] / "report.json").exists()
        ]
        (OUT / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
        if any(codes):
            raise SystemExit("Some cases failed: inspect logs")
