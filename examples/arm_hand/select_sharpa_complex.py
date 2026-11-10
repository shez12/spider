"""Rank selected full-release tool trajectories by measured motion complexity."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def main():
    """Download small candidate trajectories and rank rotation/path extent."""
    metadata = ROOT / "example_datasets/.retarget_full-download"
    revision = json.loads((metadata / "candidate.json").read_text())["data_revision"]
    runs = json.loads((metadata / "metadata/manifest.json").read_text())["runs"]
    local = ROOT / "example_datasets/retarget_full_candidates"
    chosen = []
    for term in ["037_scissors", "035_power_drill", "025_mug", "019_pitcher_base"]:
        rows = [
            r
            for r in runs
            if r["robot_type"] == "allegro"
            and r["dataset"] == "dexycb_lifted"
            and term in r["task"]
        ]
        chosen.extend(rows[:8])

    def assess(row):
        path = Path(
            hf_hub_download(
                "retarget/retarget_full",
                row["path"] + "/trajectory_mjwp.npz",
                repo_type="dataset",
                revision=revision,
                local_dir=local,
            )
        )
        with np.load(path) as f:
            q = f["qpos"].reshape(-1, f["qpos"].shape[-1])
        obj = q[:, -7:]
        rotations = Rotation.from_quat(obj[:, 3:][:, [1, 2, 3, 0]])
        angular = (rotations[:-1].inv() * rotations[1:]).magnitude()
        return {
            "dataset": row["dataset"],
            "task": row["task"],
            "source_hand": row["robot_type"],
            "scene": row["scene"],
            "path": row["path"],
            "frames": len(obj),
            "duration_s": (len(obj) - 1) * row["sim_dt"],
            "rotation_path_deg": float(np.rad2deg(angular.sum())),
            "net_rotation_deg": float(
                np.rad2deg((rotations[0].inv() * rotations[-1]).magnitude())
            ),
            "object_path_m": float(
                np.linalg.norm(np.diff(obj[:, :3], axis=0), axis=1).sum()
            ),
            "lift_m": float(obj[-1, 2] - obj[0, 2]),
        }

    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(assess, chosen))
    results.sort(key=lambda r: r["rotation_path_deg"], reverse=True)
    output = ROOT / "outputs/sharpa_complex"
    output.mkdir(exist_ok=True)
    (output / "candidates.json").write_text(json.dumps(results, indent=2) + "\n")
    for r in results:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
