"""Repeat the frozen MANO/Sharpa experiment with a different sampling budget."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from examples.arm_hand.run_sharpa_complex import ROOT, validate
from examples.arm_hand.run_sharpa_more import hashes
from examples.arm_hand.try_sharpa import optimize


def main():
    """Reuse exactly the saved scene, references and initial state for comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write the rerun to this directory instead of the legacy output path.",
    )
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    if args.samples < 10:
        parser.error("Use at least 10 samples for the elite weighting")
    source = ROOT / "outputs/mano_to_sharpa"
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT / f"outputs/mano_to_sharpa_samples_{args.samples}"
    )
    if args.record_only:
        from examples.arm_hand import record_mano_sharpa

        record_mano_sharpa.OUT = output
        record_mano_sharpa.main()
        return
    output.mkdir(exist_ok=True)
    before = hashes()
    for name in [
        "scene.xml",
        "reference.npz",
        "human_reference.npz",
        "source_reference.npz",
        "conversion.json",
        "retargeting.json",
    ]:
        shutil.copy2(source / name, output / name)
    dataset = ROOT / "example_datasets/retarget_full"
    snapshot_dataset = ROOT / "reproduction/snapshot-retarget_full"
    if not dataset.exists() and snapshot_dataset.exists():
        scene = output / "scene.xml"
        scene.write_text(
            scene.read_text().replace(str(dataset), str(snapshot_dataset))
        )
    shutil.copytree(
        source / "source_assets", output / "source_assets", dirs_exist_ok=True
    )
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    with np.load(output / "reference.npz") as archive:
        ref = {key: archive[key] for key in archive.files}
    optimize(
        SimpleNamespace(
            output_dir=output, samples=args.samples, iterations=20, threads=8
        ),
        model,
        ref,
    )
    original = json.loads((source / "report.json").read_text())
    report = validate(output, model, original["case"])
    assert hashes() == before
    for key in [
        "cost",
        "object_lift_m",
        "final_object_error_m",
        "object_rmse_m",
        "wrist_rmse_m",
    ]:
        np.testing.assert_allclose(
            report["baseline"][key], original["baseline"][key], atol=1e-12, rtol=1e-12
        )
    report.update(
        method_files_unchanged=True,
        method_sha256=before,
        conversion=original["conversion"],
        comparison_baseline_dir=str(source),
        frozen_reference_reused=True,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if (source / "optimization_parameters.json").exists():
        parameters = json.loads((source / "optimization_parameters.json").read_text())
        parameters["sampling"].update(
            samples=args.samples, elite_count=max(1, int(0.1 * args.samples))
        )
        (output / "optimization_parameters.json").write_text(
            json.dumps(parameters, indent=2) + "\n"
        )
    comparison = []
    for r in [original, report]:
        comparison.append(
            dict(
                samples=r["samples"],
                **r["optimized"],
                final_orientation_error_deg=r["final_object_orientation_error_deg"],
                retention_pass=r["retention_pass"],
                task_success=r["task_success"],
                hold_relative_displacement_m=r["hold"][
                    "object_relative_palm_displacement_m"
                ],
                speed_within_supplied_ratings=r["speed_within_supplied_ratings"],
            )
        )
    (output / "comparison_512.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
