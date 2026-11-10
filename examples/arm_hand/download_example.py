"""Download one pinned published DexYCB/Allegro episode and its exact assets."""

import argparse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

REVISION = "1bb88b0c32b33eb868def58333a516ad2f2aba30"
TASK = "dexycb_01_right_002_master_chef_can_20200709_141754_00"


def main():
    """Fetch only this episode; the full 10 GB archive is unnecessary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "example_datasets/arm_hand_source",
    )
    args = parser.parse_args()
    root = args.output_dir.resolve()
    trial = Path("processed/dexycb_lifted/allegro/right") / TASK

    def fetch(relative):
        return Path(
            hf_hub_download(
                "retarget/retarget_full",
                str(relative),
                repo_type="dataset",
                revision=REVISION,
                local_dir=root,
            )
        )

    scene = fetch(trial / "scene.xml")
    assets = (scene.parent / "../../../assets").resolve()
    paths = [
        trial / "task_info.json",
        trial / "0/config.yaml",
        trial / "0/metrics.json",
        trial / "0/trajectory_mjwp.npz",
        trial / "0/trajectory_kinematic.npz",
    ]
    paths += [
        (assets / mesh.attrib["file"]).relative_to(root)
        for mesh in ET.parse(scene).findall(".//asset/mesh")
    ]
    with ThreadPoolExecutor(4) as executor:
        for path in executor.map(fetch, paths):
            print(path.relative_to(root), flush=True)
    print(f"Downloaded dataset episode at revision {REVISION} into {root}")


if __name__ == "__main__":
    main()
