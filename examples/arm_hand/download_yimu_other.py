"""Fetch the pinned Xhand coke example and only its referenced meshes."""

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
REVISION = "8b0cdf2b24cc86d1c859050c5b0317ddbec59a63"
RELATIVE = Path("processed/fair_mon/xhand/right/coke")
OUTPUT = ROOT / "example_datasets/yimu_other_source"


def main():
    """Download into a separate folder without changing the example checkout."""

    def fetch(path):
        return Path(
            hf_hub_download(
                "retarget/retarget_example",
                str(path),
                repo_type="dataset",
                revision=REVISION,
                local_dir=OUTPUT,
            )
        )

    scene = fetch(RELATIVE / "scene.xml")
    tree = ET.parse(scene)
    meshroot = (scene.parent / tree.find("compiler").get("meshdir")).resolve()
    paths = [RELATIVE / "0/trajectory_mjwp.npz", RELATIVE / "task_info.json"]
    paths += [
        (meshroot / mesh.get("file")).relative_to(OUTPUT)
        for mesh in tree.findall("asset/mesh")
    ]
    with ThreadPoolExecutor(4) as executor:
        for path in executor.map(fetch, paths):
            print(path.relative_to(OUTPUT), flush=True)


if __name__ == "__main__":
    main()
