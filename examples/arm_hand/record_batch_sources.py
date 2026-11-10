"""Render unmodified published dataset states beside each batch result."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, default=ROOT/'outputs/kuka_sharpa_batch20')
    parser.add_argument('--dataset-dir', type=Path, default=ROOT/'example_datasets/retarget_full')
    args = parser.parse_args()
    rows = json.loads((args.batch_dir/'selected.json').read_text())
    results = []
    for index, row in enumerate(rows, 1):
        folder = args.batch_dir/f'{index:02d}_{row["task"]}'
        scene = (args.dataset_dir/row['scene']).resolve()
        trajectory = (args.dataset_dir/row['path']/'trajectory_mjwp.npz').resolve()
        hashes = {'scene.xml': sha256(scene), 'trajectory_mjwp.npz': sha256(trajectory)}
        for key, value in hashes.items():
            expected = row.get('source_sha256', {}).get(key)
            if expected and expected != value:
                raise RuntimeError(f'Source checksum mismatch: {row["task"]} {key}')
        output = folder/'source_trajectory.mp4'
        metadata = folder/'source_visualization.json'
        if output.exists() and metadata.exists():
            old = json.loads(metadata.read_text())
            if old.get('source_sha256') == hashes:
                results.append(old)
                continue
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        with np.load(trajectory) as archive:
            poses = archive['qpos'].reshape(-1, model.nq)
        if not np.isfinite(poses).all():
            raise RuntimeError(f'Nonfinite source states: {trajectory}')
        dt = float(row['sim_dt'])
        camera = mujoco.MjvCamera()
        camera.lookat[:] = np.median(poses[:, -7:-4], axis=0) + [0, 0, 0.05]
        camera.distance = max(0.85, float(np.ptp(poses[:, -7:-4], axis=0).max())*2.5)
        camera.azimuth = 140
        camera.elevation = -25
        with mujoco.Renderer(model, 720, 1280) as renderer, imageio.get_writer(
            str(output), fps=1/dt, codec='libx264', quality=8
        ) as writer:
            for frame_id, pose in enumerate(poses):
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frame = renderer.render().copy()
                cv2.rectangle(frame, (0,0), (1280,65), (255,255,255), -1)
                cv2.putText(frame, f'SOURCE DATASET | {row["task"]}', (15,26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                cv2.putText(frame, f'Published Allegro right-hand states | t={frame_id*dt:.2f}s | no mirroring or extra hold',
                            (15,53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                writer.append_data(frame)
        capture = cv2.VideoCapture(str(output))
        encoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if encoded_frames != len(poses):
            raise RuntimeError(f'Encoded frame count mismatch: {output}')
        info = dict(task=row['task'], video=str(output.resolve()),
                    source_scene=str(scene), source_trajectory=str(trajectory),
                    source_sha256=hashes, frames=len(poses), fps=1/dt,
                    source_duration_s=(len(poses)-1)*dt,
                    visualization='original published robot states; no reflection, translation, scaling or extra hold',
                    note='retarget_full published Allegro trajectory, not original DexYCB camera video or raw MANO annotations')
        metadata.write_text(json.dumps(info, indent=2)+'\n')
        results.append(info)
        print(f'{index}/{len(rows)} {output}', flush=True)
    (args.batch_dir/'source_visualizations.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'Completed {len(results)} source videos', flush=True)


if __name__ == '__main__':
    main()
