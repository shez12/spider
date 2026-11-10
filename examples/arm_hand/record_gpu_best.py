"""Render saved GPU physics states without substituting CPU dynamics."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('outputs/kuka_sharpa_gpu'))
    parser.add_argument('--scene', type=Path, default=Path('outputs/kuka_sharpa_current_environment/scene.xml'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.loads((args.input_dir/'result.json').read_text())
    output = args.output or args.input_dir/'best_50_gpu.mp4'
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0.15, -0.08]
    camera.distance = 1.9
    camera.azimuth = 145
    camera.elevation = -18
    with np.load(args.input_dir/'trajectory.npz') as trajectory:
        dt = float(trajectory['control_dt'])
        hand_only = model.nu == 22 and model.nmocap == 1
        if hand_only:
            camera.lookat[:] = np.median(trajectory['qpos'][:,-7:-4],axis=0)+[0,0,.05]
            camera.distance = .95
        title = 'SHARPA RIGHT | PRESCRIBED WRIST' if hand_only else 'KUKA + SHARPA'
        phase = 'ideal wrist support; saved GPU physics states' if hand_only else 'saved GPU physics states'
        with mujoco.Renderer(model, 720, 1280) as renderer, imageio.get_writer(str(output), fps=round(1/dt), codec='libx264', quality=8) as writer:
            for i, pose in enumerate(trajectory['qpos']):
                data.qpos[:] = pose
                data.qvel[:] = trajectory['qvel'][i]
                if 'mocap_pos' in trajectory:
                    data.mocap_pos[:] = trajectory['mocap_pos'][i]
                    data.mocap_quat[:] = trajectory['mocap_quat'][i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frame = renderer.render().copy()
                cv2.rectangle(frame, (0,0), (1280,65), (255,255,255), -1)
                cv2.putText(frame, f"{title} | GPU best cost {result['best_cost']:.3f} | iteration {result['iterations']}", (15,26), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,0,0), 1)
                cv2.putText(frame, f"t={(i+1)*dt:.2f}s | {phase} | no extra hold", (15,53), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,0,0), 1)
                writer.append_data(frame)
    print(output.resolve())


if __name__ == '__main__':
    main()
