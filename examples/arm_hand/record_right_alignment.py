"""Compare unmodified published source states with right-hand geometric IK."""

import os
import argparse
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'outputs/kuka_sharpa_right_foam')
    parser.add_argument('--source-scene',type=Path,default=ROOT/'example_datasets/retarget_full/processed/dexycb_lifted/allegro/right/dexycb_01_right_061_foam_brick_20200709_153241_00/scene.xml')
    parser.add_argument('--source-trajectory',type=Path)
    args=parser.parse_args();out=args.input_dir;src=args.source_scene.parent
    trajectory=args.source_trajectory or src/'0/trajectory_mjwp.npz'
    models=[mujoco.MjModel.from_xml_path(str(src/'scene.xml')),mujoco.MjModel.from_xml_path(str(out/'scene.xml'))]
    datas=[mujoco.MjData(m) for m in models]
    with np.load(trajectory) as a, np.load(out/'reference.npz') as b:
        source=a['qpos'].reshape(-1,models[0].nq)
        target=np.c_[b['joint_ik'],b['object_qpos']]
    camera=mujoco.MjvCamera();camera.lookat[:]=np.median(source[:,-7:-4],axis=0)+[0,0,.06]
    camera.distance=1.0;camera.azimuth=140;camera.elevation=-22
    with mujoco.Renderer(models[0],480,640) as left, mujoco.Renderer(models[1],480,640) as right, imageio.get_writer(str(out/'source_vs_right_ik.mp4'),fps=100,codec='libx264',quality=8) as writer:
        for i in range(len(source)):
            frames=[]
            for m,d,renderer,pose,label in zip(models,datas,[left,right],[source[i],target[i]],['SOURCE: Allegro right','TARGET: KUKA + Sharpa right (IK only)']):
                d.qpos[:]=pose;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera)
                frame=renderer.render().copy();cv2.rectangle(frame,(0,0),(640,55),(255,255,255),-1)
                cv2.putText(frame,label,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
                cv2.putText(frame,f't={i*.01:.2f}s | same world frame, no mirroring',(10,44),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
                frames.append(frame)
            joined=np.concatenate(frames,axis=1);writer.append_data(joined)
            if i==0:cv2.imwrite(str(out/'initial_alignment.png'),joined)
    print(out/'source_vs_right_ik.mp4')


if __name__=='__main__':main()
