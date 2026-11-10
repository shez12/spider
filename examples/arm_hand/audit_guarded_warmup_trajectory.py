# Numerical-replay rejection wrapper; frozen source SHA256 ad0f3deb8adae22106649bdf33da65c1b841d494bed37b655c3503cd1191778a
"""Audit one saved GPU trajectory under explicit warmup grasp acceptance."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from examples.arm_hand.warmup_validation_guard import audit_for_optimizer

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-dir',required=True,type=Path);p.add_argument('--gpu-dir',required=True,type=Path);a=p.parse_args()
 model=mujoco.MjModel.from_xml_path(str(a.case_dir/'scene.xml'))
 with np.load(a.case_dir/'reference.npz') as ref,np.load(a.gpu_dir/'trajectory.npz') as tr:report=audit_for_optimizer(model,ref,tr)
 (a.gpu_dir/'tracking_collision_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
