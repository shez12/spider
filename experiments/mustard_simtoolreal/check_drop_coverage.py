"""Audit native BC recovery sequences and nearest observed drop states."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 d=np.load(a.dataset);bank=np.load(a.bank);x=d['obs'];lengths=d['valid_lengths'];ids=d['candidate_ids']
 mask=np.arange(x.shape[1])[None,:]<lengths[:,None];ti,fi=np.nonzero(mask);flat=x[mask]
 hand=flat[:,98:113].reshape(-1,5,3);objrel=flat[:,113:125].reshape(-1,4,3).mean(1);xyz=objrel+flat[:,87:90]
 anatomy=[];approach_mask=np.zeros(mask.shape,dtype=bool)
 for i,length in enumerate(lengths):
  obs=x[i,:length];rel=obs[:,113:125].reshape(-1,4,3).mean(1);pos=rel+obs[:,87:90];dist=np.sort(np.linalg.norm(obs[:,98:113].reshape(-1,5,3)-rel[:,None],axis=-1),axis=1)[:,:3].mean(1)
  speed=np.r_[0,np.linalg.norm(np.diff(pos,axis=0),axis=1)*60]
  lift=pos[:,2]-pos[0,2];held=(lift>=.05)&(np.linalg.norm(rel,axis=1)<.18)&(dist<.1)&(speed<2.)
  detached=(np.linalg.norm(rel,axis=1)>.18)|(dist>.1)
  ontable=detached&(lift<=.04)&(speed<.15)
  state=0;run=0;stable=0;first=None;drops=[];recoveries=[]
  for t in range(length):
   run=run+1 if held[t] else 0
   if state==0 and run>=30:state=1;first=t
   elif state==1:
    stable=stable+1 if ontable[t] else 0
    if stable>=15:state=2;drops.append(t);run=0
   elif state==2 and run>=30:recoveries.append(t);state=1;stable=0
  end=first+1 if first is not None else length
  approach_mask[i,:end]=(lift[:end]<=.04)
  anatomy.append(dict(candidate_id=int(ids[i]),first_grasp_frame=first,drop_frames=drops,recovery_frames=recoveries))
 spec=json.loads((ROOT/'assets/robot/robot_spec.json').read_text());joints={j.attrib['name']:j for j in ET.parse(ROOT/'assets/robot/kuka_sharpa_right.urdf').getroot().findall('joint')}
 lo=np.array([float(joints[name].find('limit').attrib['lower']) for name in spec['joint_names']]);hi=np.array([float(joints[name].find('limit').attrib['upper']) for name in spec['joint_names']]);q=lo+.5*(flat[:,:29]+1)*(hi-lo)
 grasp=approach_mask[mask];queries=bank['obs'];rows=[]
 for i,ob in enumerate(queries):
  rel=ob[113:125].reshape(4,3).mean(0);obj=rel+ob[87:90];jq=lo+.5*(ob[:29]+1)*(hi-lo)
  dot=np.clip(np.abs(flat[:,94:98]@ob[94:98]),0,1);angle=2*np.arccos(dot)*180/np.pi
  metrics=np.stack([np.linalg.norm(xyz-obj,axis=1),angle,np.sqrt(((q[:,:7]-jq[:7])**2).mean(1)),np.sqrt(((q[:,7:]-jq[7:])**2).mean(1)),np.linalg.norm(objrel-rel,axis=1),np.sqrt(((flat[:,58:87]-ob[58:87])**2).mean(1)),np.sqrt(((flat[:,125:137]-ob[125:137])**2).mean(1))],axis=1)
  thresholds=np.array([.03,15.,.1,.15,.03,.15,.05]);norm=metrics/thresholds
  physical_score=(norm[:,:6]**2).mean(1);full_score=(norm**2).mean(1)
  match={}
  for label,score in [('all_physical',physical_score),('grasp_physical',np.where(grasp,physical_score,np.inf)),('all_with_goal',full_score)]:
   k=int(score.argmin());match[label]=dict(candidate_id=int(ids[ti[k]]),frame=int(fi[k]),object_center_distance_m=float(metrics[k,0]),object_rotation_deg=float(metrics[k,1]),arm_joint_rmse_rad=float(metrics[k,2]),hand_joint_rmse_rad=float(metrics[k,3]),palm_relative_object_distance_m=float(metrics[k,4]),previous_target_rmse_rad=float(metrics[k,5]),goal_delta_rmse_m=float(metrics[k,6]),normalized_score=float(score[k]))
  rows.append(dict(candidate_id=int(bank['candidate_id'][i]),compatible_physical_frames=int((norm[:,:6].max(1)<=1).sum()),compatible_grasp_frames=int(((norm[:,:6].max(1)<=1)&grasp).sum()),compatible_including_goal_frames=int((norm.max(1)<=1).sum()),nearest=match))
 report=dict(trajectories=len(ids),valid_frames=len(flat),approach_frames=int(grasp.sum()),demonstrations_with_detected_first_grasp=sum(v['first_grasp_frame'] is not None for v in anatomy),demonstrations_with_postgrasp_drop=sum(bool(v['drop_frames']) for v in anatomy),demonstrations_with_successful_regrasp=sum(bool(v['recovery_frames']) for v in anatomy),drop_queries=len(rows),queries_with_compatible_physical_frame=sum(v['compatible_physical_frames']>0 for v in rows),queries_with_compatible_grasp_frame=sum(v['compatible_grasp_frames']>0 for v in rows),queries_with_compatible_full_frame=sum(v['compatible_including_goal_frames']>0 for v in rows),thresholds=dict(object_center_m=.03,object_rotation_deg=15,arm_joint_rmse_rad=.1,hand_joint_rmse_rad=.15,palm_relative_object_m=.03,previous_target_rmse_rad=.15,goal_delta_rmse_m=.05),limitations='Compatibility thresholds are diagnostic heuristics, not proof of out-of-distribution. BC cache lacks contact forces and object velocities; speed estimated by finite differences. No padding included.',sequence_scan=anatomy,drop_matches=rows)
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['sequence_scan','drop_matches']}),flush=True)
if __name__=='__main__':main()
