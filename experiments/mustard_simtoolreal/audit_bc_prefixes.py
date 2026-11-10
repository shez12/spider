from pathlib import Path
import numpy as np,json
root=Path('/home/szc/work/spider');p=root/'outputs/mustard_classified_20261102_v2'
d=np.load(p/'teacher/native_bc_dataset.npz');x=d['obs'];ids=d['candidate_ids'];lengths=d['valid_lengths'];chosen=set(np.load(p/'selected/native_bc_dataset.npz')['candidate_ids'].tolist());rows=[]
for i,c in enumerate(ids):
 ob=x[i,:min(int(lengths[i]),961)];rel=ob[:,113:125].reshape(-1,4,3).mean(1);pos=rel+ob[:,87:90];tips=ob[:,98:113].reshape(-1,5,3)
 dist=np.sort(np.linalg.norm(tips-rel[:,None],axis=-1),axis=1)[:,:3].mean(1);speed=np.r_[0,np.linalg.norm(np.diff(pos,axis=0),axis=1)*60]
 held=(pos[:,2]-pos[0,2]>=.05)&(np.linalg.norm(rel,axis=1)<.18)&(dist<.1)&(speed<2.)
 run=np.convolve(held.astype(int),np.ones(30,dtype=int),mode='valid');found=np.flatnonzero(run==30);first=int(found[0]+29) if len(found) else None
 near=np.linalg.norm(ob[:,125:137].reshape(-1,4,3),axis=-1).max(1)<=.1125;hit=np.flatnonzero(near.cumsum()>=10);reach=int(hit[0]) if len(hit) else None
 z=1-2*(ob[0,94]**2+ob[0,95]**2);s=0 if z>.5 else 2 if z<-.5 else 1
 rows.append(dict(candidate_id=int(c),selected=int(c) in chosen,valid_length=int(lengths[i]),stratum=s,first_hold_frame=first,first_goal_frame=reach,grasp_before_goal=first is not None and reach is not None and first<=reach))
report={'rows':rows}
for name,test in [('excluded_grasp_by16',lambda r:not r['selected'] and r['first_hold_frame'] is not None),('excluded_grasp_and_reach_by16',lambda r:not r['selected'] and r['grasp_before_goal']),('all_grasp_by16',lambda r:r['first_hold_frame'] is not None)]:
 v=[r for r in rows if test(r)];report[name]={'count':len(v),'strata':{str(k):sum(r['stratum']==k for r in v) for k in range(3)}}
(root/'outputs/mustard_balance_ab_v1/audit/prefix_coverage.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
