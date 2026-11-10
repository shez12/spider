"""Static self-penetration audit of the right KUKA+Sharpa URDF (MuJoCo signed distance between collision geoms).

Pairs filtered in Isaac (parent/child, robot_spec.json exclusions, adjacent_links.py, optional extra filters) are
skipped. Poses: hand open (all hand joints at lower limit... i.e. the native 'open' -1 action), the default pose,
native resets (0.9*default + 0.1*uniform) and hand half closed. Reports the deepest penetration per link pair.
"""
import argparse,json,itertools,sys
from pathlib import Path
import numpy as np,mujoco
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--filters',nargs='*',default=[]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=200);p.add_argument('--side',choices=('left','right'),default='right');a=p.parse_args()
    sys.path.insert(0,str(HERE));import self_collision_filters
    if a.side=='right':
        urdf=HERE/'assets/robot/kuka_sharpa_right.urdf';spec=json.loads((HERE/'assets/robot/robot_spec.json').read_text())
    else:
        urdf=PROJECT/'third_party/simtoolreal/assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf';spec=dict(exclusions=[])
    import re
    xml=re.sub(r'(<robot[^>]*>)',r'\1<mujoco><compiler discardvisual="true" fusestatic="false" strippath="false"/></mujoco>',urdf.read_text(),count=1)
    tmp=urdf.parent/'_mj_audit.urdf';tmp.write_text(xml)
    try:m=mujoco.MjModel.from_xml_path(str(tmp))
    finally:tmp.unlink()
    d=mujoco.MjData(m)
    import importlib.util
    s=importlib.util.spec_from_file_location('adj',PROJECT/'third_party/simtoolreal/isaacgymenvs/tasks/simtoolreal/adjacent_links.py');adj=importlib.util.module_from_spec(s);s.loader.exec_module(adj)
    filtered=set()
    for x,y in spec['exclusions']+self_collision_filters.pairs(a.filters):filtered|={(x,y),(y,x)}
    for x,ys in (adj.RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS if a.side=='right' else adj.LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS).items():
        for y in ys:filtered|={(x,y),(y,x)}
    name=lambda b:m.body(b).name
    geoms=[g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    pairs=[]
    for g1,g2 in itertools.combinations(geoms,2):
        b1,b2=m.geom_bodyid[g1],m.geom_bodyid[g2]
        if b1==b2 or m.body_parentid[b1]==b2 or m.body_parentid[b2]==b1 or (name(b1),name(b2)) in filtered:continue
        pairs.append((g1,g2))
    jnt=[m.joint(j) for j in range(m.njnt)];lo=m.jnt_range[:,0];hi=m.jnt_range[:,1]
    hand=np.array([a.side+'_' in m.joint(j).name for j in range(m.njnt)])
    default=np.zeros(m.nq);default[:7]=[-1.571,1.571,0.,1.376,0.,1.485,1.308][:7]
    rng=np.random.default_rng(0);poses={'open':[],'default':[],'native':[],'half_closed':[]}
    for _ in range(a.samples):
        q=default.copy();q[hand]=lo[hand];poses['open'].append(q.copy())
        poses['default'].append(np.clip(default,lo,hi))
        poses['native'].append(np.clip(.9*default+.1*rng.uniform(lo,hi),lo,hi))
        q=default.copy();q[hand]=lo[hand]+.5*(hi[hand]-lo[hand]);poses['half_closed'].append(q)
    worst={}
    for label,qs in poses.items():
        for q in qs[:a.samples if label=='native' else 1]:
            d.qpos[:]=q;mujoco.mj_forward(m,d)
            for g1,g2 in pairs:
                dist=mujoco.mj_geomDistance(m,d,g1,g2,.01,None)
                if dist<-1e-3:
                    k=f'{name(m.geom_bodyid[g1])}|{name(m.geom_bodyid[g2])}';worst.setdefault(label,{});worst[label][k]=min(worst[label].get(k,0.),float(dist))
    out=dict(filters=a.filters,pairs_checked=len(pairs),penetrations_m={k:dict(sorted(v.items(),key=lambda x:x[1])) for k,v in worst.items()})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out,indent=1)[:3000])


if __name__=='__main__':main()
