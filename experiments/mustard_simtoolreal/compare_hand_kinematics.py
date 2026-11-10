"""Forward-kinematics comparison of two right Sharpa hand URDFs from the palm (right_hand_C_MC) root.

Same joint values -> per-link world position/rotation differences. Used to decide whether demos and reset-bank
states (joint space) transfer from our MuJoCo-exported hand to the official Sharpa Wave URDF.
"""
import argparse,json,sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def chain(path):
    root=ET.parse(path).getroot();js={}
    for j in root.findall('joint'):
        o=j.find('origin');xyz=np.array([float(v) for v in (o.get('xyz') if o is not None else '0 0 0').split()])
        rpy=np.array([float(v) for v in (o.get('rpy') if o is not None else '0 0 0').split()])
        ax=j.find('axis');axis=np.array([float(v) for v in ax.get('xyz').split()]) if ax is not None else np.zeros(3)
        js[j.find('child').get('link')]=dict(name=j.get('name'),parent=j.find('parent').get('link'),xyz=xyz,R=Rot.from_euler('xyz',rpy).as_matrix(),axis=axis,type=j.get('type'))
    return js


def fk(js,q,root='right_hand_C_MC'):
    out={root:(np.zeros(3),np.eye(3))}
    def get(link):
        if link in out:return out[link]
        j=js[link];p,R=get(j['parent'])
        Rj=R@j['R'];pj=p+R@j['xyz']
        if j['type']=='revolute':Rj=Rj@Rot.from_rotvec(j['axis']*q.get(j['name'],0.)).as_matrix()
        out[link]=(pj,Rj);return out[link]
    for link in js:
        try:get(link)
        except KeyError:pass
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('a',type=Path);p.add_argument('b',type=Path);p.add_argument('--samples',type=int,default=200)
    a=p.parse_args();A=chain(a.a);B=chain(a.b)
    joints={j['name']:j for j in A.values() if j['type']=='revolute' and j['name'].startswith('right_')}
    limits={}
    for path in (a.a,):
        for j in ET.parse(path).getroot().findall('joint'):
            l=j.find('limit')
            if l is not None and j.get('name') in joints:limits[j.get('name')]=(float(l.get('lower')),float(l.get('upper')))
    rng=np.random.default_rng(0);worst={};rot={}
    for s in range(a.samples):
        q={n:(0. if s==0 else rng.uniform(*limits[n])) for n in joints}
        fa,fb=fk(A,q),fk(B,q)
        for link in set(fa)&set(fb):
            if not link.startswith('right_'):continue
            worst[link]=max(worst.get(link,0.),float(np.linalg.norm(fa[link][0]-fb[link][0])))
            rot[link]=max(rot.get(link,0.),float(np.degrees(Rot.from_matrix(fa[link][1].T@fb[link][1]).magnitude())))
    out=dict(a=str(a.a),b=str(a.b),max_position_diff_mm={k:round(v*1e3,3) for k,v in sorted(worst.items(),key=lambda x:-x[1])},
        max_rotation_diff_deg={k:round(v,2) for k,v in sorted(rot.items(),key=lambda x:-x[1])})
    print(json.dumps(out,indent=1))


if __name__=='__main__':main()
