"""Uniformly scaled copy of an object asset directory (URDF + meshes + calibration.json), for object-size ablations.

Meshes are scaled about the link origin; mass scales with s^3 (same density), inertia with s^5; calibration.json
bbox, policy scale and keypoints scale with s. Usage:
  python make_scaled_object.py --scale 0.6 --output assets/object_mustard_s060
"""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,trimesh
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=HERE/'assets/object');p.add_argument('--scale',type=float,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=a.scale
    urdfs=sorted(a.source.glob('*.urdf'));assert len(urdfs)==1;a.output.mkdir(parents=True)
    tree=ET.parse(urdfs[0]);root=tree.getroot();root.set('name',f"{root.get('name')}_s{round(s*100):03d}")
    inertial=root.find('.//inertial');o=inertial.find('origin')
    o.set('xyz',' '.join(str(float(x)*s) for x in o.get('xyz').split()))
    m=inertial.find('mass');m.set('value',str(float(m.get('value'))*s**3))
    i=inertial.find('inertia')
    for k in list(i.attrib):i.set(k,str(float(i.get(k))*s**5))
    for mesh in root.iter('mesh'):
        assert mesh.get('scale') is None
        t=trimesh.load(a.source/mesh.get('filename'),force='mesh',process=False);t.vertices*=s;t.export(a.output/mesh.get('filename'))
    for origin in root.iter('origin'):
        if origin is not o:origin.set('xyz',' '.join(str(float(x)*s) for x in origin.get('xyz','0 0 0').split()))
    tree.write(a.output/f"{urdfs[0].stem}_s{round(s*100):03d}.urdf",encoding='unicode')
    c=json.loads((a.source/'calibration.json').read_text())
    for k in ('bbox_dims_m','source_to_new_root_center_in_source_m','policy_object_scale','keypoints_local_m'):c[k]=(np.array(c[k])*s).tolist()
    c['mass_kg']*=s**3;c['inertia_eigenvalues']=(np.array(c['inertia_eigenvalues'])*s**5).tolist()
    c.update(uniform_scale=s,scaled_from=str(a.source))
    (a.output/'calibration.json').write_text(json.dumps(c,indent=2)+'\n')
    vis=trimesh.load(a.output/root.find('.//visual//mesh').get('filename'),force='mesh',process=False)
    print(json.dumps(dict(output=str(a.output),bbox_dims_m=c['bbox_dims_m'],visual_extents_m=vis.extents.tolist(),mass_kg=c['mass_kg'],policy_object_scale=c['policy_object_scale'])))


if __name__=='__main__':main()
