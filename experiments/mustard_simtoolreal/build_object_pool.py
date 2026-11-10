"""Build Isaac-loadable URDFs for one instance per object category of the retargeting datasets.

Each object is re-rooted at its visual-mesh AABB centre (same convention as the mustard asset), keeps the
dataset's convex decomposition as collision, and gets explicit mass/inertia from a uniform density (the datasets
carry no masses). Output: <output>/<name>/object.urdf + manifest.json with bbox, SimToolReal normalized scale
(bbox / object_base_size) and mass.
"""
import argparse,json,re
from pathlib import Path
import numpy as np,trimesh

DATA=Path('/home/szc/work/spider/example_datasets/retarget_full/processed')
CATEGORY={'dexycb_lifted':lambda s:re.sub(r'^dexycb_\d+_(right|left)_','',s).rsplit('_',3)[0],
    'hot3d_v2':lambda s:re.sub(r'^.*_(right|left)_','',s).rsplit('_',1)[0],
    'hrdexdb_24f':lambda s:s.rsplit('-',1)[0],'oakink_lifted':lambda s:s.split('_')[3]}


def fmt(x):return ' '.join(f'{float(v):.9g}' for v in x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--density',type=float,default=500.)
    p.add_argument('--mass-range',type=float,nargs=2,default=(0.03,0.6));p.add_argument('--object-base-size',type=float,default=0.04)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);manifest=[]
    for ds,key in CATEGORY.items():
        seen={}
        for d in sorted((DATA/ds/'assets/objects').iterdir()):
            if (d/'visual.obj').exists() and any((d/'convex').glob('*.obj')):seen.setdefault(key(d.name),d)
        for cat,d in sorted(seen.items()):
            name=f'{ds.split("_")[0]}__{cat}';out=a.output/name;out.mkdir()
            vis=trimesh.load(d/'visual.obj',force='mesh');center=vis.bounds.mean(0);bbox=vis.extents
            parts=[trimesh.load(f,force='mesh') for f in sorted((d/'convex').glob('*.obj'),key=lambda f:int(f.stem))]
            vol=float(sum(abs(m.volume) for m in parts));mass=float(np.clip(a.density*vol,*a.mass_range))
            com=sum(m.center_mass*abs(m.volume) for m in parts)/vol;inertia=np.zeros((3,3))
            for m in parts:
                m.density=1.;I=m.moment_inertia*abs(m.volume)/abs(m.mass);r=m.center_mass-com
                inertia+=I+abs(m.volume)*(np.dot(r,r)*np.eye(3)-np.outer(r,r))
            inertia*=mass/vol
            for i,m in enumerate(parts):m.apply_translation(-center);m.export(out/f'collision_{i}.obj')
            vis.apply_translation(-center);vis.export(out/'visual.obj')
            col=''.join(f'<collision name="c{i}"><geometry><mesh filename="collision_{i}.obj"/></geometry></collision>' for i in range(len(parts)))
            I=inertia
            (out/'object.urdf').write_text(f'''<robot name="{name}"><link name="object_root">
<inertial><origin xyz="{fmt(com-center)}" rpy="0 0 0"/><mass value="{mass:.6g}"/>
<inertia ixx="{I[0,0]:.6g}" iyy="{I[1,1]:.6g}" izz="{I[2,2]:.6g}" ixy="{I[0,1]:.6g}" ixz="{I[0,2]:.6g}" iyz="{I[1,2]:.6g}"/></inertial>
<visual name="visual"><geometry><mesh filename="visual.obj"/></geometry></visual>{col}</link></robot>
''')
            manifest.append(dict(name=name,dataset=ds,category=cat,source=str(d),bbox_dims_m=bbox.tolist(),
                object_scale=(bbox/a.object_base_size).tolist(),mass_kg=mass,convex_volume_m3=vol,convex_parts=len(parts),
                source_center_m=center.tolist()))
    (a.output/'manifest.json').write_text(json.dumps(dict(density_kg_m3=a.density,mass_range_kg=list(a.mass_range),objects=manifest),indent=1)+'\n')
    print(len(manifest),'objects')


if __name__=='__main__':main()
