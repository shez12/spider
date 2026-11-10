"""Replace the right palm's convex-decomposition collision (32 pieces) with one convex hull, as PhysX does with the
original left hand's single palm mesh. Rewrites the URDF in the USD conversion work directory (distinct stem so the
converted USD is never shared with unmodified runs); source assets and environment.py are untouched.
Call after AppLauncher started and before create_env(); composes with physics_align (applied in the same hook chain).
"""
PALM='right_hand_C_MC'


def install():
    import xml.etree.ElementTree as ET
    from pathlib import Path
    import numpy as np,trimesh
    import isaacsimenvs.tasks.simtoolreal.utils.scene_utils as scene
    prepare=scene._prepare_urdf_for_isaacsim;info={}
    def hull_prepare(asset_path,usd_work_dir):
        source=Path(asset_path);tree=ET.parse(source);root=tree.getroot()
        link=next((l for l in root.findall('link') if l.get('name')==PALM),None)
        if link is None or len(link.findall('collision'))<2:return prepare(asset_path,usd_work_dir)
        for mesh in root.findall('.//mesh'):
            if mesh.get('filename'):mesh.set('filename',str(scene._resolve_urdf_mesh_path(source,mesh.get('filename'))))
        points=[]
        for c in link.findall('collision'):
            assert c.find('origin') is None,'palm collision with origin not supported'
            points.append(trimesh.load(c.find('geometry/mesh').get('filename'),force='mesh').vertices);link.remove(c)
        hull=trimesh.Trimesh(np.concatenate(points)).convex_hull
        out=Path(usd_work_dir)/'_palm_hull';out.mkdir(parents=True,exist_ok=True);hull.export(out/'palm_hull.obj')
        c=ET.SubElement(link,'collision',name='palm_hull');ET.SubElement(ET.SubElement(c,'geometry'),'mesh',filename=str(out/'palm_hull.obj'))
        path=out/(source.stem+'_palmhull.urdf');tree.write(path,encoding='utf-8',xml_declaration=True)
        info.update(pieces=len(points),hull_vertices=len(hull.vertices),hull_volume_cm3=float(hull.volume*1e6),urdf=str(path))
        return prepare(str(path),usd_work_dir)
    scene._prepare_urdf_for_isaacsim=hull_prepare
    return info
