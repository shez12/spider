"""Overlay actual object positions from all accepted GPU trajectories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'outputs/mustard_spatial_valid100')
    args=parser.parse_args();root=args.input_dir.resolve();out=root/'visualizations';out.mkdir(exist_ok=True)
    accepted=json.loads((root/'accepted.json').read_text())
    assert len(accepted)==100
    trajectories=[];angles=[];ids=[]
    for case in accepted:
        with np.load(case['trajectory']) as a:
            assert a['qpos'].shape[1]==36 and a['initial_state'].shape==(72,)
            # mjSTATE_FULLPHYSICS: time, qpos[36], qvel[35]. Object is the
            # last free joint, so its XYZ is qpos[-7:-4]. Include t=0.
            xyz=np.vstack([a['initial_state'][30:33],a['qpos'][:,-7:-4]])
        assert np.isfinite(xyz).all()
        trajectories.append(xyz)
        augmentation=json.loads((Path(case['folder'])/'augmentation.json').read_text())
        angles.append(augmentation['yaw_deg']);ids.append(case['candidate_id'])
    xyz=np.stack(trajectories);angles=np.array(angles)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(11,9),facecolor='white')
    ax=fig.add_subplot(111,projection='3d')
    fig.subplots_adjust(left=.04,right=.85,bottom=.18,top=.89)
    norm=Normalize(-10,10);cmap=plt.get_cmap('viridis')
    cm=xyz*100
    for path,yaw in zip(cm,angles):
        ax.plot(*path.T,color=cmap(norm(yaw)),linewidth=1.05,alpha=.67)
    ax.scatter(*cm[:,0].T,s=13,c='#228154',edgecolors='white',linewidths=.3,depthshade=False,zorder=5)
    ax.scatter(*cm[:,-1].T,s=20,c='#ce4937',marker='^',edgecolors='white',linewidths=.3,depthshade=False,zorder=5)
    ax.set_xlabel('X (cm)',labelpad=10);ax.set_ylabel('Y (cm)',labelpad=10)
    ax.set_zlabel('Z (cm)',labelpad=10)
    lo=cm.min(axis=(0,1));hi=cm.max(axis=(0,1));pad=np.maximum((hi-lo)*.08,1.)
    ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
    ax.set_zlim(0,hi[2]+pad[2])
    ax.set_box_aspect([hi[0]-lo[0]+2*pad[0],hi[1]-lo[1]+2*pad[1],hi[2]+pad[2]])
    ax.view_init(elev=23,azim=-62)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.set_facecolor((.96,.97,.98,1))
        axis._axinfo['grid'].update(color=(.78,.81,.84,.5),linewidth=.6)
    fig.suptitle('Mustard bottle：100 条有效物体轨迹',fontproperties=font,fontsize=19,y=.97)
    fig.text(.5,.925,'实际 GPU 动力学结果 · 同一世界坐标系 · 无平移对齐或缩放',
             ha='center',fontproperties=font,fontsize=11,color='#536274')
    handles=[Line2D([],[],color='none',marker='o',markerfacecolor='#228154',markeredgecolor='white',markersize=8,label='起点'),
             Line2D([],[],color='none',marker='^',markerfacecolor='#ce4937',markeredgecolor='white',markersize=9,label='终点')]
    ax.legend(handles=handles,prop=font,loc='upper left',bbox_to_anchor=(.02,.98),frameon=False)
    colorbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,fraction=.03,pad=.04,shrink=.62)
    colorbar.set_label('增强 yaw (°)',fontproperties=font,labelpad=12)
    fig.text(.5,.04,'每条曲线表示物体坐标原点的位置；颜色表示该候选的空间旋转增强角度。',
             ha='center',fontproperties=font,fontsize=10,color='#536274')
    basename=out/'object_trajectories_100'
    fig.savefig(basename.with_suffix('.png'),dpi=240,facecolor='white')
    fig.savefig(basename.with_suffix('.pdf'),facecolor='white')
    plt.close(fig)
    np.savez(out/'object_positions_100.npz',xyz_m=xyz,candidate_ids=ids,yaw_deg=angles,control_dt=.01)
    (out/'object_trajectories_100.json').write_text(json.dumps(dict(count=len(accepted),
        source='accepted.json actual selected GPU rollouts; includes initial state',
        position='object free-joint frame origin',coordinate_frame='unchanged dataset world frame',
        units_in_plot='cm',frames_per_trajectory=xyz.shape[1],candidate_ids=ids,
        min_xyz_m=xyz.min(axis=(0,1)).tolist(),max_xyz_m=xyz.max(axis=(0,1)).tolist()),indent=2)+'\n')
    print(basename.with_suffix('.png'))


if __name__=='__main__':main()
