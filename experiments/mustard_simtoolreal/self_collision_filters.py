"""Extra right-hand self-collision filtered pairs, applied through scene_utils._load_adjacent_links_map.

environment.create_env merges its MuJoCo exclusions into whatever _load_adjacent_links_map returns at call time,
so install() must run before create_env(). Assets and environment.py are untouched (reset-bank signature unchanged).
"""
FINGERS=('index','middle','ring','pinky')
SEGMENTS=('MCP_VL','PP','MP','DP')
THUMB=('right_thumb_CMC_VL','right_thumb_MC','right_thumb_MCP_VL','right_thumb_PP','right_thumb_DP')


def pairs(groups):
    out=[]
    for g in groups:
        if g=='palm-thumb':out+=[('right_hand_C_MC',t) for t in THUMB]
        elif g=='thumb-index':out+=[(t,f'right_index_{s}') for t in THUMB for s in SEGMENTS]
        elif g=='palm-fingers':out+=[('right_hand_C_MC',f'right_{f}_{s}') for f in FINGERS for s in SEGMENTS]+[('right_hand_C_MC','right_pinky_MC')]
        elif g=='pinky-ring':out+=[(f'right_pinky_{a}',f'right_ring_{b}') for a in ('MC',)+SEGMENTS for b in SEGMENTS]
        elif g=='adjacent-fingers':
            # MuJoCo-exported right finger meshes overlap their neighbours by 2-3.6 mm at the default and native
            # reset poses (audit_self_penetration.py); the original left meshes do not.
            for f1,f2 in (('index','middle'),('middle','ring'),('ring','pinky')):
                s1=SEGMENTS+(('MC',) if f1=='pinky' else ());s2=SEGMENTS+(('MC',) if f2=='pinky' else ())
                out+=[(f'right_{f1}_{a}',f'right_{f2}_{b}') for a in s1 for b in s2]
        else:raise ValueError(g)
    return out


def install(groups):
    import isaacsimenvs.tasks.simtoolreal.utils.scene_utils as scene
    original=scene._load_adjacent_links_map;extra=pairs(groups)
    def adjacency():
        result=original()
        for a,b in extra:result.setdefault(a,[]).append(b);result.setdefault(b,[]).append(a)
        return result
    scene._load_adjacent_links_map=adjacency
    return dict(groups=list(groups),pairs=extra)
