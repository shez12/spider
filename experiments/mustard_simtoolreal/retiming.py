"""Independent, continuous time parameterization for native SimToolReal actions."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def continuous_plan(previous, commands, lo, hi, arm_alpha, hand_alpha, speed_dt,
                    realized_arm=None, source_dt=.01, policy_dt=1/60, smoothing_sigma=2., hold_steps=30):
    n,steps,joints=commands.shape
    assert previous.shape==(n,joints) and 0<arm_alpha<=1 and 0<hand_alpha<=1
    margin=(hi-lo)*.001
    initial=np.clip(previous,lo+margin,hi-margin)
    reference=np.concatenate([previous[:,None],commands],axis=1).copy()
    if realized_arm is not None:
        assert realized_arm.shape==(n,steps,7)
        reference[:,1:,:7]=realized_arm
    original=reference.copy()
    if smoothing_sigma:
        reference[:,:, :7]=gaussian_filter1d(reference[:,:,:7],smoothing_sigma,axis=1,mode='nearest')
    reference[:,0]=initial
    reference[:,-1,:7]=original[:,-1,:7]
    reference=np.clip(reference,lo+margin,hi-margin)
    paths=[];labels=[];times=[];lengths=[];durations=[];hand_error=[];sources=[]
    for i in range(n):
        curve=reference[i]
        delta=np.abs(np.diff(curve[:,:7],axis=0))
        # Bound both EMA effective motion and its inner raw-target clamp.
        distance=np.minimum(np.minimum(curve[:-1,:7]-lo[:7],hi[:7]-curve[:-1,:7]),
                            np.minimum(curve[1:,:7]-lo[:7],hi[:7]-curve[1:,:7]))
        per_step=.95*arm_alpha*np.minimum(speed_dt,distance)
        segment=np.maximum((delta/np.maximum(per_step,1e-12)).max(-1),source_dt/policy_dt)
        clock=np.r_[0.,np.cumsum(segment)]
        sample=np.minimum(np.arange(1,int(np.ceil(clock[-1]))+1),clock[-1])
        source=np.interp(sample,clock,np.arange(steps+1))
        wanted=np.stack([np.interp(sample,clock,curve[:,j]) for j in range(joints)],axis=-1)
        wanted=np.concatenate([wanted,np.repeat(wanted[-1:],hold_steps,axis=0)])
        source=np.r_[source,np.repeat(source[-1],hold_steps)]
        p=initial[i].copy();a_list=[];p_list=[];err=0.
        for q in wanted:
            a=np.empty(joints)
            a[:7]=(q[:7]-p[:7])/(arm_alpha*speed_dt)
            hand_raw=(q[7:]-(1-hand_alpha)*p[7:])/hand_alpha
            # Hand lag is an explicit path adjustment, not a claim of exact
            # reproduction: retain feasible actions and record resulting error.
            a[7:]=np.clip(2*(hand_raw-lo[7:])/(hi[7:]-lo[7:])-1,-1,1)
            assert np.abs(a[:7]).max()<=1.00001
            nxt=np.empty(joints)
            nxt[:7]=(1-arm_alpha)*p[:7]+arm_alpha*np.clip(p[:7]+speed_dt*a[:7],lo[:7],hi[:7])
            nxt[7:]=(1-hand_alpha)*p[7:]+hand_alpha*(lo[7:]+.5*(a[7:]+1)*(hi[7:]-lo[7:]))
            assert np.max(np.abs(nxt[:7]-q[:7]))<1e-7
            err=max(err,float(np.abs(nxt[7:]-q[7:]).max()))
            a_list.append(a);p_list.append(nxt);p=nxt
        paths.append(np.asarray(p_list));labels.append(np.asarray(a_list));lengths.append(len(wanted))
        durations.append(float(clock[-1]*policy_dt));hand_error.append(err);sources.append(source)
    total=max(lengths)
    padded_targets=np.empty((total,n,joints),dtype=np.float32)
    padded_actions=np.empty_like(padded_targets)
    padded_source=np.empty((total,n),dtype=np.float32)
    for i,length in enumerate(lengths):
        padded_targets[:length,i]=paths[i];padded_actions[:length,i]=labels[i];padded_source[:length,i]=sources[i]
        padded_targets[length:,i]=paths[i][-1];padded_source[length:,i]=sources[i][-1]
        equilibrium=np.r_[np.zeros(7),2*(paths[i][-1,7:]-lo[7:])/(hi[7:]-lo[7:])-1]
        padded_actions[length:,i]=equilibrium
    audit=dict(method='independent_continuous_v1',source_steps=steps,retimed_steps=total,
        path_reference='realized_arm_positions_and_hand_control_targets' if realized_arm is not None else 'control_targets',
        independent_trajectory_clocks=True,wait_for_other_trajectories=False,exact_waypoint_wait=False,
        valid_lengths=lengths,trajectory_duration_s=[v*policy_dt for v in lengths],movement_duration_s=durations,
        duration_min_s=min(lengths)*policy_dt,duration_median_s=float(np.median(lengths)*policy_dt),duration_max_s=total*policy_dt,
        trajectories_within_600_steps=sum(v<=600 for v in lengths),hold_steps=hold_steps,
        arm_smoothing_sigma_source_frames=smoothing_sigma,
        arm_path_adjustment_max_rad=float(np.abs(reference[:,:,:7]-original[:,:,:7]).max()),
        initial_projection_max_rad=float(np.abs(initial-previous).max()),
        hand_lag_max_rad_per_trajectory=hand_error,
        action_max_abs=float(np.abs(padded_actions).max()),
        warning='Feasible command schedule, not a certificate of successful physical grasp. Padding is excluded from BC.')
    return initial,padded_actions,padded_targets,padded_source,audit
