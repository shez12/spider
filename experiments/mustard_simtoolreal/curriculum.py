"""Monotonic warm-start curriculum; poor performance holds difficulty fixed."""


def mean_goal_threshold(max_targets):
    # A one/two-goal episode cannot reach the native threshold of three.
    return .8*max_targets if max_targets<3 else 3.


def next_difficulty(*,mean_goals,max_targets,target_level,target_levels,position_level,
                    position_levels,next_dimension,seconds,restore_native,native_tolerance_enabled):
    """Return one forward step, or None. There is intentionally no rollback."""
    if mean_goals<mean_goal_threshold(max_targets):return None
    target_available=target_level is not None and target_level<target_levels-1
    position_available=position_level<position_levels-1
    if target_available and (next_dimension=='target' or not position_available):return 'target'
    if position_available:return 'position'
    if restore_native and seconds>10:return 'timeout'
    if not native_tolerance_enabled:return 'enable_native_tolerance'
    return None
