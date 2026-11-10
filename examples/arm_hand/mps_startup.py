"""Bounded recovery of transient MPS startup failures, without CPU fallback."""
import os
import sys
import time
import warnings


def cuda_available(torch):
    with warnings.catch_warnings(record=True)as notices:
        warnings.simplefilter('always')
        available=torch.cuda.is_available()
    for notice in notices:print(str(notice.message),file=sys.stderr,flush=True)
    temporary=any('Error 807' in str(notice.message)for notice in notices)
    retries=int(os.environ.get('SPIDER_MPS_STARTUP_RETRIES','0'))
    if not available and temporary and os.environ.get('CUDA_MPS_PIPE_DIRECTORY')and retries<6:
        # CUDA keeps initialization errors in-process. Retry in a clean process,
        # retaining all CLI arguments and existing checkpoints.
        delay=min(60,5*2**retries)
        print(f'MPS temporarily unavailable (807); clean-process retry {retries+1}/6 in {delay}s',file=sys.stderr,flush=True)
        time.sleep(delay)
        environment=dict(os.environ,SPIDER_MPS_STARTUP_RETRIES=str(retries+1))
        os.execve(sys.executable,[sys.executable,'-u','-m','examples.arm_hand.optimize_kuka_gpu',*sys.argv[1:]],environment)
    return available
