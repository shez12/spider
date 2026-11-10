"""Cross-process CPU/GPU concurrency limits for multi-source campaigns."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time


@contextmanager
def resource_slot(directory=None,count=1,block=False):
    if directory is None:
        yield
        return
    if count<1:raise ValueError('Slot count must be positive')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    streams=[(directory/f'slot_{i:02d}.lock').open('a+') for i in range(count)]
    waiter=(directory/'waiters.lock').open('a+') if block else None
    queued=False
    acquired=None
    try:
        while acquired is None:
            if waiter is not None and not queued:
                # Serialize waiting claimants, not running jobs. Keep scanning
                # ALL slots: blocking on one PID-selected slot can strand free
                # slots and serialize same-process worker threads.
                fcntl.flock(waiter,fcntl.LOCK_EX);queued=True
            for index in range(count):
                stream=streams[(index+os.getpid())%count]
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                acquired=stream;break
            if acquired is None:
                time.sleep(.05 if block else .2)
        if queued:
            fcntl.flock(waiter,fcntl.LOCK_UN);queued=False
        yield
    finally:
        if acquired is not None:fcntl.flock(acquired,fcntl.LOCK_UN)
        if waiter is not None:
            if queued:fcntl.flock(waiter,fcntl.LOCK_UN)
            waiter.close()
        for stream in streams:stream.close()
