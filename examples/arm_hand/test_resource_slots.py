"""Resource scheduling must use free slots and preserve the global cap."""
import fcntl
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from examples.arm_hand.resource_slots import resource_slot


class ResourceSlotTests(unittest.TestCase):
    def test_waiter_uses_any_available_slot(self):
        with tempfile.TemporaryDirectory()as directory:
            locks=[(Path(directory)/f'slot_{i:02d}.lock').open('a+')for i in range(2)]
            acquired=threading.Event();release=threading.Event()
            for stream in locks:fcntl.flock(stream,fcntl.LOCK_EX)
            def worker():
                with resource_slot(directory,2,block=True):
                    acquired.set();release.wait(3)
            thread=threading.Thread(target=worker);thread.start()
            try:
                time.sleep(.1)
                fcntl.flock(locks[(os.getpid()%2+1)%2],fcntl.LOCK_UN)
                self.assertTrue(acquired.wait(2),'A waiter ignored an available slot')
            finally:
                for stream in locks:fcntl.flock(stream,fcntl.LOCK_UN)
                release.set();thread.join(3)
                for stream in locks:stream.close()
            self.assertFalse(thread.is_alive())

    def test_cap_and_exception_release(self):
        with tempfile.TemporaryDirectory()as directory:
            counter=0;peak=0;mutex=threading.Lock()
            def worker():
                nonlocal counter,peak
                with resource_slot(directory,2,block=True):
                    with mutex:counter+=1;peak=max(peak,counter)
                    time.sleep(.05)
                    with mutex:counter-=1
            threads=[threading.Thread(target=worker)for _ in range(6)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(3);self.assertFalse(thread.is_alive())
            self.assertLessEqual(peak,2);self.assertEqual(counter,0)
            with self.assertRaises(RuntimeError):
                with resource_slot(directory,1,block=True):raise RuntimeError('test')
            with resource_slot(directory,1,block=True):pass


if __name__=='__main__':unittest.main()
