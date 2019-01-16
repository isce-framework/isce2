from __future__ import print_function
import sys
import threading
import itertools
import time

"""
This tool displays a rotating line while a task is running in another thread.
"""

class BarThread():

    def __init__(self, count=True):
        self.finished_event = threading.Event()
        self.progress_bar_thread = threading.Thread(target=self.run_progress_bar)
        self.starttime = time.time()
        self.count = count
        self.progress_bar_thread.start()

    def stop(self, count=None):
        self.endtime = time.time()
        self.finished_event.set()
        self.progress_bar_thread.join()
        self.elapsed = self.endtime - self.starttime
        if count is not None:
            self.count = count
        if self.count:
            sys.stdout.write('\rDone in %ds\n' % round(self.elapsed))
        else:
            sys.stdout.write('\rDone          \n')
        sys.stdout.flush()

    def run_progress_bar(self):
        chars = itertools.cycle(r'-\|/')
        while not self.finished_event.is_set():
            sys.stdout.write('\rWorking ' + next(chars))
            sys.stdout.flush()
            self.finished_event.wait(0.2)


if __name__ == "__main__":
    import time
    print('Testing...')
    t = BarThread()
    for i in range(20):
        time.sleep(0.1)
    t.stop()
    t = BarThread()
    for i in range(10):
        time.sleep(0.1)
    t.stop()


