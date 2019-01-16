#!/usr/bin/env python3

# Author: Heresh Fattahi

from queue import Queue
import threading
import os
import argparse
import configparser


helpstr = '''
   Parallel processing different run commands
   '''

class customArgparseAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        '''
        The action to be performed.
        '''
        print(helpstr)
        parser.exit()

def createParser():
    parser = argparse.ArgumentParser( description='Preparing input rdf and processing workflow for phase unwrapping')

    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='input run file which contains multiple commands to be run. Each line is assumed to be a command that can be run independent of other commands')

    parser.add_argument('-p', '--number_of_processors', dest='processors', type=int, default=8,
            help='number of processors')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    inputArgs = parser.parse_args(args=iargs)

    return inputArgs


class ThreadRun(threading.Thread):
    """Threaded processing commands """
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            rr, opt_dict = self.queue.get()
            os.system(rr)
            self.queue.task_done()


def main(iargs=None):

    inputArgs = cmdLineParse(iargs)

    opt_dict={}
    opt_dict['parallel']=inputArgs.processors

    queue = Queue()
    #spawn a pool of threads, and pass them queue instance
    for i in range(opt_dict['parallel']):
        t = ThreadRun(queue)
        t.setDaemon(True)
        t.start()

    #populate queue with data
    runs = []
    for line in open(inputArgs.input):
        runs.append(line.strip())
    #for d in sorted(len(runs), key=operator.itemgetter('collectionName')):
    for rr in runs:
        queue.put([rr, opt_dict])

    #wait on the queue until everything has been processed
    queue.join()

if __name__ == "__main__":

    # Main engine
    main()
