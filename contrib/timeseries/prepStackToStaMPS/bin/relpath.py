#!/usr/bin/env python3
import os
import argparse

def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(description='Generate absolute path')
    parser.add_argument('-p','--path', dest='relpath', type=str, default=None, 
                            required=True, help='Relative path')
    return parser.parse_args() 

def relative(path):
    """ Get path from cwd to 'path' using relative notation (../../) """
    wd = os.path.abspath(os.getcwd())
    print (os.path.relpath(path, wd))
    return (os.path.relpath(path, wd))

if __name__ == '__main__':
    inps = cmdLineParse()
    relative((inps.relpath))
