#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                  Giangi Sacco
#                        NASA Jet Propulsion Laboratory
#                      California Institute of Technology
#                        (C) 2009-2012  All Rights Reserved
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from __future__ import print_function
import inspect 

raise DeprecationWarning("DebugLiner is being deleted for want of client")

__all__ = ('printLine', 'printFile', 'printInfo')

# a decorator to do the work-- untested.
def debug_message(func):
    item = func().capitalize()
    def dfunc():
        frame = inspect.currentframe().f_back 
        result = frameInfo[1] if frame or frame is not None else ' not available'
        print(item, result)
        return None
    return dfunc()


@debug_message
def printLine():
    return "Line"

@debug_message
def printFile():
    return "File"

## Whoops, decorator doesn't do this:
def printInfo():
    frame = inspect.currentframe().f_back 
    if frame or not frame == None:
        frameInfo = inspect.getframeinfo(frame)
        print ("File %s line %s"%(frameInfo[0], str(frameInfo[1])))
    else:
        print ("Info not available")
