#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                  Giangi Sacco
#                        NASA Jet Propulsion Laboratory
#                      California Institute of Technology
#                        (C) 2009  All Rights Reserved
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import sys
import os
import math
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from iscesys.StdOE.StdOEPy import StdOEPy

def main():
     obj = StdOEPy()
    
     obj.writeStd("hello")
     filename = "testStdOut.log"
     obj.setStdOutFile(filename)
     tag = 'Test Tag Py'
     obj.setStdOutFileTag(tag)
     filename = "testStdErr.log"
     obj.setStdErrFile(filename)
     obj.writeStdOut("py first message")
     obj.writeStdErr("py first message")
     obj.writeStdOut("py second message")
     obj.writeStdErr("py second message")
     obj.setStdOut("screen")
     obj.setStdErr("screen")
     obj.writeStdOut("py first message")
     obj.writeStdErr("py first message")
     obj.writeStdOut("py second message")
     obj.writeStdErr("py second message")
     filename = "test.log"
     obj.writeStdFile(filename,"py first message")
     obj.writeStdFile(filename,"py second message")
     stdTypeRet = obj.getStdOut()
     obj.writeStd(stdTypeRet)
     stdTypeRet = obj.getStdErr()
     obj.writeStd(stdTypeRet)
     obj.setStdOut("file")
     obj.setStdErr("file")
     obj.writeStdOut("py third message")
     obj.writeStdErr("py third message")
if __name__ == "__main__":
    sys.exit(main())
