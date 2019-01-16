#!/usr/bin/env python3 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                  Giangi Sacco
#                        NASA Jet Propulsion Laboratory
#                      California Institute of Technology
#                        (C) 2009  All Rights Reserved
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from __future__ import print_function
import sys
import os
import math
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from iscesys.StdOE import StdOE

## This class provides a set of convinient methods to access the public methods of the StdOE.ccp class.
# @see StdOE.cpp
class StdOEPy():

##
# Returns the value of StdOE::StdErr, i.e. the device where the standard error is ridirected.
# @return \c char StdOE::StdErr.
# @see StdOE::StdErr.
##
    def getStdErr(self):
        return StdOE.getStdErr_Py()

##
# Returns the  value of StdOE::StdOut, i.e. the device where the standard output is ridirected.
# @return \c char StdOE::StdOut.
# @see StdOE::StdOut.
##
        
    def getStdOut(self):
        return StdOE.getStdOut_Py()
##
# Sets the standard error device. The default is screen.
# @param stdErr  standard error device i.e. file or screen.
##
    def setStdErr(self,stdErr):
        StdOE.setStdErr_Py(stdErr)
        return

##
#  Sets a tag that precedes the date in the log file.
# @param tag  string to prepend to the date and log message.
# @see setStdLogFile().
# @see writeStdLog().
##
    def setStdLogFileTag(self,tag):
        StdOE.setStdLogFileTag_Py(tag)
        return
##
#  Sets a tag that precedes the date in the standard output file if the output device is a file.
# @param tag  string to prepend to the date and output message.
# @see setStdOutFile().
# @see setStdOut().
# @see writeStdOut().
##
    def setStdOutFileTag(self,tag):
        StdOE.setStdOutFileTag_Py(tag)
        return

##
#  Sets a tag that precedes the date in the standard error file if the output device is a file.
# @param tag  string to prepend to the date and error message.
# @see setStdErrFile().
# @see setStdErr().
# @see writeStdErr().
##
    def setStdErrFileTag(self,tag):
        StdOE.setStdErrFileTag_Py(tag)
        return
##
#  Sets the name of the file where the log is redirected.
# @param stdLogFile  standard error filename.
# @see StdOE::StdLog.
##
    def setStdLogFile(self,stdLogFile):
        StdOE.setStdLogFile_Py(stdLogFile)
        return
##
#  Sets the name of the file where the standard error is redirected. StdErr is set automatically to 'f', i.e. file.
# @param stdErrFile  standard error filename.
# @see StdOE::StdErr.
##
    def setStdErrFile(self,stdErrFile):
        StdOE.setStdErrFile_Py(stdErrFile)
        return
##
# Sets the standard output device. The default is screen.
# @param stdOut standard output device i.e. file or screen.
##
    def setStdOut(self,stdOut):
        StdOE.setStdOut_Py(stdOut)
        return

# Sets the name of the file where the standard output is redirected. StdOut is set automatically to 'f', i.e. file.
# @param stdOutFile  standard output filename.
# @see StdOE::StdOut.

    def setStdOutFile(self,stdOutFile):
        StdOE.setStdOutFile_Py(stdOutFile)
        return
##            
# Writes the string message on screen. 
# @param  message  string to be displayed on screen.
##

    def writeStd(self,message):
        StdOE.writeStd_Py(message)
        return
##
# Writes the string message on the preselected standard error device. If the device is a file, 
# it is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation). 
# @param  message  string to be written on the standard error device.
# @see asctime()
##
    def writeStdErr(self,message):
        StdOE.writeStdErr_Py(message)
        return
##
# Writes the string message in the log file StdOE:FilenameLog.  
# The message is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation). 
# @param  message  string to be written on the standard error device.
# @see asctime()
##
    def writeStdLog(self,message):
        StdOE.writeStdLog_Py(message)
        return
##            
# Writes the string message in the file "filename". 
# The message is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation). 
#@param  filename  name of the file where the string is written.
#@param  message  string to be written into the file.
# @see asctime()
##
            
    def writeStdFile(self,filename,message):
        StdOE.writeStdFile_Py(filename,message)
        return
##
# Writes the string message on the preselected standard output device. If the device is a file, 
# it is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation). 
# @param  message  string to be written on the standard error device.
# @see asctime()
##
    def writeStdOut(self,message):
        StdOE.writeStdOut_Py(message)
        return


    def __init__(self):
        return


#end class




if __name__ == "__main__":
    sys.exit(main())
