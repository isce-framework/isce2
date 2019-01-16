#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




## pause is a raw_input wrapper
def pause(cont="go",ex="exit",ignore=False, message="", bell=True):
    """pause function.  Pauses execution awaiting input.
    Takes up to three optional arguments to set the action strings:
    cont   = first positional or named arg whose value is a string that causes execution
              to continue.
              Default cont="go"
    ex     = second positional or named arg whose value is a string that causes execution
              to stop.
              Default ex="exit"
    ignore = third positional or named arg whose value cause the pause to be ignored or
              paid attention to.
              Default False
    message = and optional one-time message to send to the user"
    bell    = True: ring the bell when pause is reached.
    """
    if not ignore:
        x = ""
        if message or bell:
            message += chr(7)*bell
            print(message)
        while x != cont:
            try:
                x = raw_input(
                    "Type %s to continue; %s to exit: " % (cont, ex)
                    )
            except KeyboardInterrupt:
                return None
            if x == ex:
                # return the "INTERUPT" system error.
                import errno
                import sys
                return sys.exit(errno.EINTR)
            pass
        pass
    return None
