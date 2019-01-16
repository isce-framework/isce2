#!/usr/bin/env python3
#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2016 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: David Bekaert
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import os
import glob
import sys
import isce
import isceobj
import argparse
from contrib.UnwrapComp.unwrapComponents import UnwrapComponents


def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unwrap interferogram using snaphu')
    parser.add_argument('-i', '--ifg', dest='intfile', type=str, required=True,
            help='Input interferogram')
    parser.add_argument('-c', '--coh', dest='cohfile', type=str, required=True,
            help='Coherence file')
    parser.add_argument('-u', '--unwprefix', dest='unwprefix', type=str, required=False,
            help='Output unwrapped file prefix')
    parser.add_argument('--nomcf', action='store_true', default=False,
            help='Run full snaphu and not in MCF mode, default = False')
    parser.add_argument('-a','--alks', dest='azlooks', type=int, default=1,
            help='Number of azimuth looks, default =1')
    parser.add_argument('-r', '--rlks', dest='rglooks', type=int, default=1,
            help='Number of range looks, default =1')
    parser.add_argument('-d', '--defomax', dest='defomax', type=float, default=2.0,
            help='Max cycles of deformation, default =2')    
    parser.add_argument('-m', '--method', dest='method', type=str, default='snaphu2stage',
            help='unwrapping method (snaphu, snaphu2stage= default, icu)')
    parser.add_argument('--overwrite', action='store_true', default=True,
            help='Overwrite file on re-run, default = True ===>>>> 2B implemented')
    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args = iargs)
    
 
def main(iargs=None):
    '''
    The main driver.
    ''' 

    inps = cmdLineParse(iargs)
    print(inps.method)
    if (inps.method != 'icu') and (inps.method != 'snaphu') and (inps.method != 'snaphu2stage'):
        raise Exception("Unwrapping method needs to be either icu, snaphu or snaphu2stage")



    # passign arguments
    if inps.nomcf: 
         nomcf_str =  " --nomcf "
    else:
         nomcf_str =  " "
    if inps.unwprefix:
        unwprefix_str = " - u " + inps.unwprefix
    else:
        unwprefix_str = " "



    #Get current directory
    currdir = os.getcwd()

    ##### Loop over the different intergerograms 
    for dirf in glob.glob(os.path.join(currdir, '2*',inps.intfile)):
        vals = dirf.split(os.path.sep)
        date = vals[-2]
        print(date)
        os.chdir(date)
        cmd = "step_unwrap_isce.py -i " + inps.intfile + " -c " + inps.cohfile + " -a " + str(inps.azlooks) + " -r " + str(inps.rglooks) + nomcf_str + " -m " + inps.method + unwprefix_str
        print(cmd)
        os.system(cmd)
        os.chdir('../.')

        continue

if __name__ == '__main__':

    main()
