#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Ravi Lanka
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# giangi: taken Piyush code for snaphu and adapted

import sys
import isceobj

from contrib.UnwrapComp.unwrapComponents import UnwrapComponents

def runUnwrap2Stage(self, unwrapper_2stage_name=None, solver_2stage=None):

    if unwrapper_2stage_name is None:
        unwrapper_2stage_name = 'REDARC0'
    
    if solver_2stage is None:
        # If unwrapper_2state_name is MCF then solver is ignored
        # and relaxIV MCF solver is used by default
        solver_2stage = 'pulp'
    
    print('Unwrap 2 Stage Settings:')
    print('Name: %s'%unwrapper_2stage_name)
    print('Solver: %s'%solver_2stage)

    inpFile = self.insar.unwrappedIntFilename
    ccFile  = self.insar.connectedComponentsFilename
    outFile = self.insar.unwrapped2StageFilename

    # Hand over to 2Stage unwrap
    unw = UnwrapComponents()
    unw.setInpFile(inpFile)
    unw.setConnCompFile(ccFile)
    unw.setOutFile(outFile)
    unw.setSolver(solver_2stage)
    unw.setRedArcs(unwrapper_2stage_name)
    unw.unwrapComponents()
    return
