#
# Author: Piyush Agram
# Copyright 2016
#

import sys
import isceobj
import os

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


    inpFile = os.path.join( self._insar.mergedDirname, self._insar.unwrappedIntFilename)
    ccFile = inpFile + '.conncomp'
    outFile = os.path.join( self._insar.mergedDirname, self.insar.unwrapped2StageFilename)

    # Hand over to 2Stage unwrap
    unw = UnwrapComponents()
    unw.setInpFile(inpFile)
    unw.setConnCompFile(ccFile)
    unw.setOutFile(outFile)
    unw.setSolver(solver_2stage)
    unw.setRedArcs(unwrapper_2stage_name)
    unw.unwrapComponents()
    return
