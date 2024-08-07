#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import shutil
import datetime
import numpy as np
import xml.etree.ElementTree as ET

import isce, isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

from StackPulic import loadStackUserParameters
from StackPulic import loadInsarUserParameters
from StackPulic import acquisitionModesAlos2
from StackPulic import datesFromPairs


def checkDem(fileName):
    if fileName is None:
        raise Exception('dem for coregistration, dem for geocoding, water body must be set')
    else:
        if not os.path.isfile(fileName):
            raise Exception('file not found: {}'.format(fileName))
        else:
            img = isceobj.createDemImage()
            img.load(fileName+'.xml')
            if os.path.abspath(fileName) != img.filename:
                raise Exception('please use absolute path for <property name="file_name"> in {} xml file'.format(fileName))


def getFolders(directory):
    '''
    return sorted folders in a directory
    '''
    import os
    import glob

    folders = glob.glob(os.path.join(os.path.abspath(directory), '*'))
    folders = sorted([os.path.basename(x) for x in folders if os.path.isdir(x)])

    return folders


def unionLists(list1, list2):
    import copy

    list3 = copy.deepcopy(list1)

    for x in list2:
        if x not in list1:
            list3.append(x)

    return sorted(list3)


def removeCommonItemsLists(list1, list2):
    '''
    remove common items of list1 and list2 from list1
    '''

    import copy

    list3 = copy.deepcopy(list1)

    list4 = []
    for x in list1:
        if x in list2:
            list3.remove(x)
            list4.append(x)

    return (sorted(list3), sorted(list4))


def formPairs(idir, numberOfSubsequentDates, pairTimeSpanMinimum=None, pairTimeSpanMaximum=None, 
    datesIncluded=None, pairsIncluded=None, 
    datesExcluded=None, pairsExcluded=None):
    '''
    datesIncluded: list
    pairsIncluded: list
    datesExcluded: list
    pairsExcluded: list
    '''
    datefmt = "%y%m%d"

    #get date folders [and ignore sub-folders that are not digits, e.g. ARCHIVED_FILES]
    dateDirs = sorted(glob.glob(os.path.join(os.path.abspath(idir), '*')))
    dateDirs = [x for x in dateDirs if os.path.isdir(x) and os.path.basename(x).isdigit()]
    dates = [os.path.basename(x) for x in dateDirs]
    ndate = len(dates)

    #check input parameters
    if datesIncluded is not None:
        if type(datesIncluded) != list:
            raise Exception('datesIncluded must be a list')
        for date in datesIncluded:
            if date not in dates:
                raise Exception('in datesIncluded, date {} is not found in data directory {}'.format(date, idir))

    if pairsIncluded is not None:
        if type(pairsIncluded) != list:
            raise Exception('pairsIncluded must be a list')
        #check reference must < secondary
        for pair in pairsIncluded:
            rdate = pair.split('-')[0]
            sdate = pair.split('-')[1]
            rtime = datetime.datetime.strptime(rdate, datefmt)
            stime = datetime.datetime.strptime(sdate, datefmt)
            if rtime >= stime:
                raise Exception('in pairsIncluded, first date must be reference') 
            if (sdate not in dates) or (mdate not in dates):
                raise Exception('in pairsIncluded, reference or secondary date of pair {} not in data directory {}'.format(pair, idir)) 

    if datesExcluded is not None:
        if type(datesExcluded) != list:
            raise Exception('datesExcluded must be a list')
    if pairsExcluded is not None:
        if type(pairsExcluded) != list:
            raise Exception('pairsExcluded must be a list')

    #get initial pairs to process
    pairsProcess = []
    for i in range(ndate):
        rdate = dates[i]
        rtime = datetime.datetime.strptime(rdate, datefmt)
        for j in range(numberOfSubsequentDates):
            if i+j+1 <= ndate - 1:
                sdate = dates[i+j+1]
                stime = datetime.datetime.strptime(sdate, datefmt)
                pair = rdate + '-' + sdate
                ts = np.absolute((stime - rtime).total_seconds()) / (365.0 * 24.0 * 3600)
                if pairTimeSpanMinimum is not None:
                    if ts < pairTimeSpanMinimum:
                        continue
                if pairTimeSpanMaximum is not None:
                    if ts > pairTimeSpanMaximum:
                        continue
                pairsProcess.append(pair)

    #included dates
    if datesIncluded is not None:
        pairsProcess2 = []
        for pair in pairsProcess:
            rdate = pair.split('-')[0]
            sdate = pair.split('-')[1]
            if (rdate in datesIncluded) or (sdate in datesIncluded):
                pairsProcess2.append(pair)
        pairsProcess = pairsProcess2

    #included pairs
    if pairsIncluded is not None:
        pairsProcess = pairsIncluded

    #excluded dates
    if datesExcluded is not None:
        pairsProcess2 = []
        for pair in pairsProcess:
            rdate = pair.split('-')[0]
            sdate = pair.split('-')[1]
            if (rdate not in datesExcluded) and (sdate not in datesExcluded):
                pairsProcess2.append(pair)
        pairsProcess = pairsProcess2

    #excluded pairs
    if pairsExcluded is not None:
        pairsProcess2 = []
        for pair in pairsProcess:
            if pair not in pairsExcluded:
               pairsProcess2.append(pair)
        pairsProcess = pairsProcess2

    # #datesProcess
    # datesProcess = []
    # for pair in pairsProcess:
    #     rdate = pair.split('-')[0]
    #     sdate = pair.split('-')[1]
    #     if rdate not in datesProcess:
    #         datesProcess.append(rdate)
    #     if sdate not in datesProcess:
    #         datesProcess.append(sdate)
    
    # datesProcess = sorted(datesProcess)
    pairsProcess = sorted(pairsProcess)

    #return (datesProcess, pairsProcess)
    return pairsProcess


def stackRank(dates, pairs):
    from numpy.linalg import matrix_rank

    dates = sorted(dates)
    pairs = sorted(pairs)
    ndate = len(dates)
    npair = len(pairs)

    #observation matrix
    H0 = np.zeros((npair, ndate))
    for k in range(npair):
        dateReference = pairs[k].split('-')[0]
        dateSecondary = pairs[k].split('-')[1]
        dateReference_i = dates.index(dateReference)
        H0[k, dateReference_i] = 1
        dateSecondary_i = dates.index(dateSecondary)
        H0[k, dateSecondary_i] = -1

    rank = matrix_rank(H0)

    return rank




def checkStackDataDir(idir):
    '''
    idir:          input directory where data of each date is located. only folders are recognized
    '''
    stack.dataDir

    #get date folders
    dateDirs = sorted(glob.glob(os.path.join(os.path.abspath(idir), '*')))
    dateDirs = [x for x in dateDirs if os.path.isdir(x) and os.path.basename(x).isdigit()]

    #check dates and acquisition mode
    mode = os.path.basename(sorted(glob.glob(os.path.join(dateDirs[0], 'IMG-HH-ALOS2*')))[0]).split('-')[4][0:3]
    for x in dateDirs:
        dateFolder = os.path.basename(x)
        images = sorted(glob.glob(os.path.join(x, 'IMG-HH-ALOS2*')))
        leaders = sorted(glob.glob(os.path.join(x, 'LED-ALOS2*')))
        for y in images:
            dateFile   = os.path.basename(y).split('-')[3]
            if dateFolder != dateFile:
                raise Exception('date: {} in data folder name is different from date: {} in file name: {}'.format(dateFolder, dateFile, y))
            ymode = os.path.basename(y).split('-')[4][0:3]
            if mode != ymode:
                #currently only allows S or D polarization, Q should also be OK?
                if (mode[0:2] == ymode[0:2]) and (mode[2] in ['S', 'D']) and (ymode[2] in ['S', 'D']):
                    pass
                else:
                    raise Exception('all acquisition modes should be the same')

        for y in leaders:
            dateFile   = os.path.basename(y).split('-')[2]
            if dateFolder != dateFile:
                raise Exception('date: {} in data folder name is different from date: {} in file name: {}'.format(dateFolder, dateFile, y))
            ymode = os.path.basename(y).split('-')[3][0:3]
            if mode != ymode:
                #currently only allows S or D polarization, Q should also be OK?
                if (mode[0:2] == ymode[0:2]) and (mode[2] in ['S', 'D']) and (ymode[2] in ['S', 'D']):
                    pass
                else:
                    raise Exception('all acquisition modes should be the same')


def createCmds(stack, datesProcess, pairsProcess, pairsProcessIon, mode, nthreads=4, nruns=10):
    '''
    create scripts to process an InSAR stack
    '''
    import os
    import copy

    stack.dem = os.path.abspath(stack.dem)
    stack.demGeo = os.path.abspath(stack.demGeo)
    stack.wbd = os.path.abspath(stack.wbd)

    insar = stack

    def header(txt):
        hdr  = '##################################################\n'
        hdr += '# {}\n'.format(txt)
        hdr += '##################################################\n'
        return hdr


    stackScriptPath = os.path.join(os.environ['ISCE_STACK'], 'alosStack')

    def stepSettings(nthreads=4, nruns=10, array='', loop=True):
        if loop == False:
            settings = '''
# Number of threads
export OMP_NUM_THREADS={}

'''.format(nthreads)
        else:
            settings = '''
# Number of threads
export OMP_NUM_THREADS={}

# Number of parallel runs
nruns={}

# set the array variable used. The array can be found at the
# beginning of this command file. You can redefine it here if
# you want to process different dates or pairs.
array=("${{{}[@]}}")

'''.format(nthreads, nruns, array)
        return settings

    def forLoop(functionName, array, nruns):
        txt='''for ((i=0;i<${{#{array}[@]}};i++)); do

  {functionName} ${{{array}[i]}} &

  if [[ $(((i+1) % {nruns})) -eq 0 ]]; then
    wait
  fi

  #add a wait to the final group of runs to secure the completion of the for loop before anything can proceed
  #if the wait was not added previously
  if [[ $((i+1)) -eq ${{#{array}[@]}} ]]; then
    if [[ $((${{#{array}[@]}} % {nruns})) -ne 0 ]]; then
      wait
    fi
  fi

done
'''.format(functionName              = functionName,
               array                 = array,
               nruns                 = nruns) #make it a variable, not a number, as in stepSettings
        return txt

    def stepDone():
        txt='''
echo "$(basename "$0") done"
'''
        return txt


    functionName = 'process'

    print('                       * * *')
    if stack.dateReferenceStack in datesProcess:
        print('reference date of stack in date list to be processed.')
        if os.path.isfile(os.path.join(stack.datesResampledDir, stack.dateReferenceStack, 'insar', 'affine_transform.txt')):
            print('reference date of stack already processed previously.')
            print('do not implement reference-date-related processing this time.')
            processDateReferenceStack = False
        else:
            print('reference date of stack not processed previously.')
            print('implement reference-date-related processing this time.')
            processDateReferenceStack = True
    else:
        print('reference date of stack NOT in date list to be processed.')
        if not os.path.isfile(os.path.join(stack.datesResampledDir, stack.dateReferenceStack, 'insar', 'affine_transform.txt')):
            raise Exception('but it does not seem to have been processed previously.')
        else:
            print('assume it has already been processed previously.')
            print('do not implement reference-date-related processing this time.')
            processDateReferenceStack = False
    print('                       * * *')
    print()

    #WHEN PROVIDING '-sec_date' BECAREFUL WITH 'datesProcess' AND 'datesProcessSecondary'
    datesProcessSecondary = copy.deepcopy(datesProcess)
    if stack.dateReferenceStack in datesProcessSecondary:
        datesProcessSecondary.remove(stack.dateReferenceStack)

    #pairs also processed in regular InSAR processing
    pairsProcessIon1 = [ipair for ipair in pairsProcessIon if ipair in pairsProcess]
    #pairs  not processed in regular InSAR processing
    pairsProcessIon2 = [ipair for ipair in pairsProcessIon if ipair not in pairsProcess]


    cmds = []
    step_names = []


    #start new commands: processing each date
    #################################################################################
    cmd0  = '#!/bin/bash\n\n'
    cmd0 += 'dates=({})\n'.format(' '.join(datesProcess))
    cmd0 += 'dates2=({})\n'.format(' '.join(datesProcessSecondary))
    cmd0 += '\n\n'


    #read data
    step_name = 'read data'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
    if datesProcess != []:
        #cmd += header('read data')
        cmd += os.path.join(stackScriptPath, 'read_data.py') + ' -idir {} -odir {} -ref_date {} -sec_date {} -pol {}'.format(stack.dataDir, stack.datesProcessingDir, stack.dateReferenceStack, ' '.join(datesProcess), stack.polarization)
        if stack.frames is not None:
            cmd += ' -frames {}'.format(' '.join(stack.frames))
        if stack.startingSwath is not None:
            cmd += ' -starting_swath {}'.format(stack.startingSwath)
        if stack.endingSwath is not None:
            cmd += ' -ending_swath {}'.format(stack.endingSwath)
        if insar.useVirtualFile:
            cmd += ' -virtual'
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
        #frame and swath names use those from frame and swath dirs from now on
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #compute baseline
    step_name = 'compute baseline'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
    if datesProcessSecondary != []:
        #cmd += header('compute baseline')
        cmd += os.path.join(stackScriptPath, 'compute_baseline.py') + ' -idir {} -odir {} -ref_date {} -sec_date {} -baseline_center baseline_center.txt -baseline_grid -baseline_grid_width 10 -baseline_grid_length 10'.format(stack.datesProcessingDir, stack.baselineDir, stack.dateReferenceStack, ' '.join(datesProcessSecondary))
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #compute burst synchronization
    step_name = 'compute burst synchronization'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
    spotlightModes, stripmapModes, scansarNominalModes, scansarWideModes, scansarModes = acquisitionModesAlos2()
    if mode in scansarNominalModes:
        #cmd += header('compute burst synchronization')
        cmd += os.path.join(stackScriptPath, 'compute_burst_sync.py') + ' -idir {} -burst_sync_file burst_synchronization.txt -ref_date {}'.format(stack.datesProcessingDir, stack.dateReferenceStack)
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #estimate SLC offsets
    step_name = 'estimate SLC offsets'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='dates2', loop=True)
    if datesProcessSecondary != []:
        extraArguments = ''
        if insar.useWbdForNumberOffsets is not None:
            extraArguments += ' -use_wbd_offset'
        if insar.numberRangeOffsets is not None:
            for x in insar.numberRangeOffsets:
                extraArguments += ' -num_rg_offset {}'.format(' '.join(x))
        if insar.numberAzimuthOffsets is not None:
            for x in insar.numberAzimuthOffsets:
                extraArguments += ' -num_az_offset {}'.format(' '.join(x))

        #cmd += header('estimate SLC offsets')
        #cmd += parallelSettings('dates2')
        cmd += '''{functionName} () {{
  date=$1

  {script} -idir {datesProcessingDir} -ref_date {dateReferenceStack} -sec_date ${{date}} -wbd {wbd} -dem {dem}{extraArguments}
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'estimate_slc_offset.py'),
               datesProcessingDir  = stack.datesProcessingDir,
               dateReferenceStack  = stack.dateReferenceStack,
               wbd                 = insar.wbd, 
               dem                 = stack.dem,
               extraArguments      = extraArguments)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #estimate swath offsets
    step_name = 'estimate swath offsets'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
    if processDateReferenceStack:
        #cmd += header('estimate swath offsets')
        cmd += os.path.join(stackScriptPath, 'estimate_swath_offset.py') + ' -idir {} -date {} -output swath_offset.txt'.format(os.path.join(stack.datesProcessingDir, stack.dateReferenceStack), stack.dateReferenceStack)
        if insar.swathOffsetMatching:
            cmd += ' -match'
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #estimate frame offsets
    step_name = 'estimate frame offsets'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
    if processDateReferenceStack:
        #cmd += header('estimate frame offsets')
        cmd += os.path.join(stackScriptPath, 'estimate_frame_offset.py') + ' -idir {} -date {} -output frame_offset.txt'.format(os.path.join(stack.datesProcessingDir, stack.dateReferenceStack), stack.dateReferenceStack)
        if insar.frameOffsetMatching:
            cmd += ' -match'
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #resample to a common grid
    step_name = 'resample to a common grid'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='dates', loop=True)
    if datesProcess != []:
        extraArguments = ''
        if stack.gridFrame is not None:
            extraArguments += ' -ref_frame {}'.format(stack.gridFrame)
        if stack.gridSwath is not None:
            extraArguments += ' -ref_swath {}'.format(stack.gridSwath)
        if insar.doIon:
            extraArguments += ' -subband'

        #cmd += header('resample to a common grid')
        #cmd += parallelSettings('dates')
        cmd += '''{functionName} () {{
  date=$1

  {script} -idir {datesProcessingDir} -odir {datesResampledDir} -ref_date {dateReferenceStack} -sec_date ${{date}} -nrlks1 {numberRangeLooks1} -nalks1 {numberAzimuthLooks1}{extraArguments}
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'resample_common_grid.py'),
               datesProcessingDir  = stack.datesProcessingDir,
               datesResampledDir   = stack.datesResampledDir,
               dateReferenceStack  = stack.dateReferenceStack,
               numberRangeLooks1   = insar.numberRangeLooks1, 
               numberAzimuthLooks1 = insar.numberAzimuthLooks1,
               extraArguments      = extraArguments)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #mosaic parameter
    step_name = 'mosaic parameter'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
    if datesProcess != []:
        #cmd += header('mosaic parameter')
        cmd += os.path.join(stackScriptPath, 'mosaic_parameter.py') + ' -idir {} -ref_date {} -sec_date {} -nrlks1 {} -nalks1 {}'.format(stack.datesProcessingDir, stack.dateReferenceStack, ' '.join(datesProcess), insar.numberRangeLooks1, insar.numberAzimuthLooks1)
        if stack.gridFrame is not None:
            cmd += ' -ref_frame {}'.format(stack.gridFrame)
        if stack.gridSwath is not None:
            cmd += ' -ref_swath {}'.format(stack.gridSwath)
        cmd += '\n'

    if processDateReferenceStack:
        cmd += os.path.join(stackScriptPath, 'mosaic_parameter.py') + ' -idir {} -ref_date {} -sec_date {} -nrlks1 {} -nalks1 {}'.format(stack.datesResampledDir, stack.dateReferenceStack, stack.dateReferenceStack, insar.numberRangeLooks1, insar.numberAzimuthLooks1)
        if stack.gridFrame is not None:
            cmd += ' -ref_frame {}'.format(stack.gridFrame)
        if stack.gridSwath is not None:
            cmd += ' -ref_swath {}'.format(stack.gridSwath)
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    else:
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #compute lat/lon/hgt
    step_name = 'compute latitude longtitude and height'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
    if processDateReferenceStack:
        #cmd += header('compute latitude, longtitude and height')
        cmd += 'cd {}\n'.format(os.path.join(stack.datesResampledDir, stack.dateReferenceStack))
        cmd += os.path.join(stackScriptPath, 'rdr2geo.py') + ' -date {} -dem {} -wbd {} -nrlks1 {} -nalks1 {}'.format(stack.dateReferenceStack, stack.dem, insar.wbd, insar.numberRangeLooks1, insar.numberAzimuthLooks1)
        if insar.useGPU:
            cmd += ' -gpu'
        cmd += '\n'

        # #should move it to look section???!!!
        # cmd += os.path.join(stackScriptPath, 'look_geom.py') + ' -date {} -wbd {} -nrlks1 {} -nalks1 {} -nrlks2 {} -nalks2 {}'.format(stack.dateReferenceStack, insar.wbd, insar.numberRangeLooks1, insar.numberAzimuthLooks1, insar.numberRangeLooks2, insar.numberAzimuthLooks2)
        # cmd += '\n'

        cmd += 'cd ../../'
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #compute geometrical offsets
    step_name = 'compute geometrical offsets'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='dates2', loop=True)
    if datesProcessSecondary != []:
        extraArguments = ''
        if insar.useGPU:
            extraArguments += ' -gpu'
            cmd += '''
# Number of parallel runs. Since it uses GPU and is fast. do not use multiple runs due to GPU memory limitations.
nruns=1

'''

        #cmd += header('compute geometrical offsets')
        #cmd += parallelSettings('dates2')
        cmd += '''{functionName} () {{
  date=$1

  cd {datesResampledDir}
  {script} -date ${{date}} -date_par_dir {datesProcessingDir} -lat {lat} -lon {lon} -hgt {hgt} -nrlks1 {numberRangeLooks1} -nalks1 {numberAzimuthLooks1}{extraArguments}
  cd ../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'geo2rdr.py'),
               datesResampledDir   = os.path.join(stack.datesResampledDir, '${dates2[i]}'),
               datesProcessingDir  = os.path.join('../../', stack.datesProcessingDir, '${dates2[i]}'),
               lat                 = '../{}/insar/{}_{}rlks_{}alks.lat'.format(stack.dateReferenceStack, stack.dateReferenceStack, insar.numberRangeLooks1, insar.numberAzimuthLooks1),
               lon                 = '../{}/insar/{}_{}rlks_{}alks.lon'.format(stack.dateReferenceStack, stack.dateReferenceStack, insar.numberRangeLooks1, insar.numberAzimuthLooks1), 
               hgt                 = '../{}/insar/{}_{}rlks_{}alks.hgt'.format(stack.dateReferenceStack, stack.dateReferenceStack, insar.numberRangeLooks1, insar.numberAzimuthLooks1),
               numberRangeLooks1   = insar.numberRangeLooks1,
               numberAzimuthLooks1 = insar.numberAzimuthLooks1,
               extraArguments      = extraArguments)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #save commands
    cmds_1 = cmds
    step_names_1 = step_names



    cmds = []
    step_names = []


    if pairsProcess != []:
        #start new commands: processing each pair before ionosphere correction
        #################################################################################
        cmd0  = '#!/bin/bash\n\n'
        cmd0 += 'insarpair=({})\n'.format(' '.join(pairsProcess))
        cmd0 += 'dates2=({})\n'.format(' '.join(datesProcessSecondary))
        cmd0 += '\n\n'
    else:
        cmd0  = '#!/bin/bash\n\n'
        cmd0 += '#no pairs for InSAR processing.'


    #pair up
    step_name = 'pair up'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
    if pairsProcess != []:
        #cmd += header('pair up')
        cmd += os.path.join(stackScriptPath, 'pair_up.py') + ' -idir1 {} -idir2 {} -odir {} -ref_date {} -pairs {}'.format(stack.datesProcessingDir, stack.datesResampledDir, stack.pairsProcessingDir, stack.dateReferenceStack, ' '.join(pairsProcess))
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #form interferograms
    step_name = 'form interferograms'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='insarpair', loop=True)
    if pairsProcess != []:
        #cmd += header('form interferograms')
        #cmd += parallelSettings('insarpair')
        cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1}
  cd ../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'form_interferogram.py'),
               pairsProcessingDir  = stack.pairsProcessingDir,
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #mosaic interferograms
    step_name = 'mosaic interferograms'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='insarpair', loop=True)
    if pairsProcess != []:
        wbd1 = '../../dates_resampled/{}/insar/{}_{}rlks_{}alks.wbd'.format(stack.dateReferenceStack, stack.dateReferenceStack, insar.numberRangeLooks1, insar.numberAzimuthLooks1)

        #cmd += header('mosaic interferograms')
        #cmd += parallelSettings('insarpair')
        cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -ref_date_stack {ref_date_stack} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1} -water_body {wbd}
  cd ../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'mosaic_interferogram.py'),
               pairsProcessingDir  = stack.pairsProcessingDir,
               ref_date_stack      = stack.dateReferenceStack,
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1,
               wbd                 = wbd1)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #estimate residual offsets between radar and DEM
    step_name = 'estimate residual offsets between radar and DEM'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
    if processDateReferenceStack:
    #if not os.path.isfile(os.path.join(stack.datesResampledDir, stack.dateReferenceStack, 'insar', 'affine_transform.txt')):
        #amplitde image of any pair should work, since they are all coregistered now
        if pairsProcess == []:
            pairsProcessTmp = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(stack.pairsProcessingDir, '*'))) if os.path.isdir(x)]
        else:
            pairsProcessTmp = pairsProcess
        if pairsProcessTmp == []:
            raise Exception('no InSAR pairs available for estimating residual offsets between radar and DEM')
        for x in pairsProcessTmp:
            if stack.dateReferenceStack in x.split('-'):
                pairToUse = x
                break
        track = '{}.track.xml'.format(stack.dateReferenceStack)
        wbd = os.path.join('insar', '{}_{}rlks_{}alks.wbd'.format(stack.dateReferenceStack, insar.numberRangeLooks1, insar.numberAzimuthLooks1))
        hgt = os.path.join('insar', '{}_{}rlks_{}alks.hgt'.format(stack.dateReferenceStack, insar.numberRangeLooks1, insar.numberAzimuthLooks1))
        amp = os.path.join('../../', stack.pairsProcessingDir, pairToUse, 'insar', '{}_{}rlks_{}alks.amp'.format(pairToUse, insar.numberRangeLooks1, insar.numberAzimuthLooks1))

        #cmd += header('estimate residual offsets between radar and DEM')
        cmd += 'cd {}\n'.format(os.path.join(stack.datesResampledDir, stack.dateReferenceStack))
        cmd += os.path.join(stackScriptPath, 'radar_dem_offset.py') + ' -track {} -dem {} -wbd {} -hgt {} -amp {} -output affine_transform.txt -nrlks1 {} -nalks1 {}'.format(track, stack.dem, wbd, hgt, amp, insar.numberRangeLooks1, insar.numberAzimuthLooks1)
        if insar.numberRangeLooksSim is not None:
            cmd += '-nrlks_sim {}'.format(insar.numberRangeLooksSim)
        if insar.numberAzimuthLooksSim is not None:
            cmd += '-nalks_sim {}'.format(insar.numberAzimuthLooksSim)
        cmd += '\n'
        cmd += 'cd ../../\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #rectify range offsets
    step_name = 'rectify range offsets'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='dates2', loop=True)
    if datesProcessSecondary != []:
        #cmd += header('rectify range offsets')
        #cmd += parallelSettings('dates2')
        cmd += '''{functionName} () {{
  date=$1

  cd {datesResampledDir}
  cd ${{date}}
  cd insar
  {script} -aff {aff} -input ${{date}}_{nrlks1}rlks_{nalks1}alks_rg.off -output ${{date}}_{nrlks1}rlks_{nalks1}alks_rg_rect.off -nrlks1 {nrlks1} -nalks1 {nalks1}
  cd ../../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'rect_range_offset.py'),
               datesResampledDir   = stack.datesResampledDir,
               aff                 = os.path.join('../../', stack.dateReferenceStack, 'insar', 'affine_transform.txt'),
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #diff interferograms
    step_name = 'diff interferograms'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='insarpair', loop=True)
    if pairsProcess != []:
        #cmd += header('diff interferograms')
        #cmd += parallelSettings('insarpair')
        cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -idir {idir} -ref_date_stack {ref_date_stack} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1}
  cd ../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'diff_interferogram.py'),
               pairsProcessingDir  = stack.pairsProcessingDir,
               idir                = os.path.join('../../', stack.datesResampledDir),
               ref_date_stack      = stack.dateReferenceStack,
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #look and coherence
    step_name = 'look and coherence'
    cmd = cmd0
    cmd += header(step_name)
    cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='insarpair', loop=True)
    if (pairsProcess != []) or processDateReferenceStack:
        #cmd += header('look and coherence')
        if pairsProcess != []:
            #cmd += parallelSettings('insarpair')
            cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1} -nrlks2 {nrlks2} -nalks2 {nalks2}
  cd ../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'look_coherence.py'),
               pairsProcessingDir  = stack.pairsProcessingDir,
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1,
               nrlks2              = insar.numberRangeLooks2,
               nalks2              = insar.numberAzimuthLooks2)
            cmd += forLoop(functionName, 'array', 'nruns')
            cmd += '\n'
            cmd += '\n'

        if processDateReferenceStack:
            cmd += 'cd {}\n'.format(os.path.join(stack.datesResampledDir, stack.dateReferenceStack))
            cmd += os.path.join(stackScriptPath, 'look_geom.py') + ' -date {} -wbd {} -nrlks1 {} -nalks1 {} -nrlks2 {} -nalks2 {}'.format(stack.dateReferenceStack, insar.wbd, insar.numberRangeLooks1, insar.numberAzimuthLooks1, insar.numberRangeLooks2, insar.numberAzimuthLooks2)
            cmd += '\n'
            cmd += 'cd ../../\n'
            cmd += '\n'
        
    cmd += stepDone()
    cmds.append(cmd)
    step_names.append(step_name)


    #save commands
    cmds_2 = cmds
    step_names_2 = step_names



    cmds = []
    step_names = []


    #for ionospheric correction
    if insar.doIon and (pairsProcessIon != []):
        #start new commands: ionospheric phase estimation
        #################################################################################
        cmd0  = '#!/bin/bash\n\n'
        cmd0 += 'ionpair=({})\n'.format(' '.join(pairsProcessIon))
        cmd0 += 'ionpair1=({})\n'.format(' '.join(pairsProcessIon1))
        cmd0 += 'ionpair2=({})\n'.format(' '.join(pairsProcessIon2))
        cmd0 += 'insarpair=({})\n'.format(' '.join(pairsProcess))
        cmd0 += '\n\n'


        #pair up
        step_name = 'pair up for ionospheric phase estimation'
        cmd = cmd0
        cmd += header(step_name)
        cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
        #cmd += header('pair up for ionospheric phase estimation')
        cmd += os.path.join(stackScriptPath, 'pair_up.py') + ' -idir1 {} -idir2 {} -odir {} -ref_date {} -pairs {}'.format(stack.datesProcessingDir, stack.datesResampledDir, stack.pairsProcessingDirIon, stack.dateReferenceStack, ' '.join(pairsProcessIon))
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)


        #subband interferograms
        step_name = 'subband interferograms'
        cmd = cmd0
        cmd += header(step_name)
        cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='ionpair', loop=True)
        if insar.swathPhaseDiffSnapIon is not None:
            snap = [[1 if y else 0 for y in x] for x in insar.swathPhaseDiffSnapIon]
            snapArgument = ' ' + ' '.join(['-snap {}'.format(' '.join([str(y) for y in x])) for x in snap])
        else:
            snapArgument = ''

        wbd1 = '../../dates_resampled/{}/insar/{}_{}rlks_{}alks.wbd'.format(stack.dateReferenceStack, stack.dateReferenceStack, insar.numberRangeLooks1, insar.numberAzimuthLooks1)

        #cmd += header('subband interferograms')
        #cmd += parallelSettings('ionpair')
        cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -idir {idir} -ref_date_stack {ref_date_stack} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1} -water_body {wbd}{snapArgument}
  cd ../../
}}
'''.format(functionName            = functionName,
                   script              = os.path.join(stackScriptPath, 'ion_subband.py'),
                   pairsProcessingDir  = stack.pairsProcessingDirIon,
                   idir                = os.path.join('../../', stack.datesResampledDir),
                   ref_date_stack      = stack.dateReferenceStack,
                   nrlks1              = insar.numberRangeLooks1, 
                   nalks1              = insar.numberAzimuthLooks1,
                   wbd                 = wbd1,
                   snapArgument        = snapArgument)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)


        #unwrap subband interferograms
        step_name = 'unwrap subband interferograms'
        cmd = cmd0
        cmd += header(step_name)
        cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='ionpair', loop=True)
        if insar.filterSubbandInt:
            filtArgument = ' -filt -alpha {} -win {} -step {}'.format(insar.filterStrengthSubbandInt, insar.filterWinsizeSubbandInt, insar.filterStepsizeSubbandInt)
            if not insar.removeMagnitudeBeforeFilteringSubbandInt:
                filtArgument += ' -keep_mag'
        else:
            filtArgument = ''

        #cmd += header('unwrap subband interferograms')
        #cmd += parallelSettings('ionpair')
        cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -idir {idir} -ref_date_stack {ref_date_stack} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -wbd {wbd} -nrlks1 {nrlks1} -nalks1 {nalks1} -nrlks_ion {nrlks_ion} -nalks_ion {nalks_ion}{filtArgument}
  cd ../../
}}
'''.format(functionName            = functionName,
                   script              = os.path.join(stackScriptPath, 'ion_unwrap.py'),
                   pairsProcessingDir  = stack.pairsProcessingDirIon,
                   idir                = os.path.join('../../', stack.datesResampledDir),
                   ref_date_stack      = stack.dateReferenceStack,
                   wbd                 = insar.wbd,
                   nrlks1              = insar.numberRangeLooks1, 
                   nalks1              = insar.numberAzimuthLooks1,
                   nrlks_ion           = insar.numberRangeLooksIon,
                   nalks_ion           = insar.numberAzimuthLooksIon,
                   filtArgument        = filtArgument)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)


        #filter ionosphere
        step_name = 'filter ionosphere'
        cmd = cmd0
        cmd += header(step_name)
        cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='ionpair', loop=True)
        filtArgument = ''
        if insar.fitIon:
            filtArgument += ' -fit'
        if insar.filtIon:
            filtArgument += ' -filt'
        if insar.fitAdaptiveIon:
            filtArgument += ' -fit_adaptive'
        if insar.rmOutliersIon:
            filtArgument += ' -rm_outliers'
        if insar.filtSecondaryIon:
            filtArgument += ' -filt_secondary -win_secondary {}'.format(insar.filteringWinsizeSecondaryIon)
        if insar.filterStdIon is not None:
            filtArgument += ' -filter_std_ion {}'.format(insar.filterStdIon)

        if insar.maskedAreasIon is not None:
            filtArgument += ''.join([' -masked_areas '+' '.join([str(y) for y in x]) for x in insar.maskedAreasIon])

        #cmd += header('filter ionosphere')
        #cmd += parallelSettings('ionpair')
        cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -idir {idir1} -idir2 {idir2} -ref_date_stack {ref_date_stack} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1} -nrlks2 {nrlks2} -nalks2 {nalks2} -nrlks_ion {nrlks_ion} -nalks_ion {nalks_ion} -win_min {win_min} -win_max {win_max} -fit_adaptive_order {fit_adaptive_order}{filtArgument}
  cd ../../
}}
'''.format(functionName            = functionName,
                   script              = os.path.join(stackScriptPath, 'ion_filt.py'),
                   pairsProcessingDir  = stack.pairsProcessingDirIon,
                   idir1               = os.path.join('../../', stack.datesResampledDir),
                   idir2               = os.path.join('../../', stack.datesProcessingDir),
                   ref_date_stack      = stack.dateReferenceStack,
                   nrlks1              = insar.numberRangeLooks1, 
                   nalks1              = insar.numberAzimuthLooks1,
                   nrlks2              = insar.numberRangeLooks2,
                   nalks2              = insar.numberAzimuthLooks2,
                   nrlks_ion           = insar.numberRangeLooksIon,
                   nalks_ion           = insar.numberAzimuthLooksIon,
                   win_min             = insar.filteringWinsizeMinIon,
                   win_max             = insar.filteringWinsizeMaxIon,
                   fit_adaptive_order  = insar.fitAdaptiveOrderIon,
                   filtArgument        = filtArgument)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)


        #prepare interferograms for checking ionospheric correction
        step_name = 'prepare interferograms for checking ionosphere estimation results'
        cmd = cmd0
        cmd += header(step_name)
        #cmd += header('prepare interferograms for checking ionosphere estimation results')
        if pairsProcessIon1 != []:
            cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='ionpair1', loop=True)
            #cmd += parallelSettings('ionpair1')
            if (insar.numberRangeLooksIon != 1) or (insar.numberAzimuthLooksIon != 1):
                cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  {script} -i {pairsProcessingDir}/${{pair}}/insar/diff_${{pair}}_{nrlks1}rlks_{nalks1}alks.int -o {pairsProcessingDirIon}/${{pair}}/ion/ion_cal/diff_${{pair}}_{nrlks}rlks_{nalks}alks_ori.int -r {nrlks_ion} -a {nalks_ion}
}}
'''.format(functionName            = functionName,
               script              = os.path.join('', 'looks.py'),
               pairsProcessingDir  = stack.pairsProcessingDir.strip('/'),
               pairsProcessingDirIon  = stack.pairsProcessingDirIon.strip('/'),
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1,
               nrlks_ion           = insar.numberRangeLooksIon,
               nalks_ion           = insar.numberAzimuthLooksIon,
               nrlks               = insar.numberRangeLooks1 * insar.numberRangeLooksIon, 
               nalks               = insar.numberAzimuthLooks1 * insar.numberAzimuthLooksIon)
                cmd += forLoop(functionName, 'array', 'nruns')
                cmd += '\n'
                cmd += '\n'
                cmd += '\n'
            else:
                cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cp {pairsProcessingDir}/${{pair}}/insar/diff_${{pair}}_{nrlks1}rlks_{nalks1}alks.int* {pairsProcessingDirIon}/${{pair}}/ion/ion_cal
}}
'''.format(functionName            = functionName,
               pairsProcessingDir  = stack.pairsProcessingDir.strip('/'),
               pairsProcessingDirIon  = stack.pairsProcessingDirIon.strip('/'),
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1)
                cmd += forLoop(functionName, 'array', 'nruns')
                cmd += '\n'
                cmd += '\n'
                cmd += '\n'


        if pairsProcessIon2 != []:
            cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='ionpair2', loop=True)
            #cmd += parallelSettings('ionpair2')
            cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1}
  cd ../../

done'''.format(extraCommands       = parallelCommands,
               script              = os.path.join(stackScriptPath, 'form_interferogram.py'),
               pairsProcessingDir  = stack.pairsProcessingDirIon,
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1)
            cmd += forLoop(functionName, 'array', 'nruns')
            cmd += '\n'
            cmd += '\n'

            wbd1 = '../../dates_resampled/{}/insar/{}_{}rlks_{}alks.wbd'.format(stack.dateReferenceStack, stack.dateReferenceStack, insar.numberRangeLooks1, insar.numberAzimuthLooks1)
            cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -ref_date_stack {ref_date_stack} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1} -water_body {wbd}
  cd ../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'mosaic_interferogram.py'),
               pairsProcessingDir  = stack.pairsProcessingDirIon,
               ref_date_stack      = stack.dateReferenceStack,
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1,
               wbd                 = wbd1)
            cmd += forLoop(functionName, 'array', 'nruns')
            cmd += '\n'
            cmd += '\n'

            cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -idir {idir} -ref_date_stack {ref_date_stack} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1}
  cd ../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'diff_interferogram.py'),
               pairsProcessingDir  = stack.pairsProcessingDirIon,
               idir                = os.path.join('../../', stack.datesResampledDir),
               ref_date_stack      = stack.dateReferenceStack,
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1)
            cmd += forLoop(functionName, 'array', 'nruns')
            cmd += '\n'
            cmd += '\n'

            if (insar.numberRangeLooksIon != 1) or (insar.numberAzimuthLooksIon != 1):
                cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  {script} -i {pairsProcessingDir}/${{pair}}/insar/diff_${{pair}}_{nrlks1}rlks_{nalks1}alks.int -o {pairsProcessingDir}/${{pair}}/ion/ion_cal/diff_${{pair}}_{nrlks}rlks_{nalks}alks_ori.int -r {nrlks_ion} -a {nalks_ion}
}}
'''.format(functionName            = functionName,
               script              = os.path.join('', 'looks.py'),
               pairsProcessingDir  = stack.pairsProcessingDirIon.strip('/'),
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1,
               nrlks_ion           = insar.numberRangeLooksIon,
               nalks_ion           = insar.numberAzimuthLooksIon,
               nrlks               = insar.numberRangeLooks1 * insar.numberRangeLooksIon, 
               nalks               = insar.numberAzimuthLooks1 * insar.numberAzimuthLooksIon)
                cmd += forLoop(functionName, 'array', 'nruns')
                cmd += '\n'
                cmd += '\n'
                cmd += '\n'
            else:
                cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cp {pairsProcessingDir}/${{pair}}/insar/diff_${{pair}}_{nrlks1}rlks_{nalks1}alks.int* {pairsProcessingDir}/${{pair}}/ion/ion_cal
}}
'''.format(functionName            = functionName,
               pairsProcessingDir  = stack.pairsProcessingDirIon.strip('/'),
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1)
                cmd += forLoop(functionName, 'array', 'nruns')
                cmd += '\n'
                cmd += '\n'
                cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)


        #check ionosphere estimation results
        step_name = 'check ionosphere estimation results'
        cmd = cmd0
        cmd += header(step_name)
        cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='ionpair', loop=True)
        #cmd += header('check ionosphere estimation results')
        #cmd += parallelSettings('ionpair')
        cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -e='a*exp(-1.0*J*b)' --a=ion/ion_cal/diff_${{pair}}_{nrlks}rlks_{nalks}alks_ori.int --b=ion/ion_cal/filt_ion_{nrlks}rlks_{nalks}alks.ion -s BIP -t cfloat -o ion/ion_cal/diff_${{pair}}_{nrlks}rlks_{nalks}alks.int
  cd ../../
}}
'''.format(functionName            = functionName,
                   script              = os.path.join('', 'imageMath.py'),
                   pairsProcessingDir  = stack.pairsProcessingDirIon,
                   nrlks               = insar.numberRangeLooks1*insar.numberRangeLooksIon, 
                   nalks               = insar.numberAzimuthLooks1*insar.numberAzimuthLooksIon)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'

        cmd += os.path.join(stackScriptPath, 'ion_check.py') + ' -idir {} -odir fig_ion -pairs {}'.format(stack.pairsProcessingDirIon, ' '.join(pairsProcessIon))
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)


        #estimate ionospheric phase for each date
        step_name = 'estimate ionospheric phase for each date'
        cmd = cmd0
        cmd += header(step_name)
        cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
        #cmd += header('estimate ionospheric phase for each date')
        cmd += "#check the ionospheric phase estimation results in folder 'fig_ion', and find out the bad pairs.\n"
        cmd += '#these pairs should be excluded from this step by specifying parameter -exc_pair. For example:\n'
        cmd += '#-exc_pair 150401-150624 150401-150722\n\n'
        cmd += '#MUST re-run all the following commands, each time after running this command!!!\n'
        cmd += '#uncomment to run this command\n'
        cmd += '#'
        cmd += os.path.join(stackScriptPath, 'ion_ls.py') + ' -idir {} -odir {} -ref_date_stack {} -nrlks1 {} -nalks1 {} -nrlks2 {} -nalks2 {} -nrlks_ion {} -nalks_ion {} -interp'.format(stack.pairsProcessingDirIon, stack.datesDirIon, stack.dateReferenceStack, insar.numberRangeLooks1, insar.numberAzimuthLooks1, insar.numberRangeLooks2, insar.numberAzimuthLooks2, insar.numberRangeLooksIon, insar.numberAzimuthLooksIon)
        if stack.dateReferenceStackIon is not None:
            cmd += ' -zro_date {}'.format(stack.dateReferenceStackIon)
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)


        #correct ionosphere
        step_name = 'correct ionosphere'
        cmd = cmd0
        cmd += header(step_name)
        cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='', loop=False)
        if insar.applyIon:
            #cmd += header('correct ionosphere')
            cmd += '#no need to run parallelly for this for loop, it is fast!!!\n'
            cmd += '''#redefine insarpair to include all processed InSAR pairs
insarpair=($(ls -l {pairsProcessingDir} | grep ^d | awk '{{print $9}}'))
for ((i=0;i<${{#insarpair[@]}};i++)); do

  IFS='-' read -ra dates <<< "${{insarpair[i]}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{insarpair[i]}}
  #uncomment to run this command
  #{script} -ion_dir {ion_dir} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1} -nrlks2 {nrlks2} -nalks2 {nalks2}
  cd ../../

done'''.format(script              = os.path.join(stackScriptPath, 'ion_correct.py'),
               pairsProcessingDir  = stack.pairsProcessingDir,
               ion_dir             = os.path.join('../../', stack.datesDirIon),
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1,
               nrlks2              = insar.numberRangeLooks2,
               nalks2              = insar.numberAzimuthLooks2)
            cmd += '\n'
            cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)
    else:
        cmd  = '#!/bin/bash\n\n'
        cmd += '#no pairs for estimating ionosphere.'

        #do not create steps if not doing ionospheric correction

    #save commands
    cmds_3 = cmds
    step_names_3 = step_names



    cmds = []
    step_names = []


    #if pairsProcess != []:
    if True:
        #start new commands: processing each pair after ionosphere correction
        #################################################################################
        cmd0  = '#!/bin/bash\n\n'
        if insar.doIon and insar.applyIon:
            #reprocess all pairs
            cmd0 += '''insarpair=($(ls -l {pairsProcessingDir} | grep ^d | awk '{{print $9}}'))'''.format(pairsProcessingDir = stack.pairsProcessingDir)
            cmd0 += '\n'
        else:
            cmd0 += 'insarpair=({})\n'.format(' '.join(pairsProcess))
        cmd0 += '\n\n'


        #filter interferograms
        step_name = 'filter interferograms'
        cmd = cmd0
        cmd += header(step_name)
        cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='insarpair', loop=True)
        extraArguments = ''
        if not insar.removeMagnitudeBeforeFiltering:
            extraArguments += ' -keep_mag'
        if insar.waterBodyMaskStartingStep == 'filt':
            extraArguments += ' -wbd_msk'

        #cmd += header('filter interferograms')
        #cmd += parallelSettings('insarpair')
        cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -idir {idir} -ref_date_stack {ref_date_stack} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1} -nrlks2 {nrlks2} -nalks2 {nalks2} -alpha {alpha} -win {win} -step {step}{extraArguments}
  cd ../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'filt.py'),
               pairsProcessingDir  = stack.pairsProcessingDir,
               idir                = os.path.join('../../', stack.datesResampledDir),
               ref_date_stack      = stack.dateReferenceStack,
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1,
               nrlks2              = insar.numberRangeLooks2,
               nalks2              = insar.numberAzimuthLooks2,
               alpha               = insar.filterStrength,
               win                 = insar.filterWinsize,
               step                = insar.filterStepsize,
               extraArguments      = extraArguments)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)


        #unwrap interferograms
        step_name = 'unwrap interferograms'
        cmd = cmd0
        cmd += header(step_name)
        cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='insarpair', loop=True)
        extraArguments = ''
        if insar.waterBodyMaskStartingStep == 'unwrap':
            extraArguments += ' -wbd_msk'

        #cmd += header('unwrap interferograms')
        #cmd += parallelSettings('insarpair')
        cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  {script} -idir {idir} -ref_date_stack {ref_date_stack} -ref_date ${{ref_date}} -sec_date ${{sec_date}} -nrlks1 {nrlks1} -nalks1 {nalks1} -nrlks2 {nrlks2} -nalks2 {nalks2}{extraArguments}
  cd ../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'unwrap_snaphu.py'),
               pairsProcessingDir  = stack.pairsProcessingDir,
               idir                = os.path.join('../../', stack.datesResampledDir),
               ref_date_stack      = stack.dateReferenceStack,
               nrlks1              = insar.numberRangeLooks1, 
               nalks1              = insar.numberAzimuthLooks1,
               nrlks2              = insar.numberRangeLooks2,
               nalks2              = insar.numberAzimuthLooks2,
               extraArguments      = extraArguments)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'
        cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)


        #geocode
        step_name = 'geocode'
        cmd = cmd0
        cmd += header(step_name)
        cmd += stepSettings(nthreads=stack.numberOfParallelThreads, nruns=stack.numberOfParallelRuns, array='insarpair', loop=True)
        extraArguments = ''
        if insar.geocodeInterpMethod is not None:
            extraArguments += ' -interp_method {}'.format(insar.geocodeInterpMethod)
        if insar.bbox: # is not None:
            extraArguments += ' -bbox {}'.format('/'.format(insar.bbox))

        #cmd += header('geocode')
        #cmd += parallelSettings('insarpair')
        cmd += '''{functionName} () {{
  pair=$1

  IFS='-' read -ra dates <<< "${{pair}}"
  ref_date=${{dates[0]}}
  sec_date=${{dates[1]}}

  cd {pairsProcessingDir}
  cd ${{pair}}
  cd insar
  {script} -ref_date_stack_track ../{ref_date_stack}.track.xml -dem {dem_geo} -input ${{pair}}_{nrlks}rlks_{nalks}alks.cor -nrlks {nrlks} -nalks {nalks}{extraArguments}
  {script} -ref_date_stack_track ../{ref_date_stack}.track.xml -dem {dem_geo} -input filt_${{pair}}_{nrlks}rlks_{nalks}alks.unw -nrlks {nrlks} -nalks {nalks}{extraArguments}
  {script} -ref_date_stack_track ../{ref_date_stack}.track.xml -dem {dem_geo} -input filt_${{pair}}_{nrlks}rlks_{nalks}alks_msk.unw -nrlks {nrlks} -nalks {nalks}{extraArguments}
  cd ../../../
}}
'''.format(functionName            = functionName,
               script              = os.path.join(stackScriptPath, 'geocode.py'),
               pairsProcessingDir  = stack.pairsProcessingDir,
               ref_date_stack      = stack.dateReferenceStack,
               dem_geo             = stack.demGeo,
               nrlks               = insar.numberRangeLooks1*insar.numberRangeLooks2, 
               nalks               = insar.numberAzimuthLooks1*insar.numberAzimuthLooks2,
               extraArguments      = extraArguments)
        cmd += forLoop(functionName, 'array', 'nruns')
        cmd += '\n'
        cmd += '\n'

        cmd += 'cd {}\n'.format(os.path.join(stack.datesResampledDir, stack.dateReferenceStack, 'insar'))
        cmd += os.path.join(stackScriptPath, 'geocode.py') + ' -ref_date_stack_track ../{ref_date_stack}.track.xml -dem {dem_geo} -input {ref_date_stack}_{nrlks}rlks_{nalks}alks.los -nrlks {nrlks} -nalks {nalks}{extraArguments}'.format(
                   ref_date_stack      = stack.dateReferenceStack,
                   dem_geo             = stack.demGeo,
                   nrlks               = insar.numberRangeLooks1*insar.numberRangeLooks2, 
                   nalks               = insar.numberAzimuthLooks1*insar.numberAzimuthLooks2,
                   extraArguments      = extraArguments)
        cmd += '\n'
        cmd += 'cd ../../../\n'
        cmd += '\n'
        cmd += stepDone()
        cmds.append(cmd)
        step_names.append(step_name)
    else:
        cmd  = '#!/bin/bash\n\n'
        cmd += '#no pairs for InSAR processing.'

        #do not create steps if no pairs for InSAR processing.


    #save commands
    cmds_4 = cmds
    step_names_4 = step_names


    return (cmds_1, cmds_2, cmds_3, cmds_4, step_names_1, step_names_2, step_names_3, step_names_4)


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='create commands to process a stack of acquisitions')
    parser.add_argument('-stack_par', dest='stack_par', type=str, required=True,
            help = 'stack processing input parameter file.')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    stackParameter = inps.stack_par


    #need to remove -stack_par from arguments, otherwise application class would complain
    import sys
    #sys.argv.remove(sys.argv[1])
    #sys.argv = [sys.argv[2]]
    sys.argv = [sys.argv[0], sys.argv[2]]

    stack = loadStackUserParameters(stackParameter)
    insar = stack
    print()


    #0. parameters that must be set.
    if stack.dataDir is None:
        raise Exception('data directory not set.')
    checkDem(stack.dem)
    checkDem(stack.demGeo)
    checkDem(stack.wbd)
    if stack.dateReferenceStack is None:
        raise Exception('reference date of the stack not set.')


    #1. check if date dirctories are OK
    checkStackDataDir(stack.dataDir)


    #2. regular InSAR processing
    print('get dates and pairs from user input')
    pairsProcess = formPairs(stack.dataDir, stack.numberOfSubsequentDates, 
        stack.pairTimeSpanMinimum, stack.pairTimeSpanMaximum, 
        stack.datesIncluded, stack.pairsIncluded, 
        stack.datesExcluded, stack.pairsExcluded)
    datesProcess = datesFromPairs(pairsProcess)
    print('InSAR processing:')
    print(f"dates ({len(datesProcess)}): {' '.join(datesProcess)}")
    print(f"pairs ({len(pairsProcess)}): {' '.join(pairsProcess)}")

    rank = stackRank(datesProcess, pairsProcess)
    if rank != len(datesProcess) - 1:
        print('\nWARNING: dates in stack not fully connected by pairs to be processed in regular InSAR processing\n')
    print()


    #3. ionospheric correction
    if insar.doIon:
        pairsProcessIon = formPairs(stack.dataDir, stack.numberOfSubsequentDatesIon, 
            stack.pairTimeSpanMinimumIon, stack.pairTimeSpanMaximumIon, 
            stack.datesIncludedIon, stack.pairsIncludedIon, 
            stack.datesExcludedIon, stack.pairsExcludedIon)
        datesProcessIon = datesFromPairs(pairsProcessIon)
        print('ionospheric phase estimation:')
        print(f"dates ({len(datesProcessIon)}): {' '.join(datesProcessIon)}")
        print(f"pairs ({len(pairsProcessIon)}): {' '.join(pairsProcessIon)}")

        rankIon = stackRank(datesProcessIon, pairsProcessIon)
        if rankIon != len(datesProcessIon) - 1:
            print('\nWARNING: dates in stack not fully connected by pairs to be processed in ionospheric correction\n')
        print('\n')
    else:
        pairsProcessIon = []


    #4. union
    if insar.doIon:
        datesProcess = unionLists(datesProcess, datesProcessIon)
    else:
        datesProcess = datesProcess


    #5. find acquisition mode
    mode = os.path.basename(sorted(glob.glob(os.path.join(stack.dataDir, datesProcess[0], 'LED-ALOS2*-*-*')))[0]).split('-')[-1][0:3]
    print('acquisition mode of stack: {}'.format(mode))
    print('\n')


    #6. check if already processed previously
    datesProcessedAlready = getFolders(stack.datesResampledDir)
    if not stack.datesReprocess:
        datesProcess, datesProcessRemoved = removeCommonItemsLists(datesProcess, datesProcessedAlready)
        if datesProcessRemoved != []:
            print('the following dates have already been processed, will not reprocess them.')
            print(f"dates ({len(datesProcessRemoved)}): {' '.join(datesProcessRemoved)}")
            print()

    pairsProcessedAlready = getFolders(stack.pairsProcessingDir)
    if not stack.pairsReprocess:
        pairsProcess, pairsProcessRemoved = removeCommonItemsLists(pairsProcess, pairsProcessedAlready)
        if pairsProcessRemoved != []:
            print('the following pairs for InSAR processing have already been processed, will not reprocess them.')
            print(f"pairs ({len(pairsProcessRemoved)}): {' '.join(pairsProcessRemoved)}")
            print()

    if insar.doIon:
        pairsProcessedAlreadyIon = getFolders(stack.pairsProcessingDirIon)
        if not stack.pairsReprocessIon:
            pairsProcessIon, pairsProcessRemovedIon = removeCommonItemsLists(pairsProcessIon, pairsProcessedAlreadyIon)
            if pairsProcessRemovedIon != []:
                print('the following pairs for estimating ionospheric phase have already been processed, will not reprocess them.')
                print(f"pairs ({len(pairsProcessRemovedIon)}): {' '.join(pairsProcessRemovedIon)}")
                print()

    print()

    print('dates and pairs to be processed:')
    print(f"dates ({len(datesProcess)}): {' '.join(datesProcess)}")
    print(f"pairs ({len(pairsProcess)}; for InSAR processing): {' '.join(pairsProcess)}")
    if insar.doIon:
        print(f"pairs ({len(pairsProcessIon)}; for estimating ionospheric phase): {' '.join(pairsProcessIon)}")
    print('\n')


    #7. use mode to define processing parameters
    #number of looks
    from isceobj.Alos2Proc.Alos2ProcPublic import modeProcParDict
    if insar.numberRangeLooks1 is None:
        insar.numberRangeLooks1 = modeProcParDict['ALOS-2'][mode]['numberRangeLooks1']
    if insar.numberAzimuthLooks1 is None:
        insar.numberAzimuthLooks1 = modeProcParDict['ALOS-2'][mode]['numberAzimuthLooks1']
    if insar.numberRangeLooks2 is None:
        insar.numberRangeLooks2 = modeProcParDict['ALOS-2'][mode]['numberRangeLooks2']
    if insar.numberAzimuthLooks2 is None:
        insar.numberAzimuthLooks2 = modeProcParDict['ALOS-2'][mode]['numberAzimuthLooks2']
    if insar.numberRangeLooksIon is None:
        insar.numberRangeLooksIon = modeProcParDict['ALOS-2'][mode]['numberRangeLooksIon']
    if insar.numberAzimuthLooksIon is None:
        insar.numberAzimuthLooksIon = modeProcParDict['ALOS-2'][mode]['numberAzimuthLooksIon']


    #7. create commands
    if (datesProcess == []) and (pairsProcess == []) and (pairsProcessIon == []):
        print('no dates and pairs need to be processed.')
        print('no processing script is generated.')
    else:
        # cmd1, cmd2, cmd3, cmd4 = createCmds(stack, datesProcess, pairsProcess, pairsProcessIon, mode)
        # with open('cmd_1.sh', 'w') as f:
        #     f.write(cmd1)
        # with open('cmd_2.sh', 'w') as f:
        #     f.write(cmd2)
        # with open('cmd_3.sh', 'w') as f:
        #     f.write(cmd3)
        # with open('cmd_4.sh', 'w') as f:
        #     f.write(cmd4)
        (cmds_1, cmds_2, cmds_3, cmds_4, step_names_1, step_names_2, step_names_3, step_names_4) = createCmds(stack, datesProcess, pairsProcess, pairsProcessIon, mode)

        cmdDir = 'cmd'
        os.makedirs(cmdDir, exist_ok=True)
        os.chdir(cmdDir)
        for i, (cmds_i, step_names_i) in enumerate(zip((cmds_1, cmds_2, cmds_3, cmds_4), (step_names_1, step_names_2, step_names_3, step_names_4))):
            filename1 = 'cmd_{}.sh'.format(i+1)
            cmd = ''
            for j, (cmds_j, step_names_j) in enumerate(zip(cmds_i, step_names_i)):
                filename2 = 'cmd_{}_{}_{}.sh'.format(i+1, j+1, step_names_j.replace(' ', '_'))
                with open(filename2, 'w') as f:
                    f.write(cmds_j)
                cmd += '{}/{}\n'.format(cmdDir, filename2)
            cmd +='''
echo "$(basename "$0") done"
'''
            with open(filename1, 'w') as f:
                f.write(cmd)
        runCmd('chmod +x cmd_*.sh', silent=1)
        os.chdir('../')
