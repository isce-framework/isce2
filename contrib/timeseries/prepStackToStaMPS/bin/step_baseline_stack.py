#!/usr/bin/env python3                                                     

# Script that computes the baselines for a given master based on a stack of baselines
#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2017 California Institute of Technology. ALL RIGHTS RESERVED.
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




import argparse
from glob import glob
import numpy as np
import scipy
import os

# command line parsing of input file
def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(description='baseline re-estimation for a master date')
    parser.add_argument('-i','--input', dest='baseline_dir', type=str,required=True, help='Path to the baseline directory')
    parser.add_argument('-m', '--master_date' ,dest='new_master', type=str, required=True , help='New master date for stack')
    return parser.parse_args() 
  



def baselinegrid(inps):
    """ 
        Basline files are given as grids
    """
    import gdal 

    # parsing the command line inputs
    baseline_dir = inps.baseline_dir
    new_master = int(inps.new_master)

    # check if the baseline grids are all in the same folder or if they are date YYYYMMDD_YYYYMMDD folders.
    baseline_files = glob(os.path.join(baseline_dir,"2*","2*[0-9].vrt"))
    if not baseline_files:
        # try to see if they are all local
        baseline_files = glob(os.path.join(baseline_dir,"2*[0-9].vrt"))
    if not baseline_files:
        # need to raize error as no baseline files where found
        raise ValueError('No Baseline files were found')
                          
    # finding the master baseline grid file
    master_file = False
    for baseline_file in baseline_files:
        date = os.path.basename(baseline_file)
        date = date.split('.vrt')
        date = int(date[0])
        if date == new_master:
            master_file = os.path.join(os.path.dirname(baseline_file),str(date))
    if not master_file:
        raise Exception('Could not find the master baseline grid')

    # generate new baseline grid for each slave and also store the average for overview
    baselines_new = [float(0)]
    dates_new = [new_master]
    for baseline_file in baseline_files:
        date = os.path.basename(baseline_file)
        date = date.split('.vrt')
        date = int(date[0])
        # check if this is a slave date
        if not date == new_master:
            slave_file = os.path.join(os.path.dirname(baseline_file),str(date))
            local_baseline_file = "baselineGRID_" + str(date)
            cmd = "imageMath.py -e='a-b' --a=" + slave_file + " --b=" + master_file + " -o " + local_baseline_file + " -t FLOAT -s BIP"
            os.system(cmd)
            # generating a vrt file as well
            cmd = "isce2gis.py vrt -i " + local_baseline_file
            os.system(cmd)
     
            # compute the average as well for baseline overview
            if os.path.isfile(local_baseline_file):
                dataset_avg = gdal.Open(local_baseline_file + '.vrt',gdal.GA_ReadOnly)
                stats = dataset_avg.GetRasterBand(1).GetStatistics(0,1)
                average = stats[2]
                baselines_new.append(average)
                dates_new.append(date)

    # convert to numpy arrays
    baselines_new = np.reshape(np.array(baselines_new),(-1,1))
    dates_new = np.reshape(np.array(dates_new),(-1,1))
    temp =np.hstack([dates_new,baselines_new])
    temp = temp[temp[:, 0].argsort()]
    np.savetxt('baseline_overview_new', temp, fmt='%.f %.2f ', delimiter='\t', newline='\n')

    # generate a baseline file for each acquisition
    for counter in range(temp.shape[0]):
        if temp[counter,0] == new_master:
            dir_name = 'master'
        else:
            dir_name = str(int(temp[counter,0]))
        # generate the directory if it does not exist yet
        try:
            os.stat(dir_name)
        except:
            os.mkdir(dir_name)
        np.savetxt(os.path.join(dir_name,'baseline'), [temp[counter,:]], fmt='%.f %.2f ', delimiter='\t', newline='\n')


def baselinefile(inps):
    """
        Baseline files are txt files with a single value in them
    """
    # parsing the command line inputs
    baseline_dir = inps.baseline_dir
    new_master = int(inps.new_master)

    # check if the baseline files are all in the same folder or if they are date YYYYMMDD_YYYYMMDD folders. 
    baseline_files = glob(os.path.join(baseline_dir,"2*","2*.txt"))
    if not baseline_files:
        # try to see if they are all local
        baseline_files = glob(os.path.join(baseline_dir,"2*.txt"))

    if not baseline_files:
        # need to raize error as no baseline files where found
        raise ValueError('No Baseline files were found')

    # generate an array of dates
    master = []
    slave = []
    baseline = []
    for baseline_file in baseline_files:
        dates = os.path.basename(baseline_file)
        dates = dates.split('.')[0]
        master.append(int(dates.split('_')[0]))
        slave.append(int(dates.split('_')[1]))

        # read file and either catch a single value or read for specific -perp (average):- string
        file = open(baseline_file,'r')
        file_lines = file.readlines()

        # there is only one line for the baseline
        if len(file_lines)==1:
               baseline.append(float(file_lines[0]))

        # there are multiple lines, only extract the specific string 
        else:
            baseline_temp=[]
            for file_line in file_lines:
                if file_line.find("perp (average):") != -1:
                    # getting the string with the value
                    temp = file_line.split("perp (average):")[1]
                    # removing the newline character
                    temp = temp.split("\n")[0]
                    baseline_temp.append(float(temp))
                # take the mean 
                baseline.append(np.mean(np.array(baseline_temp)))

    # converting to an numpy array
    baseline = np.reshape(np.array(baseline),(-1,1))
    master = np.reshape(np.array(master),(-1,1))
    slave = np.reshape(np.array(slave),(-1,1))

    # count the number of nan in the baseline
    ix_count = np.count_nonzero(np.isnan(baseline))
    if ix_count>0:
        for ix in range(baseline.shape[0]):
            if np.isnan(baseline[ix])==1:
                print(str(master[ix,0]) + "_" + str(slave[ix,0]))
        # now raize error
        raise ValueError('NaN found for baseline...')

    # generating an array of acquisitions
    dates= np.reshape(np.unique(np.concatenate((master,slave), axis=0)),(-1,1))
    # getting number of baseline combinations and the number of acquisitions
    n_acquistions = dates.shape[0]
    n_combinations = baseline.shape[0]


    #### mapping the baselines to a new master
    # generate the design matrix that maps the baselines to dates
    A = np.zeros((n_combinations,n_acquistions))
    for counter in range(n_combinations):
        pos_master, temp = np.where(dates == master[counter])
        pos_slave, temp = np.where(dates == slave[counter])
        A[counter,pos_master]=-1
        A[counter,pos_slave]=1
        del pos_slave
        del pos_master

    # location of the requested master
    pos_master, temp = np.where(dates == new_master)

    # remove the new master from the design matrix and acquisitions
    A[:,pos_master[0]]=0
    # compute the new baselines
    baselines_new = np.linalg.lstsq(A, baseline)[0]

#   # add the new master back in and write out the file
#   baselines_new = np.concatenate((baselines_new, np.zeros((1,1))))
 #  dates = np.concatenate((dates, [[new_master]]))
    # concatenate together to write single matrix
    temp= np.concatenate((dates, baselines_new), axis=1)
    np.savetxt('baseline_overview', temp, fmt='%.f %.2f ', delimiter='\t', newline='\n')

    # generate a baseline file for each acquisition
    for counter in range(n_acquistions):
        if temp[counter,0] == new_master:
           dir_name = 'master'
        else:
           dir_name = str(int(temp[counter,0]))
        # generate the directory if it does not exist yet
        try:
            os.stat(dir_name)
        except:
            os.mkdir(dir_name)
        np.savetxt(os.path.join(dir_name,'baseline'), [temp[counter,:]], fmt='%.f %.2f ', delimiter='\t', newline='\n')



# main script
if __name__ == '__main__':
    '''
    Main driver.                
    '''
    
    # parsing the command line inputs
    inps = cmdLineParse()

    ### check if the baselines are as single txt file or as a grid
    # baseline files 
    baseline_files = glob(os.path.join(inps.baseline_dir,"2*","2*.txt"))
    if not baseline_files:
        # try to see if they are all local
        baseline_files = glob(os.path.join(inps.baseline_dir,"2*.txt"))

    # baseline grid
    baseline_grids = glob(os.path.join(inps.baseline_dir,"2*","2*[0-9].vrt"))
    if not baseline_grids:
        # try to see if they are all local
        baseline_grids = glob(os.path.join(inps.baseline_dir,"2*[0-9].vrt"))

    ### let the grid take priority
    if baseline_grids:
        baselinegrid(inps)
    elif baseline_files:
        baselinefile(inps)

