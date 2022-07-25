#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:07:14 2022

@author: Marin Govorcin
"""

import os
import sys
import glob
import argparse
from datetime import datetime
import numpy as np
from osgeo import gdal

import isce
from Stack import run, sentinelSLC

HELPSTR= '''

Interferogram processor for Sentinel-1 data using ISCE software.

For a full list of different options, try interferogramStack.py -h

interferogramStack.py generates all configuration and run files required to be executed for genetration of Sentinel-1 interferograms.
                        the script allows use to generate the network of pairs you wnat to process. 
Some features are:
    - Network selection: single_reference (PS) -  use single_reference to define reference date (if not defined, it is selected automatically)
                        sequential (DS) - use with number of connection (num_connections)
                        delaunay (DS) - generate delaunay network
                        full  - generate all possible pairs
                        
    - Periodic pairs: add peridic pairs to the network, such as semi-annual or annual pairs
    - Stop and End Date : process a part of coregistrated stack or to add additional pairs (mini network) between certain dates

Following are required to start processing:

1) Co-registrated stack; run:
    stackSentinel.py -s ../SLC/ -d ../../MexicoCity/demLat_N18_N20_Lon_W100_W097.dem.wgs84 -b '19 20 -99.5 -98.5' -a ../../AuxDir/ -o ../../Orbits -C NESD  -W slc

Similar as stackSentinel.py, interferogramStack.py does not process any data, it  only prepares a lot of input files for processing that need to be run afterwards
Run_file can be find in 'run_ifg_config' folder

Note also that run files need to be executed in order, i.e., running run_03 needs results from run_02, etc.

##############################################

#Examples:
   
Sequential network with 4 connections between each date and subsequent dates and 1 yr pairs

    interferogramStack.py -s /path_to_stack -n sequential -c 4 -p 365 

Sequential network with 4 connections between each date and subsequent dates and 6-months and 1 yr pairs

    interferogramStack.py -s /path_to_stack -n sequential -c 4 -p 180 365  

Delauney network with 1 yr pairs, with 90m posting interferograms (multilook ratio 27x5 (rg x az))

    interferogramStack.py -s /path_to_stack -n delaunay -p 365 -r 27 -z 5

Add all possible pairs to existing network around certain dates

    interferogramStack.py -s /path_to_stack -n full --start_date 2020-12-12 --end_date 2021-01-12 --force

##############################################
'''

class customArgparseAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        '''
        The action to be performed.
        '''
        print(HELPSTR)
        parser.exit()


def createParser():
    parser = argparse.ArgumentParser( description='Prepare the directory structure and config \
                                    files for interferogram stack processing of Sentinel data')

    parser.add_argument('-H', '--hh', nargs=0, action=customArgparseAction,
                        help='Display detailed help information.')

    parser.add_argument('-s', '--stack_directory', dest='work_dir', type=str, required=True,
                        help='Directory with co-registrated Sentinel1 SLC stack')

    parser.add_argument('-n', '--network', dest='network', type=str, default='sequential',
                        help='Network type of interogram pairs, options: '
                              '[single_reference, sequential, delaunay, full]')

    parser.add_argument('-c', '--num_connections', dest='num_connections', type=str, default = '4',
                        help='connection number of interferograms between each date and subsequent dates'
                             '(default: %(default)s).')

    parser.add_argument('-p', '--periodic_connections', dest='periodic_connections', nargs='+', type=str, default = None,
                        help='number of periodic interferograms for each acquisition, in days (default: %(default)s).'
                        'Example:  90 (for 3 months pairs), 180 (for 6months pairs, semi-annual), 365 (for 1 yr, annual)')

    parser.add_argument('-pt', '--periodic_tolerance', dest='periodic_tolerance', type=str, default = '1',
                        help='tolerance for selection of periodic interferograms in days, plus minus'
                              'around the defined period (default: %(default)s).'
                              'Example:  1 (1 days around 365 period')

    parser.add_argument('-sr', '--single_reference', dest='single_reference_date', type=str, default = None,
                        help='Reference date for Single-Reference network (default: %(default)s).'
                        'format should be YYYY-MM-DD e.g., 2015-01-23')

    parser.add_argument('--start_date', dest='startDate', type=str, default=None,
                        help='Start date for interferogram network generation. Acquisitions before start date are ignored. '
                             'format should be YYYY-MM-DD e.g., 2015-01-23')

    parser.add_argument('--end_date', dest='endDate', type=str, default=None,
                        help='End date for interferogram network generation. Acquisitions after stop date are ignored. '
                             'format should be YYYY-MM-DD e.g., 2017-02-26')

    parser.add_argument('--max_bperp', dest='max_bperp', type=int, default=None,
                        help='Threshold for Maximum Perpendicular baseline, remove all pair above it [in meters] (default: %(default)s)')

    parser.add_argument('--max_btemp', dest='max_btemp', type=int, default=None,
                         help='Threshold for Maximum Temporal baseline, remove all pair above it [in days] (default: %(default)s)')

    parser.add_argument('-z', '--azimuth_looks', dest='azimuthLooks', type=str, default='3',
                        help='Number of looks in azimuth for interferogram multi-looking (default: %(default)s).')

    parser.add_argument('-r', '--range_looks', dest='rangeLooks', type=str, default='9',
                        help='Number of looks in range for interferogram multi-looking (default: %(default)s).')

    parser.add_argument('-f', '--filter_strength', dest='filtStrength', type=str, default='0.5',
                        help='Filter strength for interferogram filtering (default: %(default)s).')

    parser.add_argument('-useGPU', '--useGPU', dest='useGPU', action='store_true', default=False,
                        help='Allow App to use GPU when available')

    parser.add_argument('--num_proc', '--num_process', dest='numProcess', type=int, default=1,
                        help='number of tasks running in parallel in each run file (default: %(default)s).')

    parser.add_argument('-u', '--unw_method', dest='unwMethod', type=str, default='snaphu', choices=['icu', 'snaphu'],
                        help='Unwrapping method (default: %(default)s).')

    parser.add_argument('-rmFilter', '--rmFilter', dest='rmFilter', action='store_true', default=False,
                        help='Make an extra unwrap file in which filtering effect is removed')

    parser.add_argument('-t', '--text_cmd', dest='text_cmd', type=str, default='',
                        help="text command to be added to the beginning of each line of the run files (default: '%(default)s'). "
                             "Example : 'source ~/.bash_profile;'")
    parser.add_argument('--force', dest='force_run_files', action='store_true',
                        help="Overwrite run files (default: False. ")

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    inps = parser.parse_args(args=iargs)

    inps.work_dir = os.path.abspath(inps.work_dir)

    # MG: this will create separate run_file folder from stack one
    inps.ifg_process = True

    if any(i in iargs for i in ['--num_proc', '--num_process']) and all(
            i not in iargs for i in ['--num_proc4topo', '--num_process4topo']):
        inps.numProcess4topo = inps.numProcess

    return inps

###############################################################################
####################### Utilities for dealing with dates ######################
###############################################################################

def yyyymmdd2decimalyear(dates):
    '''
    Convert date to decimal years format according to convention that JPL uses
    for GIPSY timeseries (reference epoch 2000-01-01)
    '''
    dates = dates.astype('datetime64[s]')
    return 2000 + np.float64(dates - np.datetime64('2000-01-01')) / 86400 / 365.25

def decimalyear2yyymmdd(dates):
    '''
    Convert decimal years format (JPL-GIPSY convention) back to date (yyyy-mm-dd)
    format
    '''
    #add one sec to compensate the precison
    seconds = ((dates - 2000) * 365.25 * 86400 + 1).astype('timedelta64[s]')
    return (np.datetime64('2000-01-01') + seconds).astype('datetime64[D]')

def datetime2str(array):
    '''
    Convert np.datetime64 to string and remove '-' between year, month, day
    '''
    array = np.datetime_as_string(array)
    return np.char.replace(array, '-', '')

###############################################################################
################### Functions for ifg network generation ######################
###############################################################################

def get_dates_bperp(baseline_dir):
    '''
    Find all slc dates and bperp between stack reference and secondarys

    Input:
        baseline_dir: str - path to baseline directory
                            (/${stack}/merged/baseline) - files in gdal vrt format
                            (/${stack}/baseline) - files in txt format

    Output:
        date_list : numpy.ndarray   - 1D array of slc dates (dtype = datetime64[D])
        bperp_list : numpy.ndarray  - 1D array of slc perp. baseline between
                                        stack reference and secondary slcs'
                                        (dtype = float32)
    '''

    #Find the list of processed slc images

    slc_list = glob.glob(os.path.join(baseline_dir, '*/*.full.vrt'))

    if slc_list:
        #Get the dates
        date_list = np.vstack([datetime.strptime(os.path.basename(slc).split('.')[0], '%Y%m%d')
                    for slc in slc_list]).astype('datetime64[D]')

        #Get the Perpendicular baseline between the stack reference
        # and secondary images
        bperp_list = []
        for slc in slc_list:
            slc_ds = gdal.Open(slc)
            band = slc_ds.GetRasterBand(1)
            array = band.ReadAsArray()
            bperp_list.append(np.mean(array))
        slc_ds = None

    else:
        # Note: this option is much faster than the one from above
        slc_list = glob.glob(os.path.join(baseline_dir, '*/*.txt'))

        bperp_list = []
        date_list = []
        for slc in slc_list:
            # Get the dates
            date = os.path.basename(slc).split('.')[0].split('_')
            date1 = datetime.strptime(date[0], '%Y%m%d')
            date2 = datetime.strptime(date[1], '%Y%m%d')

            date_list.append(date2)

            #Get the Perpendicular baselines
            slc_file = open(slc, "r")
            txt = slc_file.read()
            bperp_list.append(np.mean(np.float32([txt.split('\n')[1].split(':')[-1],
                                                  txt.split('\n')[4].split(':')[-1]])))
        date_list.append(date1)
        bperp_list.append(0.0)
        date_list = np.vstack(date_list).astype('datetime64[D]')

    return date_list, np.vstack(bperp_list)

def create_pair_matrix(array):
    '''
    Create array of all pair combinations for time and baseline separation

    Input:
        array : 1D numpy.ndarray    - 1D array of dates or perpendicular
                                        baselines (dtype=float32/datetime64)

    Output:
        array : 2D numpy.ndarray    -2D array of dates/bperp dt for all pair
                                    combinations (dtype=float32/datetime64)
    '''
    # Number of slc
    num_data = array.shape[0]

    if array.ndim == 1:
        array = array[:, np.newaxis]

    array1 = np.repeat(array, num_data, axis=1)
    array2 = np.repeat(array.T, num_data, axis=0)

    #Get the time separation between slc [in days]
    # or perpendiculat baselines
    return np.float32(array2 - array1)

def check_ifgs_status(work_dir):
    '''
    Find the generated interferogram pairs to avoid redundant processing

    Input:
        work_dir : str     - path to coregistrated stack directory

    Output:
        ifg_pairs : 2D numpy.ndarray  - list of existing interferograms
                                        (dtype=datetime64[D])
    '''
    ifg_dir = os.path.join(work_dir, 'merged/interferograms')
    update_status = os.path.isdir(ifg_dir)

    if update_status:
        #Find the existing interferogram pairs
        ifg_list = glob.glob(os.path.join(ifg_dir, '*/filt_fine.unw'))
        ifg_pairs = [os.path.dirname(ifg).split('/')[-1].split('_') for ifg in ifg_list]

        ifg_pairs = np.vstack([[datetime.strptime(ifg_pair[0],'%Y%m%d'),
                                datetime.strptime(ifg_pair[1],'%Y%m%d')]
                               for ifg_pair in ifg_pairs]).astype('datetime64[D]')

    else:
        ifg_pairs = None

    return ifg_pairs

def select_pairs(date_list, bperp_list, network='sequential',
                 periodic=None, tolerance=1, single_reference_date=None, num_conn=4,
                 start_date=None, end_date=None, max_btemp=None, max_bperp=None,
                 remove_pairs=None, plot_fig=True):
    '''
    Select slc pairs for interferogram generation

    Input:
        date_list  : numpy.ndarray   - 1D array of slc dates (dtype = datetime64[D])
        bperp_list : numpy.ndarray   - 1D array of slc perp. baseline between
                                        stack reference and secondary slcs'
                                        (dtype = float32)

        network    : str             - type of network for pair selection
                                       options: single_reference, sequential, delaunay, full

        periodic   : str/list/array  - periodic pairs in days
                                       example: semi-annual and annual ['180', '360']

        tolerance  : int             - tolerance for selection of periodic pairs [in days]
                                       example: find annual pairs within 5 days around 1yr
                                       tolerance=5, 365.25 - 5 < pairs < 365.25 + 5

        single_reference_date : str  - reference date for single_reference (star)
                                        network, format '2015-01-01'

        num_conn   : int  -  number of connections for sequential network

        start_date : str  -  start_date for pair selection, format '2015-01-01'

        end_date   : str  -  end_date for pair selection, format '2015-01-01'

        max_btemp  : int  -  maximum temporal baseline for pair selection

        max_bperp  : int  -  maximum perpendicular baseline for pair selection

        remove_pairs :  2D numpy.ndarray  - array of existing pairs in
                                            $stack/merged/interferogram
                                            (dtype = datetime64[D])
        plot_fig   : bool  - options to plot generated network of interferogram pairs

    Output:
        acquisitionDates   : list of tuples  -  SLC dates in coreg_stack directory
        stackReferenceDate : tuple           -  reference date of co-registered SLC
                                                stack
        pairs              : list of tuples  -  list of interferogram pairs to process
        existing_pairs     : list of tuples  -  list of exisiting interferogram pairs
    '''

    network_options = ['single_reference', 'sequential',
                       'delaunay', 'full']

    if network not in network_options:
        raise ValueError(f'Network selection wrong, please select \
                         one of the folowing {network_options}')

    #combine the date and bperp lists, and sort
    slc_list = np.rec.fromarrays([date_list, bperp_list],
                                 dtype=([('dates','datetime64[D]'),
                                         ('bperp','float32')]))
    #Sort slc based on dates
    slc_list = np.sort(np.squeeze(slc_list), order=['dates'])

    #Get the date and baseline separation between all slc pairs
    date_dt = create_pair_matrix(slc_list.dates)
    bperp_dt = create_pair_matrix(slc_list.bperp)

    #Get the dates matrix
    decimal_dates = yyyymmdd2decimalyear(slc_list.dates)

    #Symetrical matrix of dates
    dec_dates_array = np.triu(decimal_dates) + np.triu(decimal_dates).T \
                        - np.diag(decimal_dates)
    dates_array = decimalyear2yyymmdd(dec_dates_array)

    ############## GENERATE IFG PAIR NETWORK #################################
    if network == 'sequential':
        pairs = np.empty((0,2), dtype='datetime64[D]')
        for i, reference in enumerate(slc_list.dates):
            reference_pair = np.repeat(reference, int(num_conn))
            secondary_pair = slc_list.dates[i+1 : i+int(num_conn)+1]
            second_ndim = secondary_pair.shape[0]
            pairs = np.append(pairs,
                              np.vstack((reference_pair[:second_ndim],
                                         secondary_pair)).T,
                              axis=0)

    elif network == 'delaunay':
        from scipy.spatial import Delaunay

        points = np.vstack([decimal_dates, slc_list.bperp]).T
        tri = Delaunay(points)
        simplices = points[tri.simplices, 0]
        pairs = np.vstack((
                    simplices[:, (0,1)],
                    simplices[:, (0,2)],
                    simplices[:, (1,2)]))

        pairs = decimalyear2yyymmdd(pairs)

    elif network == 'single_reference':
        if single_reference_date is not None:
            # single_reference_date in str format '2020-01-01'
            single_reference_date = np.datetime64(
                datetime.strptime(single_reference_date,'%Y-%m-%d'))
            reference = np.where(slc_list.dates == single_reference_date)[0]

        else:
            #find the reference date by mininimizing the bperp and temp baselines
            sum_bperp_date = np.sum(np.abs(bperp_dt), axis=0)
            minimal_bperp = np.where(sum_bperp_date == np.min(sum_bperp_date))

            sum_dt_date = np.sum(np.abs(date_dt), axis=0)
            reference = np.where(sum_dt_date == np.min(sum_dt_date[minimal_bperp]))[0]

        if reference.size != 1:
            reference = reference[0]

        # Generate the network pairs
        reference_pair = np.repeat(slc_list.dates[reference],
                                   slc_list.dates.size,
                                   axis=0)
        secondary_pair = slc_list.dates
        pairs = np.vstack([reference_pair,
                           secondary_pair]).T
        pairs = np.delete(pairs, reference, axis=0)
        periodic = None

    elif network == 'full':
        from itertools import permutations
        pairs = slc_list.dates[np.vstack(list(permutations(
            np.linspace(0, slc_list.dates.size -1 , slc_list.dates.size), 2))).astype(np.int32)]
        periodic = None

    ####################### PERIODIC PAIRS ####################################
    #find periodic pairs, like annual pairs 365 days, periodic is in days
    if periodic is not None:
        if not isinstance(periodic, (np.ndarray, np.generic)):
            periodic = np.array(periodic, dtype=np.float32)
        if periodic.ndim == 0:
            periodic = periodic.reshape(1,1)

        for dt in periodic:
            index12 = np.where((date_dt < dt + np.float32(tolerance)) &
                               (date_dt > dt - np.float32(tolerance)))

            pairs = np.append(pairs,
                              np.vstack((dates_array[index12[0], 0],
                                         dates_array[index12[1], 0])).T,
                              axis=0)

    ###################### FILTER USING START/END DATE ########################
    if start_date or end_date:
        try:
            start_date = np.datetime64(datetime.strptime(start_date,'%Y-%m-%d'), 'D')  \
                            if start_date else slc_list.dates[0]

            end_date = np.datetime64(datetime.strptime(end_date,'%Y-%m-%d'), 'D') \
                            if end_date else slc_list.dates[-1]

        except:
            raise ValueError('Start or End date is in wrong format, please \
                             define it as str(2000-01-01) for Jan 01, 2000')

        pairs = pairs[np.where(((pairs >= start_date).all(axis=1)) &
                                ((pairs <= end_date).all(axis=1)))]

    ###########################################################################
    #Turn pairs to index
    pairs_ix = np.vstack([np.where(np.isin(slc_list.dates, pair)) for pair in pairs])

    #Remove duplicated pairs
    pairs_ix = np.unique(pairs_ix, axis=0)

    #################### UPDATE EXISTING IFG STACK #############################
    #Remove existing pairs
    if remove_pairs is not None:
        #turn existing dates to index values
        existing_pairs_ix = np.vstack([np.where(np.isin(slc_list.dates, ifg))
                                       for ifg in remove_pairs])

        # find existing pairs in newly generated pair list
        pairs_to_remove = [np.where((pairs_ix == ix).all(axis=1))
                                     for ix in existing_pairs_ix]

        #remove empty pairs
        pairs_to_remove = [pair_rm for pair_rm in pairs_to_remove
                           if pair_rm[0].size !=0]
        # Remove existing pairs from the list
        if pairs_to_remove:
            pairs_ix = np.delete(pairs_ix, np.vstack(pairs_to_remove), axis=0)

        # Generate tuple list of existing ifg pairs
        existing_pairs = tuple([tuple(pair) for pair in datetime2str(slc_list.dates[existing_pairs_ix])])
    else:
        existing_pairs = None

    ############ FILTER PAIRS USING MAX  BPERP/TBASE BASELINES ################
    # Filter pairs according to max_tbase and max_bperp
    bperp_tbase_ix = np.empty((1, 2))
    if max_bperp:
        bperp_tbase_ix = np.append(bperp_tbase_ix,
                                   np.vstack(np.where(np.abs(bperp_dt) > max_bperp)).T,
                                   axis=0)
    if max_btemp:
        bperp_tbase_ix = np.append(bperp_tbase_ix,
                                   np.vstack(np.where(np.abs(date_dt) > max_btemp)).T,
                                   axis=0)

    if bperp_tbase_ix.shape[0] != 1:
        pairs_to_remove = [np.where((pairs_ix == ix).all(axis=1))
                                     for ix in bperp_tbase_ix]
        pairs_to_remove = [pair_rm for pair_rm in pairs_to_remove
                           if pair_rm[0].size !=0]

        # Remove existing pairs from the list
        if pairs_to_remove:
            pairs_ix = np.delete(pairs_ix, np.vstack(pairs_to_remove), axis=0)

    ########################## PREPARE OUTPUTS ###############################
    #Convert output datetime format to str for interferogram_stack func
    acquisitionDates = tuple(datetime2str(slc_list.dates))
    stackReferenceDate = tuple(datetime2str(slc_list.dates[slc_list.bperp==0.0]))[0]

    #Convert pairs array to interferogramStack.py format
    # array to tuple
    pairs = tuple([tuple(pair) for pair in datetime2str(slc_list.dates[pairs_ix])])

    ##################### PLOT THE NETWORK GRAPH ##############################
    if plot_fig:
        from matplotlib import pyplot as plt
        import matplotlib
        import logging
        logging.getLogger('matplotlib.font_manager').disabled = True
        matplotlib.use('Agg')

        fig, ax = plt.subplots(1, figsize=(10, 6))
        # new pairs
        ax.plot([slc_list.dates[pairs_ix][:,0], slc_list.dates[pairs_ix][:,1]],
                [slc_list.bperp[pairs_ix][:,0], slc_list.bperp[pairs_ix][:,1]],
                '-', marker='o', color='darkolivegreen', linewidth=1.7, alpha=0.7,
                markerfacecolor='darkslategrey', markeredgecolor='black')

        ax.set_xlabel('Date')
        ax.set_ylabel('Perp. Baseline [m]')
        ax.set_title('Interferogram Network')
        extra = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="darkolivegreen",
                                             fill=True, edgecolor='none', linewidth=0)

        if remove_pairs is not None:
            # previous pairs
            ax.plot([slc_list.dates[existing_pairs_ix][:,0], slc_list.dates[existing_pairs_ix][:,1]],
                    [slc_list.bperp[existing_pairs_ix][:,0], slc_list.bperp[existing_pairs_ix][:,1]],
                    '-', marker='o', color='lightsteelblue', linewidth=1.4, alpha=0.6,
                    markerfacecolor='darkslategrey', markeredgecolor='none')

            extra1 = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="lightsteelblue", alpha=0.6,
                                                 fill=True, edgecolor='none', linewidth=0)
            extra2 = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="none",
                                                 fill=False, edgecolor='none', linewidth=0)
            extra = [extra, extra1, extra2]
            ax.legend(extra, [f'New pairs: {len(pairs_ix)}', f'Existing pairs: {len(existing_pairs_ix)}',
                              f'Num of all pairs: {len(pairs_ix) + len(existing_pairs_ix)}'],
                      facecolor='slategrey', framealpha=0.2)
        else:
            ax.legend([extra], [f'Num of pairs: {len(pairs_ix)}'], facecolor='slategrey', framealpha=0.2)

        # Save network figure
        fig.savefig('interferogram_network.pdf')

    return acquisitionDates, stackReferenceDate, pairs, existing_pairs


###############################################################################
################# interferogram stack - gen. of run_files #####################
################# Func.from isce2/contrib/stack/topsStack/stackSentinel.py  ###

def interferogramStack(inps, acquisitionDates, stackReferenceDate, secondaryDates,
                       safe_dict, pairs, updateStack):

    i = 20 #Set it to 20 as a second step run files

    '''
    MG: skip as this step has been already done in stack processing
    i+=1
    runObj = run()
    runObj.configure(inps, 'run_{:02d}_merge_reference_secondary_slc'.format(i))
    runObj.mergeReference(stackReferenceDate, virtual = virtual_merge)
    runObj.mergeSecondarySLC(secondaryDates, virtual = virtual_merge)
    runObj.finalize()
    '''
    i+=1
    runObj = run()
    runObj.configure(inps, 'run_{:02d}_generate_burst_igram'.format(i))
    runObj.generate_burstIgram(acquisitionDates, safe_dict, pairs)
    runObj.finalize()

    i += 1
    runObj = run()
    runObj.configure(inps, 'run_{:02d}_merge_burst_igram'.format(i))
    runObj.igram_mergeBurst(acquisitionDates, safe_dict, pairs)
    runObj.finalize()

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_{:02d}_filter_coherence'.format(i))
    runObj.filter_coherence(pairs)
    runObj.finalize()

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_{:02d}_unwrap'.format(i))
    runObj.unwrap(pairs)
    runObj.finalize()

    return i

###############################################################################
##############################     MAIN     s##################################
###############################################################################

def main(iargs=None):

    inps = cmdLineParse(iargs)

    #Define the coreg_slc merged/baseline folder
    baseline_dir = os.path.join(inps.work_dir, 'merged/baselines')

    if not os.path.exists(baseline_dir):
        baseline_dir = os.path.join(inps.work_dir, 'baselines')

    # Souble check if all needed directory exist in the work_dir
    if not os.path.exists(os.path.join(inps.work_dir, 'coreg_secondarys')):
        raise ValueError('coreg_secondarys does not exist in defined stack dir, \
                         double check the stack dir path or re-run stackSentinel.py')
    if not os.path.exists(os.path.join(inps.work_dir, 'reference')):
        raise ValueError('reference does not exist in defined stack dir, \
                         double check the stack dir path or re-run stackSentinel.py')

    if os.path.exists(os.path.join(inps.work_dir, 'run_ifg_files')):
        print('')
        print('**************************')
        print('run_ifg_files folder exists.')
        print(os.path.join(inps.work_dir, 'run_ifg_files'), ' already exists.')
        if inps.force_run_files:
            #Delete all run_files
            for f in os.listdir(os.path.join(inps.work_dir, 'run_ifg_files')):
                os.remove(os.path.join(os.path.join(inps.work_dir, 'run_ifg_files'), f))
            print('Deleting all existing run_files in this folder.')
            print('')
            print('**************************')
        else:
            print('Please remove or rename this folder or use --force option and try again.')
            print('')
            print('**************************')
            sys.exit(1)

    # Get the date and bperp list for coregistrated slc
    date_list, bperp_list = get_dates_bperp(baseline_dir)

    # Check if there are existing interferograms in work_dir, if yes use update mode
    remove_pairs = check_ifgs_status(inps.work_dir)

    acquisitionDates, stackReferenceDate, pairs, existing_pairs = select_pairs(
                                                                    date_list,
                                                                    bperp_list,
                                                                    network=inps.network,
                                                                    periodic=inps.periodic_connections,
                                                                    tolerance = inps.periodic_tolerance,
                                                                    single_reference_date=inps.single_reference_date,
                                                                    num_conn=inps.num_connections,
                                                                    start_date=inps.startDate,
                                                                    end_date=inps.endDate,
                                                                    max_btemp=inps.max_btemp,
                                                                    max_bperp=inps.max_bperp,
                                                                    remove_pairs=remove_pairs)

    if remove_pairs is not None:
        print('')
        print('Updating an existing interferogram stack ...')
        print('')

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('')
        print(f'Existing pairs in the interferogram stack (# {len(existing_pairs)}): ')
        print('')
        [print(existing_pair) for existing_pair in existing_pairs]

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('')
        print(f'New pairs in this update(# {len(pairs)}): ')
        print('')
        [print(pair) for pair in pairs]

    else:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('')
        print(f'Selected pairs (# {len(pairs)}): ')
        print('')
        [print(pair) for pair in pairs]

    #Create safe_dict needed for interferogramStack func
    safe_dict = {}
    for date in acquisitionDates:
        safe_dict[date] = sentinelSLC()

    #Add the reference date to input class
    inps.reference_date = stackReferenceDate

    #Crete run and config files
    interferogramStack(inps, acquisitionDates, stackReferenceDate, None, safe_dict, pairs, None)

if __name__ == "__main__":
    # Main engine
    main(sys.argv[1:])
