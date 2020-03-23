#!/usr/bin/env python3

#Author: Heresh Fattahi

import os, imp, sys, glob
import argparse
import configparser
import datetime
import numpy as np
import shelve
import isce
import isceobj
from mroipac.baseline.Baseline import Baseline
from Stack import config, run, selectPairs


filtStrength = '0.8'
noMCF = 'False'
defoMax = '2'
maxNodes = 72

def createParser():
    parser = argparse.ArgumentParser( description='Preparing the directory structure and config files for stack processing of StripMap data')

    parser.add_argument('-s', '--slc_directory', dest='slcDir', type=str, required=True,
            help='Directory with all stripmap SLCs')

    parser.add_argument('-x', '--bbox', dest='bbox', type=str, default=None, help='Lat/Lon Bounding SNWE')

    parser.add_argument('-w', '--working_directory', dest='workDir', type=str, default='./',
            help='Working directory ')

    parser.add_argument('-d', '--dem', dest='dem', type=str, required=True,
            help='DEM file (with .xml and .vrt files)')

    parser.add_argument('-m', '--master_date', dest='masterDate', type=str, default=None,
            help='Directory with master acquisition')
    
    parser.add_argument('-t', '--time_threshold', dest='dtThr', type=float, default=10000.0,
            help='Time threshold (max temporal baseline in days)')

    parser.add_argument('-b', '--baseline_threshold', dest='dbThr', type=float, default=5000.0,
            help='Baseline threshold (max bperp in meters)')

    parser.add_argument('-a', '--azimuth_looks', dest='alks', type=str, default='10',
            help='Number of looks in azimuth (automaticly computed as AspectR*looks when '
                 '"S" or "sensor" is defined to give approximately square multi-look pixels)')
    parser.add_argument('-r', '--range_looks', dest='rlks', type=str, default='10',
            help='Number of looks in range')
    parser.add_argument('-S', '--sensor', dest='sensor', type=str, required=False,
            help='SAR sensor used to define square multi-look pixels')

    parser.add_argument('-u', '--unw_method', dest='unwMethod', type=str, default='snaphu', 
            help='unwrapping method (icu, snaphu, or snaphu2stage)')

    parser.add_argument('-f','--filter_strength', dest='filtStrength', type=str, default=filtStrength,
            help='strength of Goldstein filter applied to the wrapped phase before spatial coherence estimation.'
                 ' Default: {}'.format(filtStrength))

    iono = parser.add_argument_group('Ionosphere', 'Configurationas for ionospheric correction')
    iono.add_argument('-L', '--low_band_frequency', dest='fL', type=str, default=None,
            help='low band frequency')
    iono.add_argument('-H', '--high_band_frequency', dest='fH', type=str, default=None,
            help='high band frequency')
    iono.add_argument('-B', '--subband_bandwidth ', dest='bandWidth', type=str, default=None,
            help='sub-band band width')

    iono.add_argument('--filter_sigma_x', dest='filterSigmaX', type=str, default='100', 
            help='filter sigma for gaussian filtering the dispersive and nonDispersive phase')

    iono.add_argument('--filter_sigma_y', dest='filterSigmaY', type=str, default='100.0',
            help='sigma of the gaussian filter in Y direction, default=100')

    iono.add_argument('--filter_size_x', dest='filterSizeX', type=str, default='800.0',
            help='size of the gaussian kernel in X direction, default = 800')

    iono.add_argument('--filter_size_y', dest='filterSizeY', type=str, default='800.0',
            help='size of the gaussian kernel in Y direction, default=800')

    iono.add_argument('--filter_kernel_rotation', dest='filterKernelRotation', type=str, default='0.0',
            help='rotation angle of the filter kernel in degrees (default = 0.0)')

    parser.add_argument('-W', '--workflow', dest='workflow', type=str, default='slc', 
            help='The InSAR processing workflow : (slc, interferogram, ionosphere)')

    parser.add_argument('-z', '--zero', dest='zerodop', action='store_true', default=False, 
            help='Use zero doppler geometry for processing - Default : No')
    parser.add_argument('--nofocus', dest='nofocus', action='store_true', default=False, 
            help='If input data is already focused to SLCs - Default : do focus')
    parser.add_argument('-c', '--text_cmd', dest='text_cmd', type=str, default='', 
            help='text command to be added to the beginning of each line of the run files. Example : source ~/.bash_profile;')
    parser.add_argument('-useGPU', '--useGPU', dest='useGPU',action='store_true', default=False,
             help='Allow App to use GPU when available')

    parser.add_argument('--summary', dest='summary', action='store_true', default=False, help='Show summary only')
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    inps = parser.parse_args(args=iargs)
    inps.slcDir = os.path.abspath(inps.slcDir)
    inps.workDir = os.path.abspath(inps.workDir)
    inps.dem = os.path.abspath(inps.dem)
     
    return inps

    
def get_dates(inps):
 
    dirs = glob.glob(inps.slcDir+'/*')
    acuisitionDates = []
    for dirf in dirs:
        if inps.nofocus:
            expectedRaw = os.path.join(dirf,os.path.basename(dirf) + '.slc')
        else:
            expectedRaw = os.path.join(dirf, os.path.basename(dirf) + '.raw')

        if os.path.exists(expectedRaw):
             acuisitionDates.append(os.path.basename(dirf))

    acuisitionDates.sort()
    print (dirs)
    print (acuisitionDates)
    if inps.masterDate not in acuisitionDates:
        print ('master date was not found. The first acquisition will be considered as the stack master date.')
    if inps.masterDate is None or inps.masterDate not in acuisitionDates:
        inps.masterDate = acuisitionDates[0]
    slaveDates = acuisitionDates.copy()
    slaveDates.remove(inps.masterDate)
    return acuisitionDates, inps.masterDate, slaveDates 
  
def slcStack(inps, acquisitionDates, stackMasterDate, slaveDates, pairs, splitFlag=False, rubberSheet=False):
    # A coregistered stack of SLCs
    i=0


    if inps.bbox:
        i+=1
        runObj = run()
        runObj.configure(inps, 'run_' + str(i) + "_crop")
        config_prefix = "config_crop_"
        runObj.crop(acquisitionDates, config_prefix, native=not inps.zerodop, israw=not inps.nofocus)
        runObj.finalize()


    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_master")
    config_prefix = "config_master_"
    runObj.master_focus_split_geometry(stackMasterDate, config_prefix, split=splitFlag, focus=not inps.nofocus, native=not inps.zerodop)
    runObj.finalize()

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_focus_split")
    config_prefix = "config_focus_split"
    runObj.slaves_focus_split(slaveDates, config_prefix, split=splitFlag, focus=not inps.nofocus, native=not inps.zerodop)
    runObj.finalize()

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_geo2rdr_coarseResamp")
    config_prefix = "config_geo2rdr_coarseResamp_"
    runObj.slaves_geo2rdr_resampleSlc(stackMasterDate, slaveDates, config_prefix, native=(not inps.nofocus) or (not inps.zerodop))
    runObj.finalize()

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_refineSlaveTiming")
    config_prefix = 'config_refineSlaveTiming_'
    runObj.refineSlaveTiming_Network(pairs, stackMasterDate, slaveDates, config_prefix)
    runObj.finalize()

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_invertMisreg")
    runObj.invertMisregPoly()
    runObj.finalize()

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_fineResamp")
    config_prefix = 'config_fineResamp_'
    runObj.slaves_fine_resampleSlc(stackMasterDate, slaveDates, config_prefix, split=splitFlag)
    runObj.finalize()
    
    if rubberSheet:
       i+=1
       runObj = run()
       runObj.configure(inps, 'run_' + str(i) + "_denseOffset")
       config_prefix = 'config_denseOffset_'
       runObj.denseOffsets_Network(pairs, stackMasterDate, slaveDates, config_prefix) 
       runObj.finalize()

       i+=1
       runObj = run()
       runObj.configure(inps, 'run_' + str(i) + "_invertDenseOffsets")
       runObj.invertDenseOffsets()
       runObj.finalize()
 
       i+=1
       runObj = run()
       runObj.configure(inps, 'run_' + str(i) + "_resampleOffset")
       config_prefix = 'config_resampOffsets_'
       runObj.resampleOffset(slaveDates, config_prefix)
       runObj.finalize()

       i+=1
       runObj = run()
       runObj.configure(inps, 'run_' + str(i) + "_replaceOffsets")
       runObj.replaceOffsets(slaveDates)
       runObj.finalize()

       i+=1
       runObj = run()
       runObj.configure(inps, 'run_' + str(i) + "_fineResamp")
       config_prefix = 'config_fineResamp_'
       runObj.slaves_fine_resampleSlc(stackMasterDate, slaveDates, config_prefix, split=splitFlag)
       runObj.finalize()

    # adding the baseline grid generation
    i+=1
    config_prefix = 'config_baselinegrid_'
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_grid_baseline")
    runObj.gridBaseline(stackMasterDate, slaveDates,config_prefix)
    runObj.finalize()

    return i

def interferogramStack(inps, acquisitionDates, stackMasterDate, slaveDates, pairs):
    # an interferogram stack without ionosphere correction. 
    # coregistration is with geometry + const offset


    i = slcStack(inps, acquisitionDates, stackMasterDate, slaveDates, pairs, splitFlag=False, rubberSheet=False)
    
    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_igram")
    config_prefix = 'config_igram_'
    low_or_high = "/"
    runObj.igrams_network(pairs, acquisitionDates, stackMasterDate, low_or_high, config_prefix)    
    runObj.finalize()

def interferogramIonoStack(inps, acquisitionDates, stackMasterDate, slaveDates, pairs):

    # raise exception for ALOS-1 if --fbd2fbs was used
    run_unpack_file = os.path.join(inps.workDir, 'run_unPackALOS')
    if os.path.isfile(run_unpack_file):
        with open(run_unpack_file, 'r') as f:
            lines = f.readlines()
        if any('fbd2fbs' in line for line in lines):
            msg = 'ALOS-1 FBD mode data exists with fbd2fbs enabled, which is not applicable for ionosphere workflow'
            msg += '\nsolution: restart from prepRawALOS.py WITHOUT --dual2single/--fbd2fbs option.'
            raise ValueError(msg)

    # an interferogram stack with ionosphere correction.
    # coregistration is with geometry + const offset + rubbersheeting

    i = slcStack(inps, acquisitionDates, stackMasterDate, slaveDates, pairs, splitFlag=True, rubberSheet=True)

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_igram")
    config_prefix = 'config_igram_'
    low_or_high = "/"
    runObj.igrams_network(pairs, acquisitionDates, stackMasterDate, low_or_high, config_prefix)
    runObj.finalize()    

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_igramLowBand")
    config_prefix = 'config_igramLowBand_'
    low_or_high = "/LowBand/"
    runObj.igrams_network(pairs, acquisitionDates, stackMasterDate, low_or_high, config_prefix)
    runObj.finalize()

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_igramHighBand")
    config_prefix = 'config_igramHighBand_'
    low_or_high = "/HighBand/"
    runObj.igrams_network(pairs, acquisitionDates, stackMasterDate, low_or_high, config_prefix)
    runObj.finalize()

    i+=1
    runObj = run()
    runObj.configure(inps, 'run_' + str(i) + "_iono")
    config_prefix = 'config_iono_'
    lowBand = '/LowBand/'
    highBand = '/HighBand/'
    runObj.dispersive_nonDispersive(pairs, acquisitionDates, stackMasterDate,
                           lowBand, highBand, config_prefix)
    runObj.finalize()

def main(iargs=None):

  inps = cmdLineParse(iargs)
  # name of the folder of the coreg SLCs including baselines, SLC, geom_master subfolders
  inps.stack_folder = 'merged'
  inps.dense_offsets_folder = 'dense_offsets'


  # check if a sensor is defined and update if needed azimuth looks to give square pixels
  ar=1
  if inps.sensor:
      if inps.sensor.lower() == "alos":
          ar=4            
          print("Looks like " + inps.sensor.lower() + ", multi-look AR=" + str(ar))
      elif inps.sensor.lower() == "envisat" or inps.sensor.lower() == "ers":
          ar=5
          print("Looks like " + inps.sensor.lower() + ", multi-look AR=" + str(ar))
      else:
          print("Sensor is not hard-coded (ers, envisat, alos), will keep default alks")
          # sensor is not recognised, report to user and state default 
  inps.alks = str(int(inps.alks)*int(ar))
 
  # getting the acquisitions
  acquisitionDates, stackMasterDate, slaveDates = get_dates(inps)
  configDir = os.path.join(inps.workDir,'configs')
  if not os.path.exists(configDir):
       os.makedirs(configDir)
  runDir = os.path.join(inps.workDir,'run_files')
  if not os.path.exists(runDir):
       os.makedirs(runDir)

  if inps.sensor.lower() == 'uavsar_stack':    # don't try to calculate baselines for UAVSAR_STACK data
    pairs = selectPairs(inps,stackMasterDate, slaveDates, acquisitionDates,doBaselines=False)
  else:
    pairs = selectPairs(inps,stackMasterDate, slaveDates, acquisitionDates,doBaselines=True)  
  print ('number of pairs: ', len(pairs))

  ###If only a summary is requested quit after this
  if inps.summary:
      return

  #if cropping is requested, then change the slc directory:
  inps.fullFrameSlcDir = inps.slcDir

  if inps.bbox:
     inps.slcDir = inps.slcDir + "_crop"
  #############################

  if inps.workflow == 'slc':
     slcStack(inps, acquisitionDates, stackMasterDate, slaveDates, pairs, splitFlag=False, rubberSheet=False)

  elif inps.workflow == 'interferogram':
     interferogramStack(inps, acquisitionDates, stackMasterDate, slaveDates, pairs)

  elif inps.workflow == 'ionosphere':
     interferogramIonoStack(inps, acquisitionDates, stackMasterDate, slaveDates, pairs) 

  
if __name__ == "__main__":
       
  # Main engine  
  main()
       

