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

filtStrength = '0.8'
noMCF = 'False'
defoMax = '2'
maxNodes = 72

    
class config(object):
    """
       A class representing the config file
    """
    def __init__(self, outname):
       self.f= open(outname,'w')
       self.f.write('[Common]'+'\n')
       self.f.write('')
       self.f.write('##########################'+'\n')

    def configure(self,inps):
       for k in inps.__dict__.keys():
           setattr(self, k, inps.__dict__[k])
       self.plot = 'False'
       self.misreg = None

    def cropFrame(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('cropFrame : ' + '\n')
        self.f.write('input : ' +  self.inputDir + '\n')
        self.f.write('box_str : ' + self.bbox + '\n')
        self.f.write('output : ' + self.cropOutputDir + '\n')

        ##For booleans, just having an entry makes it True
        ##Value of the text doesnt matter
        if self.nativeDoppler:
            self.f.write('native : True \n')
        if self.israw:
            self.f.write('raw : True \n')
        self.f.write('##########################'+'\n')

    def focus(self,function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('focus : '+'\n')
        self.f.write('input : ' + self.slcDir +'\n')
        self.f.write('##########################'+'\n')

    def topo(self,function):

        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('topo : '+'\n')
        self.f.write('master : ' + self.slcDir +'\n')
        self.f.write('dem : ' + self.dem +'\n')
        self.f.write('output : ' + self.geometryDir +'\n')
        self.f.write('alks : ' + self.alks +'\n')
        self.f.write('rlks : ' + self.rlks +'\n')
        if self.nativeDoppler:
            self.f.write('native : True\n')
        if self.useGPU:
            self.f.write('useGPU : True \n')
        else:
            self.f.write('useGPU : False\n')
        self.f.write('##########################'+'\n')

    def createWaterMask(self, function):

        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('createWaterMask : '+'\n')
        self.f.write('dem_file : ' + self.dem +'\n')
        self.f.write('lat_file : ' + self.latFile +'\n')
        self.f.write('lon_file : ' + self.lonFile +'\n')
        self.f.write('output : ' + self.waterMaskFile + '\n')
        self.f.write('##########################'+'\n')

    def geo2rdr(self, function):

        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('geo2rdr : '+'\n')
        self.f.write('master : ' + self.masterSlc +'\n')
        self.f.write('slave : ' + self.slaveSlc +'\n')
        self.f.write('geom : ' + self.geometryDir +'\n')
        if self.nativeDoppler:
            self.f.write('native : True\n')
        if self.useGPU:
            self.f.write('useGPU : True \n')
        else:
            self.f.write('useGPU : False\n')
        self.f.write('outdir : ' + self.offsetDir+'\n')
        self.f.write('##########################'+'\n')

    def resampleSlc(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('resampleSlc : '+'\n')
        self.f.write('master : ' + self.masterSlc + '\n')
        self.f.write('slave : ' + self.slaveSlc +'\n')
        self.f.write('coreg : ' + self.coregSlaveSlc +'\n')
        self.f.write('offsets : ' + self.offsetDir +'\n')
        if self.misreg:
           self.f.write('poly : ' + self.misreg + '\n')
        self.f.write('##########################'+'\n')

    def resampleSlc_subband(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('resampleSlc_subBand : '+'\n')
        #self.f.write('master : ' + self.masterSlc + '\n')
        self.f.write('slave : ' + self.slaveSlc +'\n')
        self.f.write('coreg : ' + self.coregSlaveSlc +'\n')
        self.f.write('offsets : ' + self.offsetDir +'\n')
        if self.misreg:
           self.f.write('poly : ' + self.misreg + '\n')
        self.f.write('##########################'+'\n')

    def baselineGrid(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function + '\n')
        self.f.write('baselineGrid : ' + '\n')
        self.f.write('master : ' + self.coregSlaveSlc + "/masterShelve" + '\n')
        self.f.write('slave : '  + self.coregSlaveSlc + "/slaveShelve" + '\n')
        self.f.write('baseline_file : ' + self.baselineGridFile + '\n')

    def refineSlaveTiming(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('refineSlaveTiming : '+'\n')
        self.f.write('master : ' + self.masterSlc + '\n')
        self.f.write('slave : ' + self.slaveSlc +'\n')
        self.f.write('mm : ' + self.masterMetaData + '\n')
        self.f.write('ss : ' + self.slaveMetaData + '\n')
        self.f.write('outfile : '+ self.outfile + '\n')

    def denseOffsets(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('denseOffsets : '+'\n')
        self.f.write('master : ' + self.masterSlc + '\n')
        self.f.write('slave : ' + self.slaveSlc +'\n')
        self.f.write('outPrefix : '+ self.outfile + '\n')

    def filterOffsets(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('MaskAndFilter : '+'\n')
        self.f.write('dense_offset : ' + self.denseOffset + '\n')
        self.f.write('snr : ' + self.snr +'\n')
        self.f.write('output_directory : '+ self.outDir + '\n') 

    def resampleOffset(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('resampleOffsets : ' + '\n')
        self.f.write('input : ' + self.input + '\n')
        self.f.write('target_file : '+ self.targetFile + '\n')
        self.f.write('output : ' + self.output + '\n')

    def rubbersheet(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('rubberSheeting : ' + '\n')
        self.f.write('geometry_azimuth_offset : ' + self.geometry_azimuth_offset + '\n')
        self.f.write('dense_offset : '+ self.dense_offset + '\n')
        self.f.write('snr : ' + self.snr + '\n')
        self.f.write('output_azimuth_offset : ' + self.output_azimuth_offset + '\n')
        self.f.write('output_directory : ' + self.output_directory + '\n')

    def generateIgram(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('crossmul : '+'\n')
        self.f.write('master : ' + self.masterSlc +'\n')
        self.f.write('slave : ' + self.slaveSlc +'\n')
        self.f.write('outdir : ' + self.outDir + '\n')
        self.f.write('alks : ' + self.alks + '\n')
        self.f.write('rlks : ' + self.rlks + '\n')
        self.f.write('##########################'+'\n')

    def filterCoherence(self, function): 
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('FilterAndCoherence : ' + '\n')
        self.f.write('input : ' + self.igram + '\n')
        self.f.write('filt : ' + self.filtIgram + '\n')
        self.f.write('coh : ' + self.coherence  + '\n')
        self.f.write('strength : ' + self.filtStrength + '\n')
        self.f.write('##########################'+'\n')

    def unwrap(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n') 
        self.f.write('unwrap : ' + '\n')
        self.f.write( 'ifg : ' + self.igram + '\n')  
        self.f.write( 'coh : ' + self.coherence + '\n')
        self.f.write( 'unwprefix : ' + self.unwIfg + '\n')      
        self.f.write('nomcf : ' + self.noMCF + '\n')
        self.f.write('master : ' + self.master + '\n')
        self.f.write('defomax : ' + self.defoMax + '\n')
        self.f.write('alks : ' + self.alks + '\n')
        self.f.write('rlks : ' + self.rlks + '\n')
        self.f.write('method : ' + self.unwMethod + '\n')
        self.f.write('##########################'+'\n')

    def splitRangeSpectrum(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('splitSpectrum : ' + '\n')
        self.f.write('slc : ' + self.slc + '\n')
        self.f.write('outDir : ' + self.outDir + '\n')
        self.f.write('shelve : ' + self.shelve + '\n')
        if self.fL and self.fH and self.bandWidth:
           self.f.write('dcL : ' + self.fL + '\n')
           self.f.write('dcH : ' + self.fH + '\n')
           self.f.write('bw : ' + self.bandWidth + '\n')
        self.f.write('##########################'+'\n')
   
    def estimateDispersive(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('estimateIono :' + '\n')
        self.f.write('low_band_igram_prefix : ' + self.lowBandIgram + '\n')
        self.f.write('high_band_igram_prefix : ' + self.highBandIgram + '\n')
        self.f.write('low_band_igram_unw_method : ' + self.unwMethod + '\n')
        self.f.write('high_band_igram_unw_method : ' + self.unwMethod + '\n')
        self.f.write('low_band_shelve : '+ self.lowBandShelve +'\n')
        self.f.write('high_band_shelve : '+ self.highBandShelve +'\n')
        self.f.write('low_band_coherence : ' + self.lowBandCor + '\n')
        self.f.write('high_band_coherence : ' + self.highBandCor + '\n')
        self.f.write('azimuth_looks : ' + self.alks + '\n')
        self.f.write('range_looks : ' + self.rlks + '\n')
        self.f.write('filter_sigma_x : ' + self.filterSigmaX + '\n')
        self.f.write('filter_sigma_y : ' + self.filterSigmaY + '\n')
        self.f.write('filter_size_x : ' + self.filterSizeX + '\n')
        self.f.write('filter_size_y : ' + self.filterSizeY + '\n')
        self.f.write('filter_kernel_rotation : ' + self.filterKernelRotation + '\n')
        self.f.write('outDir : ' + self.outDir + '\n')
        self.f.write('##########################'+'\n')

    def finalize(self):
        self.f.close()



def get_dates(inps):
 
  dirs = glob.glob(inps.slcDir+'/*')
  acuisitionDates = []
  for dir in dirs:
     expectedRaw = os.path.join(dir,os.path.basename(dir) + '.slc')
     if os.path.exists(expectedRaw):
        acuisitionDates.append(os.path.basename(dir))

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
  
class run(object):
    """
       A class representing a run which may contain several functions
    """
    #def __init__(self):

    def configure(self,inps, runName):
        for k in inps.__dict__.keys():
            setattr(self, k, inps.__dict__[k])
        self.runDir = os.path.join(self.workDir, 'run_files')
        if not os.path.exists(self.runDir):
            os.makedirs(self.runDir)

        self.run_outname = os.path.join(self.runDir, runName)
        print ('writing ', self.run_outname)

        self.configDir = os.path.join(self.workDir,'configs')
        if not os.path.exists(self.configDir):
            os.makedirs(self.configDir)

        # passing argument of started from raw
        if inps.nofocus is  False:
            self.raw_string = '.raw'
        else:
            self.raw_string = '' 


        # folder structures
        self.stack_folder = inps.stack_folder
        selfdense_offsets_folder = inps.dense_offsets_folder

        self.runf= open(self.run_outname,'w')

    def crop(self, acquisitionDates, config_prefix, native=True, israw=True):
        for d in acquisitionDates:
             configName = os.path.join(self.configDir, config_prefix + d)
             configObj = config(configName)
             configObj.configure(self)
             configObj.inputDir = os.path.join(self.fullFrameSlcDir, d)
             configObj.cropOutputDir = os.path.join(self.slcDir, d)
             configObj.bbox = self.bbox
             configObj.nativeDoppler = native
             configObj.israw = israw
             configObj.cropFrame('[Function-1]')
             self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')
 
    def master_focus_split_geometry(self, stackMaster, config_prefix, split=False, focus=True, native=True):
        """focusing master and producing geometry files"""
        configName = os.path.join(self.configDir, config_prefix + stackMaster)
        configObj = config(configName)
        configObj.configure(self)
        configObj.slcDir = os.path.join(self.slcDir,stackMaster)
        configObj.geometryDir = os.path.join(self.workDir,self.stack_folder, 'geom_master')

        counter=1
        if focus:
            configObj.focus('[Function-{0}]'.format(counter))
            counter += 1
       
        configObj.nativeDoppler = focus or native
        configObj.topo('[Function-{0}]'.format(counter))
        counter += 1

        if split:
            configObj.slc = os.path.join(configObj.slcDir,stackMaster+self.raw_string+'.slc')
            configObj.outDir = configObj.slcDir
            configObj.shelve = os.path.join(configObj.slcDir, 'data')
            configObj.splitRangeSpectrum('[Function-{0}]'.format(counter))
            counter += 1

        # generate water mask in radar coordinates
        configObj.latFile = os.path.join(self.workDir, 'geom_master/lat.rdr')
        configObj.lonFile = os.path.join(self.workDir, 'geom_master/lon.rdr')
        configObj.waterMaskFile = os.path.join(self.workDir, 'geom_master/waterMask.rdr')
        configObj.createWaterMask('[Function-{0}]'.format(counter))
        counter += 1

        configObj.finalize()
        del configObj
        self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')


    def slaves_focus_split(self, slaveDates, config_prefix, split=False, focus=True, native=True):
        for slave in slaveDates:
             configName = os.path.join(self.configDir, config_prefix + '_'+slave)
             configObj = config(configName)
             configObj.configure(self)
             configObj.slcDir = os.path.join(self.slcDir,slave)

             counter=1
             if focus:
                 configObj.focus('[Function-{0}]'.format(counter))
                 counter += 1

             if split:
                  configObj.slc = os.path.join(configObj.slcDir,slave + self.raw_string + '.slc')
                  configObj.outDir = configObj.slcDir
                  configObj.shelve = os.path.join(configObj.slcDir, 'data')
                  configObj.splitRangeSpectrum('[Function-{0}]'.format(counter))

             configObj.finalize()
             del configObj
             self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')

    def slaves_geo2rdr_resampleSlc(self, stackMaster, slaveDates, config_prefix, native=True):

        for slave in slaveDates:
             configName = os.path.join(self.configDir,config_prefix+slave) 
             configObj = config(configName)
             configObj.configure(self)
             configObj.masterSlc = os.path.join(self.slcDir, stackMaster)
             configObj.slaveSlc = os.path.join(self.slcDir, slave)
             configObj.geometryDir = os.path.join(self.workDir, self.stack_folder,'geom_master')
             configObj.offsetDir = os.path.join(self.workDir, 'offsets',slave)
             configObj.nativeDoppler = native

             configObj.geo2rdr('[Function-1]')
             configObj.coregSlaveSlc = os.path.join(self.workDir, 'coregSLC','Coarse',slave)
             configObj.resampleSlc('[Function-2]')
             configObj.finalize()
             del configObj
             self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')


    def refineSlaveTiming_singleMaster(self, stackMaster, slaveDates, config_prefix):
  
        for slave in slaveDates:
             configName = os.path.join(self.configDir,config_prefix+slave)
             configObj = config(configName)  
             configObj.configure(self)
             configObj.masterSlc = os.path.join(self.slcDir, stackMaster,stackMaster+self.raw_string+'.slc')
             configObj.slaveSlc = os.path.join(self.workDir, 'coregSLC','Coarse', slave,slave +'.slc')
             configObj.masterMetaData = os.path.join(self.slcDir, stackMaster)
             configObj.slaveMetaData = os.path.join(self.slcDir, slave)
             configObj.outfile = os.path.join(self.workDir, 'offsets', slave ,'misreg')
             configObj.refineSlaveTiming('[Function-1]')
             configObj.finalize()
             self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')


    def refineSlaveTiming_Network(self, pairs, stackMaster, slaveDates,  config_prefix):
  
        for pair in pairs:
             configName = os.path.join(self.configDir,config_prefix + pair[0] + '_' + pair[1])
             configObj = config(configName)
             configObj.configure(self)

             if pair[0] == stackMaster:
                  configObj.masterSlc = os.path.join(self.slcDir,stackMaster,stackMaster+self.raw_string+'.slc')
             else:
                 configObj.masterSlc = os.path.join(self.workDir, 'coregSLC','Coarse', pair[0]  , pair[0] + '.slc')

             if pair[1] == stackMaster:
                 configObj.slaveSlc = os.path.join(self.slcDir,stackMaster, stackMaster+self.raw_string+'.slc')
             else:
                 configObj.slaveSlc = os.path.join(self.workDir, 'coregSLC','Coarse', pair[1], pair[1] + '.slc')

             configObj.masterMetaData = os.path.join(self.slcDir, pair[0])
             configObj.slaveMetaData = os.path.join(self.slcDir, pair[1])
             configObj.outfile = os.path.join(self.workDir, 'refineSlaveTiming','pairs', pair[0] + '_' + pair[1] ,'misreg')
             configObj.refineSlaveTiming('[Function-1]')
             configObj.finalize()
             del configObj
             self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')


    def denseOffsets_Network(self, pairs, stackMaster, slaveDates, config_prefix):

        for pair in pairs:
             configName = os.path.join(self.configDir,config_prefix + pair[0] + '_' + pair[1])
             configObj = config(configName)
             configObj.configure(self)


             if pair[0] == stackMaster:
                  configObj.masterSlc = os.path.join(self.slcDir,stackMaster , stackMaster+self.raw_string + '.slc')
             else:
                  configObj.masterSlc = os.path.join(self.workDir, self.stack_folder, 'SLC', pair[0] , pair[0] + '.slc')

             if pair[1] == stackMaster:
                  configObj.slaveSlc = os.path.join(self.slcDir,stackMaster, stackMaster+self.raw_string+'.slc')
             else:
                  configObj.slaveSlc = os.path.join(self.workDir, self.stack_folder,'SLC', pair[1] , pair[1] + '.slc')

             configObj.outfile = os.path.join(self.workDir, self.dense_offsets_folder,'pairs',  pair[0] + '_' + pair[1] ,  pair[0] + '_' + pair[1])
             configObj.denseOffsets('[Function-1]')
             configObj.denseOffset = configObj.outfile + '.bil'
             configObj.snr = configObj.outfile + '_snr.bil'
             configObj.outDir = os.path.join(self.workDir, self.dense_offsets_folder,'pairs' , pair[0] + '_' + pair[1])
             configObj.filterOffsets('[Function-2]')
             configObj.finalize()
             del configObj
             self.runf.write(self.text_cmd + 'stripmapWrapper.py -c '+ configName+'\n')

  
    def invertMisregPoly(self):

        pairDirs = os.path.join(self.workDir, 'refineSlaveTiming/pairs/')
        dateDirs = os.path.join(self.workDir, 'refineSlaveTiming/dates/')
        cmd = self.text_cmd + 'invertMisreg.py -i ' + pairDirs + ' -o ' + dateDirs
        self.runf.write(cmd + '\n')
        

    def  invertDenseOffsets(self):

        pairDirs = os.path.join(self.workDir, self.dense_offsets_folder, 'pairs')
        dateDirs = os.path.join(self.workDir, self.dense_offsets_folder, 'dates')
        cmd = self.text_cmd + 'invertOffsets.py -i ' + pairDirs + ' -o ' + dateDirs
        self.runf.write(cmd + '\n')

    def rubbersheet(self, slaveDates, config_prefix):

        for slave in slaveDates:
             configName = os.path.join(self.configDir, config_prefix+slave)
             configObj = config(configName)
             configObj.configure(self)
             configObj.geometry_azimuth_offset = os.path.join(self.workDir, 'offsets' , slave , 'azimuth.off')
             configObj.dense_offset = os.path.join(self.workDir,self.dense_offsets_folder,'dates', slave , slave + '.bil')
             configObj.snr = os.path.join(self.workDir,self.dense_offsets_folder,'dates' , slave , slave + '_snr.bil')
             configObj.output_azimuth_offset = 'azimuth.off'
             configObj.output_directory = os.path.join(self.workDir,self.dense_offsets_folder,'dates', slave)
             configObj.rubbersheet('[Function-1]')
             configObj.finalize()
             self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')
   

    def resampleOffset(self, slaveDates, config_prefix):

        for slave in slaveDates:
             configName = os.path.join(self.configDir, config_prefix+slave)
             configObj = config(configName)
             configObj.configure(self)
             configObj.targetFile = os.path.join(self.workDir, 'offsets/'+slave + '/azimuth.off')
             configObj.input = os.path.join(self.workDir,self.dense_offsets_folder,'dates',slave  , slave + '.bil')
             configObj.output = os.path.join(self.workDir,self.dense_offsets_folder,'dates',slave, 'azimuth.off')
             configObj.resampleOffset('[Function-1]')
             configObj.finalize()
             self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')
  

    def  replaceOffsets(self, slaveDates):

        dateDirs = os.path.join(self.workDir, self.dense_offsets_folder,'dates')
        for slave in slaveDates:
             geometryOffset = os.path.join(self.workDir, 'offsets', slave  , 'azimuth.off')
             geometryOnlyOffset = os.path.join(self.workDir, 'offsets' , slave , 'azimuth.off.geometry')
             rubberSheeted = os.path.join(self.workDir,self.dense_offsets_folder,'dates' , slave , 'azimuth.off')
             cmd = self.text_cmd + 'mv ' + geometryOffset + ' ' + geometryOnlyOffset
             cmd = cmd + '; mv ' + rubberSheeted + ' ' + geometryOffset
             self.runf.write(cmd + '\n')

  
    def gridBaseline(self, stackMaster, slaveDates, config_prefix, split=False):
        for slave in slaveDates:
            configName = os.path.join(self.configDir, config_prefix+slave)
            configObj = config(configName)
            configObj.coregSlaveSlc = os.path.join(self.workDir,self.stack_folder,'SLC',slave)
            configObj.baselineGridFile = os.path.join(self.workDir, self.stack_folder,'baselines', slave,slave )
            configObj.baselineGrid('[Function-1]')
            configObj.finalize()
            self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')
            
        # also add the master to be included
        configName = os.path.join(self.configDir, config_prefix+stackMaster)
        configObj = config(configName)
        configObj.coregSlaveSlc = os.path.join(self.workDir,self.stack_folder,'SLC',stackMaster)        
        configObj.baselineGridFile = os.path.join(self.workDir, self.stack_folder,'baselines', stackMaster,stackMaster )
        configObj.baselineGrid('[Function-1]')
        configObj.finalize()
        self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')

    def slaves_fine_resampleSlc(self, stackMaster, slaveDates, config_prefix, split=False):
        # copy over the master into the final SLC folder as well
        self.runf.write(self.text_cmd + ' masterStackCopy.py -i ' + os.path.join(self.slcDir,stackMaster, stackMaster+self.raw_string + '.slc') + ' -o ' + os.path.join(self.workDir, self.stack_folder,'SLC', stackMaster, stackMaster+'.slc' )+ '\n')
        
        # now resample each of the slaves to the master geometry
        for slave in slaveDates:
            configName = os.path.join(self.configDir, config_prefix+slave)
            configObj = config(configName)
            configObj.configure(self)
            configObj.masterSlc = os.path.join(self.slcDir, stackMaster)
            configObj.slaveSlc = os.path.join(self.slcDir, slave)
            configObj.offsetDir = os.path.join(self.workDir, 'offsets',slave)
            configObj.coregSlaveSlc = os.path.join(self.workDir,self.stack_folder,'SLC',slave) 
            configObj.misreg = os.path.join(self.workDir, 'refineSlaveTiming','dates', slave, 'misreg')
            configObj.resampleSlc('[Function-1]')
            
            if split:
                configObj.slaveSlc = os.path.join(self.slcDir, slave,'LowBand')
                configObj.coregSlaveSlc = os.path.join(self.workDir, self.stack_folder,'SLC',  slave, 'LowBand')
                configObj.resampleSlc_subband('[Function-2]')
                
                configObj.slaveSlc = os.path.join(self.slcDir, slave,'HighBand')
                configObj.coregSlaveSlc = os.path.join(self.workDir,self.stack_folder, 'SLC', slave, 'HighBand')
                configObj.resampleSlc_subband('[Function-3]')
            configObj.finalize()
            self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')

    def igrams_network(self,  pairs, acuisitionDates, stackMaster,low_or_high, config_prefix):

        for pair in pairs:
             configName = os.path.join(self.configDir,config_prefix + pair[0] + '_' + pair[1])
             configObj = config(configName)
             configObj.configure(self)
             
             if pair[0] == stackMaster:
                  configObj.masterSlc = os.path.join(self.slcDir,stackMaster + low_or_high + stackMaster+self.raw_string +'.slc')
             else:
                  configObj.masterSlc = os.path.join(self.workDir, self.stack_folder, 'SLC',  pair[0] + low_or_high + pair[0] + '.slc')

             if pair[1] == stackMaster:
                  configObj.slaveSlc = os.path.join(self.slcDir,stackMaster + low_or_high + stackMaster+self.raw_string+'.slc')
             else:
                  configObj.slaveSlc = os.path.join(self.workDir, self.stack_folder, 'SLC',  pair[1] + low_or_high + pair[1] + '.slc')

             configObj.outDir = os.path.join(self.workDir, 'Igrams' + low_or_high + 
                         pair[0] + '_'  + pair[1] +'/'+pair[0] + '_'  + pair[1])
             configObj.generateIgram('[Function-1]')

             configObj.igram = configObj.outDir+'.int'
             configObj.filtIgram = os.path.dirname(configObj.outDir) + '/filt_' + pair[0] + '_'  + pair[1] + '.int'
             configObj.coherence = os.path.dirname(configObj.outDir) + '/filt_' + pair[0] + '_'  + pair[1] + '.cor'
             #configObj.filtStrength = filtStrength
             configObj.filterCoherence('[Function-2]')

             configObj.igram = configObj.filtIgram
             configObj.unwIfg = os.path.dirname(configObj.outDir) + '/filt_' + pair[0] + '_'  + pair[1] 
             configObj.noMCF = noMCF
             configObj.master = os.path.join(self.slcDir,stackMaster +'/data') 
             configObj.defoMax = defoMax
             configObj.unwrap('[Function-3]')

             configObj.finalize()
             self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')


    def dispersive_nonDispersive(self, pairs, acuisitionDates, stackMaster, 
                           lowBand, highBand, config_prefix):
        for pair in pairs:
             configName = os.path.join(self.configDir,config_prefix + pair[0] + '_' + pair[1])
             configObj = config(configName) 
             configObj.configure(self)
             configObj.lowBandIgram = os.path.join(self.workDir, 'Igrams' + lowBand + pair[0] + '_'  + pair[1]                      + '/filt_'+pair[0] + '_'  + pair[1])    
             configObj.highBandIgram = os.path.join(self.workDir, 'Igrams' + highBand + pair[0] + '_'  + pair[1] 
                   + '/filt_'+pair[0] + '_'  + pair[1])
             configObj.lowBandCor = os.path.join(self.workDir, 'Igrams' + lowBand + pair[0] + '_'  + pair[1]
                   + '/filt_'+pair[0] + '_'  + pair[1] + '.cor')
             configObj.highBandCor = os.path.join(self.workDir, 'Igrams' + highBand + pair[0] + '_'  + pair[1]
                   + '/filt_'+pair[0] + '_'  + pair[1] + '.cor')
             configObj.lowBandShelve = os.path.join(self.slcDir,pair[0] + lowBand  + 'data') 
             configObj.highBandShelve = os.path.join(self.slcDir,pair[0] + highBand  + 'data')   
             configObj.outDir = os.path.join(self.workDir, 'Ionosphere/'+pair[0]+'_'+pair[1])
             configObj.estimateDispersive('[Function-1]')
             configObj.finalize()
             self.runf.write(self.text_cmd+'stripmapWrapper.py -c '+ configName+'\n')

    def finalize(self):
        self.runf.close() 
        writeJobFile(self.run_outname)


'''

class workflow(object):
    """
       A class representing a run which may contain several functions
    """
    #def __init__(self):

    def configure(self,inps, runName):
        for k in inps.__dict__.keys():
            setattr(self, k, inps.__dict__[k])

    def  

'''

##############################

def baselinePair(baselineDir, master, slave,doBaselines=True):
    
    if doBaselines: # open files to calculate baselines
        try:
            mdb = shelve.open( os.path.join(master, 'raw'), flag='r')
            sdb = shelve.open( os.path.join(slave, 'raw'), flag='r')
        except:
            mdb = shelve.open( os.path.join(master, 'data'), flag='r')
            sdb = shelve.open( os.path.join(slave, 'data'), flag='r')

        mFrame = mdb['frame']
        sFrame = sdb['frame']


        bObj = Baseline()
        bObj.configure()
        bObj.wireInputPort(name='masterFrame', object=mFrame)
        bObj.wireInputPort(name='slaveFrame', object=sFrame)
        bObj.baseline()    # calculate baseline from orbits
        pBaselineBottom = bObj.pBaselineBottom
        pBaselineTop = bObj.pBaselineTop
    else:       # set baselines to zero if not calculated
        pBaselineBottom = 0.0
        pBaselineTop = 0.0
        
    baselineOutName = os.path.basename(master) + "_" + os.path.basename(slave) + ".txt"
    f = open(os.path.join(baselineDir, baselineOutName) , 'w')
    f.write("PERP_BASELINE_BOTTOM " + str(pBaselineBottom) + '\n')
    f.write("PERP_BASELINE_TOP " + str(pBaselineTop) + '\n')
    f.close()
    print('Baseline at top/bottom: %f %f'%(pBaselineTop,pBaselineBottom))
    return (pBaselineTop+pBaselineBottom)/2.

def baselineStack(inps,stackMaster,acqDates,doBaselines=True):
    from collections import OrderedDict
    baselineDir = os.path.join(inps.workDir,'baselines')
    if not os.path.exists(baselineDir):
        os.makedirs(baselineDir)
    baselineDict = OrderedDict()
    timeDict = OrderedDict()
    datefmt = '%Y%m%d'
    t0 = datetime.datetime.strptime(stackMaster, datefmt)
    master = os.path.join(inps.slcDir, stackMaster)
    for slv in acqDates:
        if slv != stackMaster:
            slave = os.path.join(inps.slcDir, slv)
            baselineDict[slv]=baselinePair(baselineDir, master, slave, doBaselines)
            t = datetime.datetime.strptime(slv, datefmt)
            timeDict[slv] = t - t0
        else:
            baselineDict[stackMaster] = 0.0
            timeDict[stackMaster] = datetime.timedelta(0.0)

    return baselineDict, timeDict

def selectPairs(inps,stackMaster, slaveDates, acuisitionDates,doBaselines=True):

    baselineDict, timeDict = baselineStack(inps, stackMaster, acuisitionDates,doBaselines)
    for slave in slaveDates:
       print (slave,' : ' , baselineDict[slave])
    numDates = len(acuisitionDates)
    pairs = []
    for i in range(numDates-1):
       for j in range(i+1,numDates):
          db = np.abs(baselineDict[acuisitionDates[j]] - baselineDict[acuisitionDates[i]])
          dt  = np.abs(timeDict[acuisitionDates[j]].days - timeDict[acuisitionDates[i]].days)
          if (db < inps.dbThr) and (dt < inps.dtThr):
              pairs.append((acuisitionDates[i],acuisitionDates[j]))

    plotNetwork(baselineDict, timeDict, pairs,os.path.join(inps.workDir,'pairs.pdf'))
    return pairs 


def plotNetwork(baselineDict, timeDict, pairs,save_name='pairs.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    datefmt='%Y%m%d'
    fig1 = plt.figure(1)
    ax1=fig1.add_subplot(111)

    ax1.cla()
    for ni in range(len(pairs)):
#        ax1.plot(np.array([timeDict[pairs[ni][0]].days,timeDict[pairs[ni][1]].days]), 
         ax1.plot([datetime.datetime.strptime(pairs[ni][0],datefmt), datetime.datetime.strptime(pairs[ni][1], datefmt)], 
                 np.array([baselineDict[pairs[ni][0]],baselineDict[pairs[ni][1]]]),
                 '-ko',lw=1, ms=4, alpha=0.7, mfc='r')
  
    

    myFmt = mdates.DateFormatter('%Y-%m')
    ax1.xaxis.set_major_formatter(myFmt)

    plt.title('Baseline plot')
    plt.xlabel('Time')
    plt.ylabel('Perp. Baseline')
    plt.tight_layout()


    plt.savefig(save_name)

    ###Check degree of each SLC
    datelist = [k for k,v in list(timeDict.items())]
    connMat = np.zeros((len(pairs), len(timeDict)))
    for ni in range(len(pairs)):
        connMat[ni, datelist.index(pairs[ni][0])] = 1.0
        connMat[ni, datelist.index(pairs[ni][1])] = -1.0

    slcSum = np.sum( np.abs(connMat), axis=0)
    minDegree = np.min(slcSum)

    print('##################')
    print('SLCs with min degree connection of {0}'.format(minDegree))
    for ii in range(slcSum.size):
        if slcSum[ii] == minDegree:
            print(datelist[ii])
    print('##################')
    
    if np.linalg.matrix_rank(connMat) != (len(timeDict) - 1):
        raise Exception('The network for cascading coregistration   is not connected')

def writeJobFile(runFile):

  
  jobName = runFile + ".job"
  dirName = os.path.dirname(runFile)
  with open(runFile) as ff:
    nodes = len(ff.readlines())
  if nodes >maxNodes:
     nodes = maxNodes

  f = open (jobName,'w')
  f.write('#!/bin/bash '+ '\n')
  f.write('#PBS -N Parallel_GNU'+ '\n')
  f.write('#PBS -l nodes=' + str(nodes) + '\n')

  jobTxt='''#PBS -V
#PBS -l walltime=05:00:00
#PBS -q default

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

echo Running on host `hostname`
echo Time is `date`

### Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS cpus

# Tell me which nodes it is run on
echo " "
echo This jobs runs on the following processors:
echo `cat $PBS_NODEFILE`
echo " "

# 
# Run the parallel with the nodelist and command file
#

'''
  f.write(jobTxt+ '\n')
  f.write('parallel --sshloginfile $PBS_NODEFILE  -a '+runFile+'\n')
  f.write('')
  f.close()


def main(iargs=None):
    '''nothing to do'''

if __name__ == "__main__":
       
  # Main engine  
  main()
       

