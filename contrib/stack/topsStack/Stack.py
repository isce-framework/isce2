#!/usr/bin/env python3
########################
#Author: Heresh Fattahi

#######################

import os, glob, sys
import datetime


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
        self.misreg_az = None
        self.misreg_rng = None
        self.multilook_tool = None
        self.no_data_value = None

    def Sentinel1_TOPS(self,function):
        self.f.write('##########################'+'\n')
        self.f.write(function+'\n')
        self.f.write('Sentinel1_TOPS : ' + '\n')
        self.f.write('dirname : ' + self.dirName+'\n')
        self.f.write('swaths : ' + self.swaths+'\n')

        if self.orbit_type == 'precise':
            self.f.write('orbitdir : '+self.orbit_dirname+'\n')
        else:
            self.f.write('orbit : '+self.orbit_file+'\n')

        self.f.write('outdir : ' + self.outDir + '\n')
        self.f.write('auxdir : ' + self.aux_dirname + '\n')
        if self.bbox is not None:
            self.f.write('bbox : ' + self.bbox + '\n')
        self.f.write('pol : ' + self.polarization + '\n')
        self.f.write('##########################' + '\n')

    def computeAverageBaseline(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function + '\n')
        self.f.write('computeBaseline : '+'\n')
        self.f.write('reference : ' + self.reference + '\n')
        self.f.write('secondary : ' + self.secondary + '\n')
        self.f.write('baseline_file : ' + self.baselineFile + '\n')

    def computeGridBaseline(self, function):
        self.f.write('##########################'+'\n')
        self.f.write(function + '\n')
        self.f.write('baselineGrid : '+'\n')
        self.f.write('reference : ' + self.reference + '\n')
        self.f.write('secondary : ' + self.secondary + '\n')
        self.f.write('baseline_file : ' + self.baselineFile + '\n')

    def topo(self,function):
        self.f.write('##########################'+'\n')
        self.f.write('#Call topo to produce reference geometry files'+'\n')
        self.f.write(function + '\n')
        self.f.write('topo : ' + '\n')
        self.f.write('reference : ' + self.outDir + '\n')
        self.f.write('dem : ' + self.dem + '\n')
        self.f.write('geom_referenceDir : ' + self.geom_referenceDir + '\n')
        self.f.write('numProcess : ' + str(self.numProcess4topo) + '\n')
        self.f.write('##########################' + '\n')

    def geo2rdr(self,function):
        self.f.write('##########################' + '\n')
        self.f.write(function + '\n')
        self.f.write('geo2rdr :' + '\n')
        self.f.write('secondary : ' + self.secondaryDir + '\n')
        self.f.write('reference : ' + self.referenceDir + '\n')
        self.f.write('geom_referenceDir : ' + self.geom_reference + '\n')
        self.f.write('coregSLCdir : ' + self.coregSecondaryDir + '\n')
        self.f.write('overlap : ' + self.overlapTrueOrFalse + '\n')
        if self.useGPU:
            self.f.write('useGPU : True \n')
        else:
            self.f.write('useGPU : False\n')
        if self.misreg_az is not None:
            self.f.write('azimuth_misreg : ' + self.misreg_az + '\n')
        if self.misreg_rng is not None:
            self.f.write('range_misreg : ' + self.misreg_rng + '\n')

    def resamp_withCarrier(self,function):
        self.f.write('##########################' + '\n')
        self.f.write(function + '\n')
        self.f.write('resamp_withCarrier : ' + '\n')
        self.f.write('secondary : ' + self.secondaryDir + '\n')
        self.f.write('reference : ' + self.referenceDir + '\n')
       #self.f.write('interferogram_prefix :' + self.interferogram_prefix + '\n')
        self.f.write('coregdir : ' + self.coregSecondaryDir + '\n')
        self.f.write('overlap : ' + self.overlapTrueOrFalse + '\n')
        if self.misreg_az is not None:
            self.f.write('azimuth_misreg : ' + self.misreg_az + '\n')
        if self.misreg_rng is not None:
            self.f.write('range_misreg : ' + self.misreg_rng + '\n')

    def generateIgram(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('generateIgram : ' + '\n')
        self.f.write('reference : ' + self.referenceDir + '\n')
        self.f.write('secondary : ' + self.secondaryDir + '\n')
        self.f.write('interferogram : ' + self.interferogramDir + '\n')
        self.f.write('flatten : ' + self.flatten + '\n')
        self.f.write('interferogram_prefix : ' + self.interferogram_prefix +'\n')
        self.f.write('overlap : ' + self.overlapTrueOrFalse + '\n')
        if self.misreg_az is not None:
            self.f.write('azimuth_misreg : ' + self.misreg_az + '\n')
        if self.misreg_rng is not None:
            self.f.write('range_misreg : ' + self.misreg_rng + '\n')
        self.f.write('###################################' + '\n')

    def overlap_withDEM(self,function):

        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('overlap_withDEM : '+'\n')
        self.f.write('interferogram : ' + self.interferogramDir +'\n')
        self.f.write('reference_dir : ' + self.referenceDir+'\n')
        self.f.write('secondary_dir : ' + self.secondaryDir+'\n')
        self.f.write('overlap_dir : ' + self.overlapDir+'\n')

    def azimuthMisreg(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('estimateAzimuthMisreg : ' + '\n')
        self.f.write('overlap_dir : ' + self.overlapDir + '\n')
        self.f.write('out_azimuth : ' + self.misregFile + '\n')
        self.f.write('coh_threshold : ' + self.esdCoherenceThreshold + '\n')
        self.f.write('plot : ' + self.plot + '\n')

    def rangeMisreg(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('estimateRangeMisreg : ' + '\n')
        self.f.write('reference : ' + self.referenceDir + '\n')
        self.f.write('secondary : ' + self.secondaryDir + '\n')
        self.f.write('out_range : ' + self.misregFile + '\n')
        self.f.write('snr_threshold : ' + self.snrThreshold + '\n')

    def mergeBurst(self, function):

        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('mergeBursts : ' + '\n')
        try:
            self.f.write('stack : ' + self.stack +'\n')
        except:
            pass

        self.f.write('inp_reference : ' + self.reference +'\n')
        self.f.write('dirname : ' + self.dirName + '\n')
        self.f.write('name_pattern : ' + self.namePattern + '\n')
        self.f.write('outfile : ' + self.mergedFile + '\n')
        self.f.write('method : ' + self.mergeBurstsMethod + '\n')
        self.f.write('aligned : ' + self.aligned + '\n')
        self.f.write('valid_only : ' + self.validOnly + '\n')
        self.f.write('use_virtual_files : ' + self.useVirtualFiles + '\n')
        self.f.write('multilook : ' + self.multiLook + '\n')
        self.f.write('range_looks : ' + self.rangeLooks + '\n')
        self.f.write('azimuth_looks : ' + self.azimuthLooks + '\n')
        if self.multilook_tool:
            self.f.write('multilook_tool : ' + self.multilook_tool + '\n')
        if self.no_data_value is not None:
            self.f.write('no_data_value : ' + self.no_data_value + '\n')

    def mergeSwaths(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('mergeSwaths : ' + '\n')
        self.f.write('input : ' + self.inputDirs  +  '\n')
        self.f.write('file : ' + self.fileName + '\n')
        self.f.write('metadata : ' + self.metadata + '\n')
        self.f.write('output : ' + self.outDir +'\n')

    def multiLook(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('looks_withDEM : ' + '\n')
        self.f.write('input : ' + self.input + '\n')
        self.f.write('output : ' + self.output + '\n')
        self.f.write('range : ' + self.rangeLooks + '\n')
        self.f.write('azimuth : ' + self.azimuthLooks + '\n')

    def FilterAndCoherence(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('FilterAndCoherence : ' + '\n')
        self.f.write('input : ' + self.input + '\n')
        self.f.write('filt : ' + self.filtName + '\n')
        self.f.write('coh : ' + self.cohName + '\n')
        self.f.write('strength : ' + self.filtStrength + '\n')
        self.f.write('slc1 : ' + self.slc1 + '\n')
        self.f.write('slc2 : ' + self.slc2 + '\n')
        self.f.write('complex_coh : '+ self.cpxCohName + '\n')
        self.f.write('range_looks : ' + self.rangeLooks + '\n')
        self.f.write('azimuth_looks : ' + self.azimuthLooks + '\n')

    def unwrap(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('unwrap : ' + '\n')
        self.f.write('ifg : ' + self.ifgName + '\n')
        self.f.write('unw : ' + self.unwName + '\n')
        self.f.write('coh : ' + self.cohName + '\n')
        self.f.write('nomcf : ' + self.noMCF + '\n')
        self.f.write('reference : ' + self.reference + '\n')
        self.f.write('defomax : ' + self.defoMax + '\n')
        self.f.write('rlks : ' + self.rangeLooks + '\n')
        self.f.write('alks : ' + self.azimuthLooks + '\n')
        if self.rmFilter:
            self.f.write('rmfilter : True \n')
        else:
            self.f.write('rmfilter : False\n')
        self.f.write('method : ' + self.unwMethod + '\n')

    def unwrapSnaphu(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('unwrapSnaphu : ' + '\n')
        self.f.write('ifg : ' + self.ifgName + '\n')
        self.f.write('unw : ' + self.unwName + '\n')
        self.f.write('coh : ' + self.cohName + '\n')
        self.f.write('nomcf : ' + self.noMCF + '\n')
        self.f.write('reference : ' + self.reference + '\n')
        self.f.write('defomax : ' + self.defoMax + '\n')
        self.f.write('rlks : ' + self.rangeLooks + '\n')
        self.f.write('alks : ' + self.azimuthLooks + '\n')
        self.f.write('numProcess : ' + self.numProcess + '\n')

    def denseOffset(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')

        # CPU or GPU
        self.f.write('denseOffsets : ' + '\n')
        #self.f.write('DenseOffsets : ' + '\n')
        #self.f.write('cuDenseOffsets : ' + '\n')
        self.f.write('reference : ' + self.reference + '\n')
        self.f.write('secondary : ' + self.secondary + '\n')
        self.f.write('outprefix : ' + self.output + '\n')

        #self.f.write('ww : 256\n')
        #self.f.write('wh : 128\n')

    def subband_and_resamp(self, function):
        self.f.write('##########################' + '\n')
        self.f.write(function + '\n')
        self.f.write('subband_and_resamp : ' + '\n')
        self.f.write('reference : ' + self.reference + '\n')
        self.f.write('secondary : ' + self.secondary + '\n')
        self.f.write('coregdir : ' + self.coregdir + '\n')
        self.f.write('azimuth_misreg : ' + self.azimuth_misreg + '\n')
        self.f.write('range_misreg : ' + self.range_misreg + '\n')

    def subband(self, function):
        self.f.write('##########################' + '\n')
        self.f.write(function + '\n')
        self.f.write('subband : ' + '\n')
        self.f.write('directory : ' + self.reference + '\n')

    def generateIgram_ion(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('generateIgram : ' + '\n')
        self.f.write('reference : ' + self.reference + '\n')
        self.f.write('reference_suffix : ' + self.reference_suffix + '\n')
        self.f.write('secondary : ' + self.secondary + '\n')
        self.f.write('secondary_suffix : ' + self.secondary_suffix + '\n')
        self.f.write('interferogram : ' + self.interferogram + '\n')
        self.f.write('flatten : ' + self.flatten + '\n')
        self.f.write('interferogram_prefix : ' + self.interferogram_prefix +'\n')
        self.f.write('overlap : ' + self.overlap + '\n')

    def mergeBurstsIon(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('mergeBurstsIon : ' + '\n')
        self.f.write('reference : ' + self.reference + '\n')
        self.f.write('stack : ' + self.stack + '\n')
        self.f.write('dirname : ' + self.dirname + '\n')
        self.f.write('name_pattern : ' + self.name_pattern + '\n')
        self.f.write('outfile : ' + self.outfile + '\n')
        self.f.write('nrlks : ' + '{}'.format(self.nrlks) + '\n')
        self.f.write('nalks : ' + '{}'.format(self.nalks) + '\n')
        self.f.write('nrlks0 : ' + '{}'.format(self.nrlks0) +'\n')
        self.f.write('nalks0 : ' + '{}'.format(self.nalks0) + '\n')
        if self.rvalid is not None:
            self.f.write('rvalid : ' + '{}'.format(self.rvalid) + '\n')
        if self.swath is not None:
            self.f.write('swath : ' + self.swath + '\n')

    def coherenceIon(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('coherenceIon : ' + '\n')
        self.f.write('lower : ' + self.lower + '\n')
        self.f.write('upper : ' + self.upper + '\n')
        self.f.write('coherence : ' + self.coherence + '\n')
        if self.nrlks is not None:
            self.f.write('nrlks : ' + '{}'.format(self.nrlks) + '\n')
        if self.nalks is not None:
            self.f.write('nalks : ' + '{}'.format(self.nalks) + '\n')

    def lookUnwIon(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('lookUnwIon : ' + '\n')
        self.f.write('unw : ' + self.unw + '\n')
        self.f.write('cor : ' + self.cor + '\n')
        self.f.write('output : ' + self.output + '\n')
        self.f.write('nrlks : ' + '{}'.format(self.nrlks) + '\n')
        self.f.write('nalks : ' + '{}'.format(self.nalks) + '\n')

    def filtIon(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('filtIon : ' + '\n')
        self.f.write('input : ' + self.input + '\n')
        self.f.write('coherence : ' + self.coherence + '\n')
        self.f.write('output : ' + self.output + '\n')
        self.f.write('win_min : ' + '{}'.format(self.win_min) + '\n')
        self.f.write('win_max : ' + '{}'.format(self.win_max) + '\n')




    def write_wrapper_config2run_file(self, configName, line_cnt, numProcess = 1):
        # dispassionate list of commands for single process
        if numProcess == 1:
            self.runf.write(self.text_cmd + 'SentinelWrapper.py -c ' + configName + '\n')
        # aggregate background commands between wait blocks for speed gains
        elif numProcess > 1:
            self.runf.write(self.text_cmd + 'SentinelWrapper.py -c ' + configName + ' &\n')
            if line_cnt == numProcess:
                self.runf.write('wait\n\n')
                line_cnt = 0
        return line_cnt

    def finalize(self):
        self.f.close()


class ionParamUsr(object):
    '''A class containing parameters for ionosphere estimation specified by user
       while considerBurstProperties is not availavle for stack processing,
       ionParam still has parameters associated with considerBurstProperties for bookkeeping.
    '''

    def __init__(self, usrInput):
        # usrInput: usrInput txt file
        self.usrInput = usrInput

    def configure(self):
        #default values same as topsApp.py
        #only ION_applyIon is removed, compared with topsApp.py
        self.ION_doIon = False
        self.ION_considerBurstProperties = False

        self.ION_ionHeight = 200.0
        self.ION_ionFit = True
        self.ION_ionFilteringWinsizeMax = 200
        self.ION_ionFilteringWinsizeMin = 100
        self.ION_ionshiftFilteringWinsizeMax = 150
        self.ION_ionshiftFilteringWinsizeMin = 75
        self.ION_azshiftFlag = 1

        self.ION_maskedAreas = None

        self.ION_numberAzimuthLooks = 50
        self.ION_numberRangeLooks = 200
        self.ION_numberAzimuthLooks0 = 10
        self.ION_numberRangeLooks0 = 40


        #get above parameters from usr input
        with open(self.usrInput, 'r') as f:
            lines = f.readlines()
        
        for x in lines:
            x = x.strip()
            if x == '' or x.strip().startswith('#'):
                continue
            else:
                x2 = x.split(':')
                if 'do ionosphere correction' == x2[0].strip():
                    self.ION_doIon = eval(x2[1].strip().capitalize())
                if 'consider burst properties in ionosphere computation' == x2[0].strip():
                    self.ION_considerBurstProperties = eval(x2[1].strip().capitalize())

                if 'height of ionosphere layer in km' == x2[0].strip():
                    self.ION_ionHeight = float(x2[1].strip())
                if 'apply polynomial fit before filtering ionosphere phase' == x2[0].strip():
                    self.ION_ionFit = eval(x2[1].strip().capitalize())
                if 'maximum window size for filtering ionosphere phase' == x2[0].strip():
                    self.ION_ionFilteringWinsizeMax = int(x2[1].strip())
                if 'minimum window size for filtering ionosphere phase' == x2[0].strip():
                    self.ION_ionFilteringWinsizeMin = int(x2[1].strip())
                if 'maximum window size for filtering ionosphere azimuth shift' == x2[0].strip():
                    self.ION_ionshiftFilteringWinsizeMax = int(x2[1].strip())
                if 'minimum window size for filtering ionosphere azimuth shift' == x2[0].strip():
                    self.ION_ionshiftFilteringWinsizeMin = int(x2[1].strip())
                if 'correct phase error caused by ionosphere azimuth shift' == x2[0].strip():
                    self.ION_azshiftFlag = int(x2[1].strip())

                if 'areas masked out in ionospheric phase estimation' == x2[0].strip():
                    if x2[1].strip().capitalize() == 'None':
                        self.ION_maskedAreas = None
                    else:
                        self.ION_maskedAreas = []
                        x3 = x2[1].replace('[', '').replace(']', '').split(',')
                        if len(x3)%4 != 0:
                            raise Exception('there must be four elements for each area.')
                        else:
                            narea = int(len(x3)/4)
                            for i in range(narea):
                                self.ION_maskedAreas.append([int(x3[i*4+0].strip()), int(x3[i*4+1].strip()), int(x3[i*4+2].strip()), int(x3[i*4+3].strip())])

                if 'total number of azimuth looks in the ionosphere processing' == x2[0].strip():
                    self.ION_numberAzimuthLooks = int(x2[1].strip())
                if 'total number of range looks in the ionosphere processing' == x2[0].strip():
                    self.ION_numberRangeLooks = int(x2[1].strip())
                if 'number of azimuth looks at first stage for ionosphere phase unwrapping' == x2[0].strip():
                    self.ION_numberAzimuthLooks0 = int(x2[1].strip())
                if 'number of range looks at first stage for ionosphere phase unwrapping' == x2[0].strip():
                    self.ION_numberRangeLooks0 = int(x2[1].strip())

    def print(self):
        '''print parameters'''

        print()

        print('ionosphere estimation parameters:')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print("do ionosphere correction (ION_doIon): {}".format(self.ION_doIon))
        print("consider burst properties in ionosphere computation (ION_considerBurstProperties): {}".format(self.ION_considerBurstProperties))

        print("height of ionosphere layer in km (ION_ionHeight): {}".format(self.ION_ionHeight))
        print("apply polynomial fit before filtering ionosphere phase (ION_ionFit): {}".format(self.ION_ionFit))
        print("maximum window size for filtering ionosphere phase (ION_ionFilteringWinsizeMax): {}".format(self.ION_ionFilteringWinsizeMax))
        print("minimum window size for filtering ionosphere phase (ION_ionFilteringWinsizeMin): {}".format(self.ION_ionFilteringWinsizeMin))
        print("maximum window size for filtering ionosphere azimuth shift (ION_ionshiftFilteringWinsizeMax): {}".format(self.ION_ionshiftFilteringWinsizeMax))
        print("minimum window size for filtering ionosphere azimuth shift (ION_ionshiftFilteringWinsizeMin): {}".format(self.ION_ionshiftFilteringWinsizeMin))
        print("correct phase error caused by ionosphere azimuth shift (ION_azshiftFlag): {}".format(self.ION_azshiftFlag))
        print("areas masked out in ionospheric phase estimation (ION_maskedAreas): {}".format(self.ION_maskedAreas))

        print("total number of azimuth looks in the ionosphere processing (ION_numberAzimuthLooks): {}".format(self.ION_numberAzimuthLooks))
        print("total number of range looks in the ionosphere processing (ION_numberRangeLooks): {}".format(self.ION_numberRangeLooks))
        print("number of azimuth looks at first stage for ionosphere phase unwrapping (ION_numberAzimuthLooks0): {}".format(self.ION_numberAzimuthLooks0))
        print("number of range looks at first stage for ionosphere phase unwrapping (ION_numberRangeLooks0): {}".format(self.ION_numberRangeLooks0))
        
        print()


class ionParam(object):
    '''A class containing parameters for ionosphere estimation
       while considerBurstProperties is not availavle for stack processing,
       ionParam still has parameters associated with considerBurstProperties for bookkeeping.
    '''

    def __init__(self, usrInput=None, safeObjFirst=None, safeObjSecondary=None):
        # usrInput: usrInput parameter object
        # safeObjFirst: sentinelSLC object defined in Stack.py of first date
        # safeObjSecond: sentinelSLC object defined in Stack.py of second date
        self.usrInput = usrInput
        self.safeObjFirst = safeObjFirst
        self.safeObjSecondary = safeObjSecondary

    def configure(self):
        #all paramters have default values, update the relevant parameters using
        #self.usrInput, self.safeObjFirst, self.safeObjSecondary
        #when they are not None

        ###################################################################
        #users are supposed to change parameters of this section ONLY
        #SECTION 1. PROCESSING CONTROL PARAMETERS
        #1. suggested default values of the parameters
        self.doIon = False
        self.considerBurstProperties = False

        #ionospheric layer height (m)
        self.ionHeight = 200.0 * 1000.0
        #before filtering ionosphere, if applying polynomial fitting
        #False: no fitting
        #True: with fitting
        self.ionFit = True
        #window size for filtering ionosphere
        self.ionFilteringWinsizeMax = 200
        self.ionFilteringWinsizeMin = 100
        #window size for filtering azimuth shift caused by ionosphere
        self.ionshiftFilteringWinsizeMax = 150
        self.ionshiftFilteringWinsizeMin = 75
        #correct phase error caused by non-zero center frequency and azimuth shift caused by ionosphere
        #0: no correction
        #1: use mean value of a burst
        #2: use full burst
        self.azshiftFlag = 1
        self.maskedAreas = None

        #better NOT try changing the following two parameters, since they are related
        #to the filtering parameters above
        #number of azimuth looks in the processing of ionosphere estimation
        self.numberAzimuthLooks = 50
        #number of range looks in the processing of ionosphere estimation
        self.numberRangeLooks = 200
        #number of azimuth looks of the interferogram to be unwrapped
        self.numberAzimuthLooks0 = 5*2
        #number of range looks of the interferogram to be unwrapped
        self.numberRangeLooks0 = 20*2


        #2. accept the above parameters from topsApp.py
        if self.usrInput is not None:
            self.doIon = self.usrInput.ION_doIon
            self.considerBurstProperties = self.usrInput.ION_considerBurstProperties

            self.ionHeight = self.usrInput.ION_ionHeight * 1000.0
            self.ionFit = self.usrInput.ION_ionFit
            self.ionFilteringWinsizeMax = self.usrInput.ION_ionFilteringWinsizeMax
            self.ionFilteringWinsizeMin = self.usrInput.ION_ionFilteringWinsizeMin
            self.ionshiftFilteringWinsizeMax = self.usrInput.ION_ionshiftFilteringWinsizeMax
            self.ionshiftFilteringWinsizeMin = self.usrInput.ION_ionshiftFilteringWinsizeMin
            self.azshiftFlag = self.usrInput.ION_azshiftFlag
            self.maskedAreas = self.usrInput.ION_maskedAreas

            self.numberAzimuthLooks = self.usrInput.ION_numberAzimuthLooks
            self.numberRangeLooks = self.usrInput.ION_numberRangeLooks
            self.numberAzimuthLooks0 = self.usrInput.ION_numberAzimuthLooks0
            self.numberRangeLooks0 = self.usrInput.ION_numberRangeLooks0


        #3. check parameters
        #check number of looks
        if not ((self.numberAzimuthLooks % self.numberAzimuthLooks0 == 0) and \
           (1 <= self.numberAzimuthLooks0 <= self.numberAzimuthLooks)):
            raise Exception('numberAzimuthLooks must be integer multiples of numberAzimuthLooks0')
        if not ((self.numberRangeLooks % self.numberRangeLooks0 == 0) and \
           (1 <= self.numberRangeLooks0 <= self.numberRangeLooks)):
            raise Exception('numberRangeLooks must be integer multiples of numberRangeLooks0')
        ###################################################################


        #SECTION 2. DIRECTORIES AND FILENAMES
        #directories
        self.ionDirname = 'ion'
        self.lowerDirname = 'lower'
        self.upperDirname = 'upper'
        self.ioncalDirname = 'ion_cal'
        self.ionBurstDirname = 'ion_burst'
        #these are same directory names as topsApp.py/TopsProc.py
        #self.referenceSlcProduct = 'reference'
        #self.secondarySlcProduct = 'secondary'
        #self.fineCoregDirname = 'fine_coreg'
        self.fineIfgDirname = 'fine_interferogram'
        self.mergedDirname = 'merged'
        #filenames
        self.ionRawNoProj = 'raw_no_projection.ion'
        self.ionCorNoProj = 'raw_no_projection.cor'
        self.ionRaw = 'raw.ion'
        self.ionCor = 'raw.cor'
        self.ionFilt = 'filt.ion'
        self.ionShift = 'azshift.ion'
        self.warning = 'warning.txt'

        #SECTION 3. DATA PARAMETERS
        #earth's radius (m)
        self.earthRadius = 6371 * 1000.0
        #reference range (m) for moving range center frequency to zero, center of center swath
        self.rgRef = 875714.0
        #range bandwidth (Hz) for splitting, range processingBandwidth: [5.650000000000000e+07, 4.830000000000000e+07, 4.278991840322842e+07]
        self.rgBandwidthForSplit = 40.0 * 10**6
        self.rgBandwidthSub = self.rgBandwidthForSplit / 3.0

        #SECTION 4. DEFINE WAVELENGTHS AND DETERMINE IF CALCULATE IONOSPHERE WITH MERGED INTERFEROGRAM
        #Sentinel-1A/B radar wavelengths are the same.
        self.radarWavelength = 0.05546576
        self.passDirection = None

        #self.safeObjFirst, self.safeObjSecondary should have already get these parameters
        #use the 1/3, 1/3, 1/3 scheme for splitting
        from isceobj.Constants import SPEED_OF_LIGHT
        if self.safeObjFirst is not None:
            #use this to determine which polynomial to use to calculate a ramp when calculating ionosphere for cross A/B interferogram
            self.passDirection = self.safeObjFirst.passDirection.lower()
            self.radarWavelength = self.safeObjFirst.radarWavelength
        self.radarWavelengthLower = SPEED_OF_LIGHT / (SPEED_OF_LIGHT / self.radarWavelength - self.rgBandwidthForSplit / 3.0)
        self.radarWavelengthUpper = SPEED_OF_LIGHT / (SPEED_OF_LIGHT / self.radarWavelength + self.rgBandwidthForSplit / 3.0)

        
        self.calIonWithMerged = False
        self.rampRemovel = 0
        #update the above two parameters depending on self.safeObjFirst and self.safeObjSecondary 
        if (self.safeObjFirst is not None) and (self.safeObjSecondary is not None):
            #determine if calculate ionosphere using merged interferogram
            #check if already got parameters needed
            if hasattr(self.safeObjFirst, 'startingRanges') ==  False:
                self.safeObjFirst.get_starting_ranges()
            if hasattr(self.safeObjSecondary, 'startingRanges') ==  False:
                self.safeObjSecondary.get_starting_ranges()
            if self.safeObjFirst.startingRanges == self.safeObjSecondary.startingRanges:
                self.calIonWithMerged = False
            else:
                self.calIonWithMerged = True
            #for cross Sentinel-1A/B interferogram, always not using merged interferogram
            if self.safeObjFirst.platform != self.safeObjSecondary.platform:
                self.calIonWithMerged = False
            #there is no need to process swath by swath when there is only one swath
            #ionSwathBySwath only works when number of swaths >=2
            #CONSIDER THIS LATTER!!!
            #if len(swathList) == 1:
            #    self.calIonWithMerged = True

            #determine if remove an empirical ramp
            if self.safeObjFirst.platform == self.safeObjSecondary.platform:
                self.rampRemovel = 0
            else:
                #estimating ionospheric phase for cross Sentinel-1A/B interferogram
                #an empirical ramp will be removed from the estimated ionospheric phase
                if self.safeObjFirst.platform == 'S1A' and self.safeObjSecondary.platform == 'S1B':
                    self.rampRemovel = 1
                else:
                    self.rampRemovel = -1


class run(object):
    """
       A class representing a run which may contain several functions
    """
    #def __init__(self):

    def configure(self,inps, runName):
        for k in inps.__dict__.keys():
            setattr(self, k, inps.__dict__[k])
        self.runDir = os.path.join(self.work_dir, 'run_files')
        os.makedirs(self.runDir, exist_ok=True)

        self.run_outname = os.path.join(self.runDir, runName)
        print ('writing ', self.run_outname)

        self.config_path = os.path.join(self.work_dir,'configs')
        os.makedirs(self.config_path, exist_ok=True)

        self.runf= open(self.run_outname,'w')

    def unpackSLC(self, acquisitionDates, safe_dict):
        swath_path = self.work_dir
        os.makedirs(self.config_path, exist_ok=True)

        line_cnt = 0
        for slcdate in acquisitionDates:
            configName = os.path.join(self.config_path,'config_unpack_'+slcdate)
            configObj = config(configName)
            configObj.configure(self)
            configObj.dirName = safe_dict[slcdate].safe_file
            configObj.orbit_file = safe_dict[slcdate].orbit
            configObj.orbit_type = safe_dict[slcdate].orbitType
            configObj.swaths = self.swath_num
            configObj.outDir = os.path.join(self.work_dir, 'slc/' + slcdate)
            configObj.geom_referenceDir = os.path.join(self.work_dir, 'geom_slc/' + slcdate)
            configObj.dem = os.path.join(self.work_dir, configObj.dem)
            configObj.Sentinel1_TOPS('[Function-1]')
            configObj.topo('[Function-2]')
            configObj.finalize()
            
            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
            del configObj

    def unpackStackReferenceSLC(self, safe_dict):
        swath_path = self.work_dir
        os.makedirs(self.config_path, exist_ok=True)
        configName = os.path.join(self.config_path,'config_reference')
        configObj = config(configName)
        configObj.configure(self)
        configObj.dirName = safe_dict[self.reference_date].safe_file
        configObj.orbit_file = safe_dict[self.reference_date].orbit
        configObj.orbit_type = safe_dict[self.reference_date].orbitType
        configObj.swaths = self.swath_num
        configObj.outDir = os.path.join(self.work_dir, 'reference')
        configObj.geom_referenceDir = os.path.join(self.work_dir, 'geom_reference')
        configObj.dem = os.path.join(self.work_dir, configObj.dem)
        configObj.Sentinel1_TOPS('[Function-1]')
        configObj.topo('[Function-2]')
        configObj.finalize()

        line_cnt = 1
        line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
        del configObj

    def unpackSecondarysSLC(self,  stackReferenceDate, secondaryList, safe_dict):

        line_cnt = 0
        for secondary in secondaryList:
            configName = os.path.join(self.config_path,'config_secondary_'+secondary)
            outdir = os.path.join(self.work_dir,'secondarys/'+secondary)
            configObj = config(configName)
            configObj.configure(self)
            configObj.dirName = safe_dict[secondary].safe_file
            configObj.orbit_file = safe_dict[secondary].orbit
            configObj.orbit_type = safe_dict[secondary].orbitType
            configObj.swaths = self.swath_num
            configObj.outDir = outdir
            configObj.Sentinel1_TOPS('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj

    def averageBaseline(self, stackReferenceDate, secondaryList):

        line_cnt = 0
        for secondary in secondaryList:
            configName = os.path.join(self.config_path,'config_baseline_'+secondary)
            configObj = config(configName)
            configObj.configure(self)
            configObj.reference = os.path.join(self.work_dir,'reference/')
            configObj.secondary = os.path.join(self.work_dir,'secondarys/'+secondary)
            configObj.baselineFile = os.path.join(self.work_dir,'baselines/' + stackReferenceDate +'_' + secondary + '/' + stackReferenceDate +'_'+ secondary  + '.txt')
            configObj.computeAverageBaseline('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj

    def gridBaseline(self, stackReferenceDate, secondaryList):

        line_cnt = 0
        for secondary in secondaryList:
            configName = os.path.join(self.config_path,'config_baselinegrid_'+secondary)
            configObj = config(configName)
            configObj.configure(self)
            configObj.reference = os.path.join(self.work_dir,'reference/')
            configObj.secondary = os.path.join(self.work_dir,'secondarys/'+secondary)
            configObj.baselineFile = os.path.join(self.work_dir, 'merged/baselines/' + secondary + '/' + secondary )
            configObj.computeGridBaseline('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
            del configObj
        # also add the reference in itself to be consistent with the SLC dir
        configName = os.path.join(self.config_path,'config_baselinegrid_reference')
        configObj = config(configName)
        configObj.configure(self)
        configObj.reference = os.path.join(self.work_dir,'reference/')
        configObj.secondary = os.path.join(self.work_dir,'reference/')
        configObj.baselineFile = os.path.join(self.work_dir, 'merged/baselines/' + stackReferenceDate + '/' + stackReferenceDate)
        configObj.computeGridBaseline('[Function-1]')
        configObj.finalize()

        line_cnt = 1
        line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
        del configObj


    def extractOverlaps(self):

        self.runf.write(self.text_cmd + 'subsetReference.py -m ' + os.path.join(self.work_dir, 'reference') + ' -g ' + os.path.join(self.work_dir, 'geom_reference') + '\n')


    def geo2rdr_offset(self, secondaryList, fullBurst='False'):

        line_cnt = 0
        for secondary in secondaryList:
            reference = self.reference_date
            if fullBurst == 'True':
                configName = os.path.join(self.config_path, 'config_fullBurst_geo2rdr_' + secondary)
            else:
                configName = os.path.join(self.config_path, 'config_overlap_geo2rdr_'+secondary)
            ###########
            configObj = config(configName)
            configObj.configure(self)
            configObj.secondaryDir = os.path.join(self.work_dir, 'secondarys/'+secondary)
            configObj.referenceDir = os.path.join(self.work_dir, 'reference')
            configObj.geom_reference = os.path.join(self.work_dir, 'geom_reference')
            configObj.coregSecondaryDir = os.path.join(self.work_dir, 'coreg_secondarys/'+secondary)
            if fullBurst == 'True':
                configObj.misreg_az = os.path.join(self.work_dir, 'misreg/azimuth/dates/' + secondary + '.txt')
                configObj.misreg_rng = os.path.join(self.work_dir, 'misreg/range/dates/' + secondary + '.txt')
                configObj.overlapTrueOrFalse = 'False'
            else:
                configObj.overlapTrueOrFalse = 'True'
            configObj.geo2rdr('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj

    def resample_with_carrier(self, secondaryList, fullBurst='False'):

        line_cnt = 0
        for secondary in secondaryList:
            reference = self.reference_date
            if fullBurst == 'True':
                configName = os.path.join(self.config_path, 'config_fullBurst_resample_' + secondary)
            else:
                configName = os.path.join(self.config_path, 'config_overlap_resample_' + secondary)
            ###########
            configObj = config(configName)
            configObj.configure(self)
            configObj.secondaryDir = os.path.join(self.work_dir, 'secondarys/' + secondary)
            configObj.referenceDir = os.path.join(self.work_dir, 'reference')
            configObj.coregSecondaryDir = os.path.join(self.work_dir, 'coreg_secondarys/' + secondary)
            configObj.interferogram_prefix = 'coarse'
            configObj.referenceDir = os.path.join(self.work_dir, 'reference')
            if fullBurst == 'True':
                configObj.misreg_az = os.path.join(self.work_dir, 'misreg/azimuth/dates/' + secondary + '.txt')
                configObj.misreg_rng = os.path.join(self.work_dir, 'misreg/range/dates/' + secondary + '.txt')
                configObj.overlapTrueOrFalse = 'False'
            else:
                configObj.overlapTrueOrFalse = 'True'
            configObj.resamp_withCarrier('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj


    def pairs_misregistration(self, dateList, safe_dict):
        # generating overlap interferograms, estimate azimuth misregistration for each pair:
        pairs = []
        num_overlap_connections = int(self.num_overlap_connections) + 1

        for i in range(len(dateList)-1):
            for j in range(i+1,i+num_overlap_connections):
                if j<len(dateList):
                    pairs.append((dateList[i],dateList[j]))

        for date in dateList:
            safe_dict[date].slc = os.path.join(self.work_dir , 'coreg_secondarys/'+date)
            safe_dict[date].slc_overlap = os.path.join(self.work_dir , 'coreg_secondarys/'+date)
        safe_dict[self.reference_date].slc = os.path.join(self.work_dir , 'reference')
        safe_dict[self.reference_date].slc_overlap = os.path.join(self.work_dir , 'reference')

        line_cnt = 0
        for pair in pairs:
            reference = pair[0]
            secondary = pair[1]
            interferogramDir = os.path.join(self.work_dir, 'coarse_interferograms/'+reference+'_'+secondary)
            configName = os.path.join(self.config_path ,'config_misreg_'+reference+'_'+secondary)
            configObj = config(configName)
            configObj.configure(self)
            configObj.referenceDir = safe_dict[reference].slc_overlap
            configObj.secondaryDir = safe_dict[secondary].slc_overlap
            configObj.interferogramDir = interferogramDir
            configObj.interferogram_prefix = 'int'
            configObj.flatten = 'False'
            configObj.overlapTrueOrFalse = 'True'
            configObj.generateIgram('[Function-1]')
            ########################
            configObj.reference = interferogramDir + '/' + 'coarse_ifg'
            configObj.referenceDir = safe_dict[reference].slc_overlap
            configObj.secondaryDir = safe_dict[secondary].slc_overlap
            configObj.overlapDir = os.path.join(self.work_dir, 'ESD/' + reference + '_' + secondary)
            configObj.overlap_withDEM('[Function-2]')
            ########################

            configObj.misregFile = os.path.join(self.work_dir , 'misreg/azimuth/pairs/' + reference+'_'+secondary + '/' + reference+'_'+secondary + '.txt')
            configObj.azimuthMisreg('[Function-3]')
            ########################
            configObj.misregFile = os.path.join(self.work_dir , 'misreg/range/pairs/' + reference+'_'+secondary + '/' + reference+'_'+secondary + '.txt')
            configObj.rangeMisreg('[Function-4]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj
            ########################


    def timeseries_misregistration(self):
        #inverting the misregistration offsets of the overlap pairs to estimate the offsets of each date
        self.runf.write(self.text_cmd + 'invertMisreg.py -i ' + os.path.join(self.work_dir,'misreg/azimuth/pairs/') + ' -o ' + os.path.join(self.work_dir,'misreg/azimuth/dates/') + '\n')
        self.runf.write(self.text_cmd + 'invertMisreg.py -i ' + os.path.join(self.work_dir,'misreg/range/pairs/') + ' -o ' + os.path.join(self.work_dir,'misreg/range/dates/') + '\n')

    def extractStackValidRegion(self):
        referenceDir = os.path.join(self.work_dir, 'reference')
        coregSecondaryDir = os.path.join(self.work_dir, 'coreg_secondarys')
        self.runf.write(self.text_cmd + 'extractCommonValidRegion.py -m ' + referenceDir + ' -s ' + coregSecondaryDir + '\n')

    def generate_burstIgram(self, dateList, safe_dict, pairs):

        for date in dateList:
            safe_dict[date].slc = os.path.join(self.work_dir, 'coreg_secondarys/'+date)
        safe_dict[self.reference_date].slc = os.path.join(self.work_dir , 'reference')

        line_cnt = 0
        for pair in pairs:
            reference = pair[0]
            secondary = pair[1]
            interferogramDir = os.path.join(self.work_dir, 'interferograms/' + reference + '_' + secondary)
            configName = os.path.join(self.config_path ,'config_generate_igram_' + reference + '_' + secondary)
            configObj = config(configName)
            configObj.configure(self)
            configObj.referenceDir = safe_dict[reference].slc
            configObj.secondaryDir = safe_dict[secondary].slc
            configObj.interferogramDir = interferogramDir
            configObj.interferogram_prefix = 'fine'
            configObj.flatten = 'False'
            configObj.overlapTrueOrFalse = 'False'
            configObj.generateIgram('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj



    def igram_mergeBurst(self, dateList, safe_dict, pairs):
        for date in dateList:
            safe_dict[date].slc = os.path.join(self.work_dir, 'coreg_secondarys/'+date)
        safe_dict[self.reference_date].slc = os.path.join(self.work_dir , 'reference')

        line_cnt = 0
        for pair in pairs:
            reference = pair[0]
            secondary = pair[1]
            interferogramDir = os.path.join(self.work_dir, 'interferograms/' + reference + '_' + secondary)
            mergedDir = os.path.join(self.work_dir, 'merged/interferograms/' + reference + '_' + secondary)
            configName = os.path.join(self.config_path ,'config_merge_igram_' + reference + '_' + secondary)
            configObj = config(configName)
            configObj.configure(self)
            configObj.interferogram_prefix = 'fine'
            configObj.reference = interferogramDir
            configObj.dirName = interferogramDir
            configObj.namePattern = 'fine*int'
            configObj.mergedFile = mergedDir + '/' + configObj.interferogram_prefix + '.int'
            configObj.mergeBurstsMethod = 'top'
            configObj.aligned = 'True'
            configObj.validOnly = 'True'
            configObj.useVirtualFiles = 'True'
            configObj.multiLook = 'True'
            configObj.stack = os.path.join(self.work_dir, 'stack')
            configObj.mergeBurst('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj

    def mergeSecondarySLC(self, secondaryList, virtual='True'):

        line_cnt = 0
        for secondary in secondaryList:
            configName = os.path.join(self.config_path,'config_merge_' + secondary)
            configObj = config(configName)
            configObj.configure(self)
            configObj.stack = os.path.join(self.work_dir, 'stack')
            configObj.reference = os.path.join(self.work_dir, 'coreg_secondarys/' + secondary)
            configObj.dirName = configObj.reference
            configObj.namePattern = 'burst*slc'
            configObj.mergedFile = os.path.join(self.work_dir, 'merged/SLC/' + secondary + '/' + secondary + '.slc')
            configObj.mergeBurstsMethod = 'top'
            configObj.aligned = 'True'
            configObj.validOnly = 'True'
            configObj.useVirtualFiles = virtual
            configObj.multiLook = 'False'
            configObj.stack = os.path.join(self.work_dir, 'stack')
            configObj.mergeBurst('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
            del configObj


    def mergeReference(self, stackReference, virtual='True'):

        configName = os.path.join(self.config_path,'config_merge_' + stackReference)
        configObj = config(configName)
        configObj.configure(self)
        configObj.stack = os.path.join(self.work_dir, 'stack')
        configObj.reference = os.path.join(self.work_dir, 'reference')
        configObj.dirName = configObj.reference
        configObj.namePattern = 'burst*slc'
        configObj.mergedFile = os.path.join(self.work_dir, 'merged/SLC/' + stackReference + '/' + stackReference + '.slc')
        configObj.mergeBurstsMethod = 'top'
        configObj.aligned = 'False'
        configObj.validOnly = 'True'
        configObj.useVirtualFiles = virtual
        configObj.multiLook = 'False'
        configObj.mergeBurst('[Function-1]')
        configObj.finalize()

        line_cnt = 1
        line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
        del configObj

        geometryList = ['lat*rdr', 'lon*rdr', 'los*rdr', 'hgt*rdr', 'shadowMask*rdr','incLocal*rdr']
        multiookToolDict = {'lat*rdr': 'gdal', 'lon*rdr': 'gdal', 'los*rdr': 'gdal' , 'hgt*rdr':"gdal", 'shadowMask*rdr':"isce",'incLocal*rdr':"gdal"}
        noDataDict = {'lat*rdr': '0', 'lon*rdr': '0', 'los*rdr': '0' , 'hgt*rdr':None, 'shadowMask*rdr':None,'incLocal*rdr':"0"}

        for i in range(len(geometryList)):
            pattern = geometryList[i]
            configName = os.path.join(self.config_path,'config_merge_' + pattern.split('*')[0])
            configObj = config(configName)
            configObj.configure(self)
            configObj.reference = os.path.join(self.work_dir, 'reference')
            configObj.dirName = os.path.join(self.work_dir, 'geom_reference')
            configObj.namePattern = pattern
            configObj.mergedFile = os.path.join(self.work_dir, 'merged/geom_reference/' + pattern.split('*')[0] + '.' + pattern.split('*')[1])
            configObj.mergeBurstsMethod = 'top'
            configObj.aligned = 'False'
            configObj.validOnly = 'False'
            configObj.useVirtualFiles = virtual
            configObj.multiLook = 'True'
            configObj.multilook_tool = multiookToolDict[pattern]
            configObj.no_data_value = noDataDict[pattern]
            configObj.stack = os.path.join(self.work_dir, 'stack')
            configObj.mergeBurst('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
            del configObj

    def mergeSLC(self, aquisitionDates, virtual='True'):

        line_cnt  = 0
        for slcdate in aquisitionDates:
            configName = os.path.join(self.config_path,'config_merge_' + slcdate)
            configObj = config(configName)
            configObj.configure(self)
            configObj.reference = os.path.join(self.work_dir, 'slc/' + slcdate)
            configObj.dirName = configObj.reference
            configObj.namePattern = 'burst*slc'
            configObj.mergedFile = os.path.join(self.work_dir, 'merged/slc/' + slcdate + '/' + slcdate + '.slc')
            configObj.mergeBurstsMethod = 'top'
            configObj.aligned = 'False'
            configObj.validOnly = 'True'
            configObj.useVirtualFiles = virtual
            configObj.multiLook = 'False'
            configObj.stack = os.path.join(self.work_dir, 'stack')
            configObj.mergeBurst('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)


            geometryList = ['lat*rdr', 'lon*rdr', 'los*rdr', 'hgt*rdr', 'shadowMask*rdr','incLocal*rdr']
            
            g_line_cnt = 0
            for i in range(len(geometryList)):
                pattern = geometryList[i]
                configName = os.path.join(self.config_path,'config_merge_' + slcdate + '_' +pattern.split('*')[0])
                configObj = config(configName)
                configObj.configure(self)
                configObj.reference = os.path.join(self.work_dir, 'slc/' + slcdate)
                configObj.dirName = os.path.join(self.work_dir, 'geom_slc/' + slcdate)
                configObj.namePattern = pattern
                configObj.mergedFile = os.path.join(self.work_dir, 'merged/geom_slc/' + slcdate + '/' + pattern.split('*')[0] + '.' + pattern.split('*')[1])
                configObj.mergeBurstsMethod = 'top'
                configObj.aligned = 'False'
                configObj.validOnly = 'False'
                configObj.useVirtualFiles = virtual
                configObj.multiLook = 'False'
                configObj.stack = os.path.join(self.work_dir, 'stack')
                configObj.mergeBurst('[Function-1]')
                configObj.finalize()
                
                g_line_cnt += 1
                g_line_cnt = configObj.write_wrapper_config2run_file(configName, g_line_cnt)
                del configObj

    def filter_coherence(self, pairs):

        line_cnt = 0
        for pair in pairs:
            reference = pair[0]
            secondary = pair[1]
            mergedDir = os.path.join(self.work_dir, 'merged/interferograms/' + reference + '_' + secondary)
            mergedSLCDir = os.path.join(self.work_dir, 'merged/SLC')
            configName = os.path.join(self.config_path, 'config_igram_filt_coh_' + reference + '_' + secondary)
            configObj = config(configName)
            configObj.configure(self)
            configObj.input = os.path.join(mergedDir, 'fine.int')
            configObj.filtName = os.path.join(mergedDir, 'filt_fine.int')
            configObj.cohName = os.path.join(mergedDir, 'filt_fine.cor')
            configObj.slc1 = os.path.join(mergedSLCDir, '{}/{}.slc.full'.format(reference, reference))
            configObj.slc2 = os.path.join(mergedSLCDir, '{}/{}.slc.full'.format(secondary, secondary))
            configObj.cpxCohName = os.path.join(mergedDir, 'fine.cor')
            if int(self.rangeLooks) * int(self.azimuthLooks) == 1:
                configObj.cpxCohName += '.full'
            #configObj.filtStrength = str(self.filtStrength)
            configObj.FilterAndCoherence('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj


    def unwrap(self, pairs):

        line_cnt = 0
        for pair in pairs:
            reference = pair[0]
            secondary = pair[1]
            mergedDir = os.path.join(self.work_dir, 'merged/interferograms/' + reference + '_' + secondary)
            configName = os.path.join(self.config_path ,'config_igram_unw_' + reference + '_' + secondary)
            configObj = config(configName)
            configObj.configure(self)
            configObj.ifgName = os.path.join(mergedDir,'filt_fine.int')
            configObj.cohName = os.path.join(mergedDir,'filt_fine.cor')
            configObj.unwName = os.path.join(mergedDir,'filt_fine.unw')
            configObj.noMCF = noMCF
            configObj.rmFilter = self.rmFilter
            configObj.reference = os.path.join(self.work_dir,'reference')
            configObj.defoMax = defoMax
            configObj.unwMethod = self.unwMethod
            configObj.unwrap('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj

    def denseOffsets(self, pairs):

        line_cnt = 0
        for pair in pairs:
            reference = pair[0]
            secondary = pair[1]
            configName = os.path.join(self.config_path ,'config_denseOffset_' + reference + '_' + secondary)
            configObj = config(configName)
            configObj.configure(self)
            configObj.reference = os.path.join(self.work_dir, 'merged/SLC/' + reference + '/' + reference + '.slc.full')
            configObj.secondary = os.path.join(self.work_dir, 'merged/SLC/' + secondary + '/' + secondary + '.slc.full')

            configObj.output = os.path.join(self.work_dir , 'merged/dense_offsets/'+reference+'_'+secondary + '/'  + reference+'_'+secondary)
            configObj.denseOffset('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
            del configObj


    def subband_and_resamp(self, dateListIon, stackReferenceDate):

        line_cnt = 0
        for date in dateListIon:
            configName = os.path.join(self.config_path,'config_subband_and_resamp_{}'.format(date))
            configObj = config(configName)
            configObj.configure(self)
            configObj.reference = os.path.join(self.work_dir, 'reference')
            configObj.secondary = os.path.join(self.work_dir, 'secondarys', date)
            configObj.coregdir = os.path.join(self.work_dir, 'coreg_secondarys', date)
            configObj.azimuth_misreg = os.path.join(self.work_dir, 'misreg', 'azimuth', 'dates', '{}.txt'.format(date))
            configObj.range_misreg = os.path.join(self.work_dir, 'misreg', 'range', 'dates', '{}.txt'.format(date))
            if date == stackReferenceDate:
                configObj.subband('[Function-1]')
            else:
                configObj.subband_and_resamp('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj


    def generateIgram_ion(self, pairs, stackReferenceDate):

        line_cnt = 0
        for p in pairs:
            configName = os.path.join(self.config_path,'config_generateIgram_ion_{}_{}'.format(p[0], p[1]))
            configObj = config(configName)
            configObj.configure(self)
            if p[0] == stackReferenceDate:
                configObj.reference = os.path.join(self.work_dir, 'reference')
            else:
                configObj.reference = os.path.join(self.work_dir, 'coreg_secondarys', p[0])
            if p[1] == stackReferenceDate:
                configObj.secondary = os.path.join(self.work_dir, 'reference')
            else:
                configObj.secondary = os.path.join(self.work_dir, 'coreg_secondarys', p[1])
            configObj.flatten = 'False'
            configObj.interferogram_prefix = 'fine'
            configObj.overlap = 'False'

            configObj.reference_suffix = '_lower'
            configObj.secondary_suffix = '_lower'
            configObj.interferogram = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'interferograms')
            configObj.generateIgram_ion('[Function-1]')

            configObj.reference_suffix = '_upper'
            configObj.secondary_suffix = '_upper'
            configObj.interferogram = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'upper', 'interferograms')
            configObj.generateIgram_ion('[Function-2]')
            configObj.finalize()

            line_cnt += 1
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            del configObj


    def mergeBurstsIon(self, pairs_same_starting_ranges_update, pairs_diff_starting_ranges_update):
        import numpy as np

        swath_list = sorted(self.swath_num.split())
        nswath = len(swath_list)

        ionParamUsrObj = ionParamUsr(self.param_ion)
        ionParamUsrObj.configure()

        if nswath == 1:
            pairs1 = pairs_same_starting_ranges_update + pairs_diff_starting_ranges_update
            pairs2 = []
        else:
            pairs1 = pairs_same_starting_ranges_update
            pairs2 = pairs_diff_starting_ranges_update

        line_cnt = 0
        for p in pairs1+pairs2:
            configName = os.path.join(self.config_path,'config_mergeBurstsIon_{}-{}'.format(p[0], p[1]))
            configObj = config(configName)
            configObj.configure(self)

            if p in pairs1:
                for subband, function in zip(['lower', 'upper'], ['[Function-1]', '[Function-2]']):
                    configObj.reference = os.path.join(self.work_dir, 'reference')
                    configObj.stack = os.path.join(self.work_dir, 'stack')
                    configObj.dirname = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'interferograms')
                    configObj.name_pattern = 'fine_*.int'
                    configObj.outfile = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'merged', 'fine.int')
                    configObj.nrlks = ionParamUsrObj.ION_numberRangeLooks
                    configObj.nalks = ionParamUsrObj.ION_numberAzimuthLooks
                    configObj.nrlks0 = ionParamUsrObj.ION_numberRangeLooks0
                    configObj.nalks0 = ionParamUsrObj.ION_numberAzimuthLooks0
                    configObj.rvalid = None
                    configObj.swath = None
                    configObj.mergeBurstsIon(function)

                configObj.lower = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged', 'fine.int')
                configObj.upper = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'upper', 'merged', 'fine.int')
                configObj.coherence = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged', 'fine.cor')
                configObj.nrlks = None
                configObj.nalks = None
                configObj.coherenceIon('[Function-3]')

            if p in pairs2:
                num = 2 * nswath
                subbandAll = ['lower' for i in range(nswath)] + ['upper' for i in range(nswath)]
                swathAll = swath_list + swath_list
                functionAll = ['[Function-{}]'.format(i+1) for i in range(num)]
                for subband, swath, function in zip(subbandAll, swathAll, functionAll):
                    configObj.reference = os.path.join(self.work_dir, 'reference')
                    configObj.stack = os.path.join(self.work_dir, 'stack')
                    configObj.dirname = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'interferograms')
                    configObj.name_pattern = 'fine_*.int'
                    configObj.outfile = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'merged_IW{}'.format(swath), 'fine.int')
                    configObj.nrlks = ionParamUsrObj.ION_numberRangeLooks
                    configObj.nalks = ionParamUsrObj.ION_numberAzimuthLooks
                    configObj.nrlks0 = ionParamUsrObj.ION_numberRangeLooks0
                    configObj.nalks0 = ionParamUsrObj.ION_numberAzimuthLooks0
                    configObj.rvalid = np.int32(np.around(ionParamUsrObj.ION_numberRangeLooks/8.0))
                    configObj.swath = swath
                    configObj.mergeBurstsIon(function)

                swathAll = swath_list
                functionAll = ['[Function-{}]'.format(num+i+1) for i in range(nswath)]
                for swath, function in zip(swathAll, functionAll):
                    configObj.lower = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged_IW{}'.format(swath), 'fine.int')
                    configObj.upper = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'upper', 'merged_IW{}'.format(swath), 'fine.int')
                    configObj.coherence = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged_IW{}'.format(swath), 'fine.cor')
                    configObj.nrlks = None
                    configObj.nalks = None
                    configObj.coherenceIon(function)

            configObj.finalize()

            line_cnt += 1
            #line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
            del configObj


    def unwrap_ion(self, pairs_same_starting_ranges_update, pairs_diff_starting_ranges_update):
        import numpy as np

        swath_list = sorted(self.swath_num.split())
        nswath = len(swath_list)

        ionParamUsrObj = ionParamUsr(self.param_ion)
        ionParamUsrObj.configure()

        if nswath == 1:
            pairs1 = pairs_same_starting_ranges_update + pairs_diff_starting_ranges_update
            pairs2 = []
        else:
            pairs1 = pairs_same_starting_ranges_update
            pairs2 = pairs_diff_starting_ranges_update

        line_cnt = 0
        for p in pairs1+pairs2:
            configName = os.path.join(self.config_path,'config_unwrap_ion_{}-{}'.format(p[0], p[1]))
            configObj = config(configName)
            configObj.configure(self)

            if p in pairs1:
                for subband, function in zip(['lower', 'upper'], ['[Function-1]', '[Function-2]']):
                    configObj.ifgName = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'merged', 'fine.int')
                    configObj.unwName = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'merged', 'fine.unw')
                    configObj.cohName = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged', 'fine.cor')
                    configObj.noMCF = 'False'
                    configObj.reference = os.path.join(self.work_dir, 'reference')
                    configObj.defoMax = '2'
                    configObj.rangeLooks = '{}'.format(ionParamUsrObj.ION_numberRangeLooks0)
                    configObj.azimuthLooks = '{}'.format(ionParamUsrObj.ION_numberAzimuthLooks0)
                    configObj.rmfilter = False
                    configObj.unwMethod = 'snaphu'
                    configObj.unwrap(function)

            if p in pairs2:
                num = 2 * nswath
                subbandAll = ['lower' for i in range(nswath)] + ['upper' for i in range(nswath)]
                swathAll = swath_list + swath_list
                functionAll = ['[Function-{}]'.format(i+1) for i in range(num)]
                for subband, swath, function in zip(subbandAll, swathAll, functionAll):
                    configObj.ifgName = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'merged_IW{}'.format(swath), 'fine.int')
                    configObj.unwName = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'merged_IW{}'.format(swath), 'fine.unw')
                    configObj.cohName = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged_IW{}'.format(swath), 'fine.cor')
                    configObj.noMCF = 'False'
                    configObj.reference = os.path.join(self.work_dir, 'reference')
                    configObj.defoMax = '2'
                    configObj.rangeLooks = '{}'.format(ionParamUsrObj.ION_numberRangeLooks0)
                    configObj.azimuthLooks = '{}'.format(ionParamUsrObj.ION_numberAzimuthLooks0)
                    configObj.rmfilter = False
                    configObj.unwMethod = 'snaphu'
                    configObj.unwrap(function)

            configObj.finalize()

            line_cnt += 1
            #line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
            del configObj


    def look_ion(self, pairs_same_starting_ranges_update, pairs_diff_starting_ranges_update):
        import numpy as np

        swath_list = sorted(self.swath_num.split())
        nswath = len(swath_list)

        ionParamUsrObj = ionParamUsr(self.param_ion)
        ionParamUsrObj.configure()

        if nswath == 1:
            pairs1 = pairs_same_starting_ranges_update + pairs_diff_starting_ranges_update
            pairs2 = []
        else:
            pairs1 = pairs_same_starting_ranges_update
            pairs2 = pairs_diff_starting_ranges_update

        line_cnt = 0
        for p in pairs1+pairs2:
            configName = os.path.join(self.config_path,'config_look_ion_{}-{}'.format(p[0], p[1]))
            configObj = config(configName)
            configObj.configure(self)

            if p in pairs1:
                for subband, function in zip(['lower', 'upper'], ['[Function-1]', '[Function-2]']):
                    configObj.unw = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'merged', 'fine.unw')
                    configObj.cor = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged', 'fine.cor')
                    configObj.output = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'merged', 'fine_look.unw')
                    configObj.nrlks = np.int32(np.around(ionParamUsrObj.ION_numberRangeLooks / ionParamUsrObj.ION_numberRangeLooks0))
                    configObj.nalks = np.int32(np.around(ionParamUsrObj.ION_numberAzimuthLooks / ionParamUsrObj.ION_numberAzimuthLooks0))
                    configObj.lookUnwIon(function)

                configObj.lower = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged', 'fine.int')
                configObj.upper = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'upper', 'merged', 'fine.int')
                configObj.coherence = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged', 'fine_look.cor')
                #configObj.nrlks = np.int32(np.around(ionParamUsrObj.ION_numberRangeLooks / ionParamUsrObj.ION_numberRangeLooks0))
                #configObj.nalks = np.int32(np.around(ionParamUsrObj.ION_numberAzimuthLooks / ionParamUsrObj.ION_numberAzimuthLooks0))
                configObj.coherenceIon('[Function-3]')

            if p in pairs2:
                num = 2 * nswath
                subbandAll = ['lower' for i in range(nswath)] + ['upper' for i in range(nswath)]
                swathAll = swath_list + swath_list
                functionAll = ['[Function-{}]'.format(i+1) for i in range(num)]
                for subband, swath, function in zip(subbandAll, swathAll, functionAll):
                    configObj.unw = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'merged_IW{}'.format(swath), 'fine.unw')
                    configObj.cor = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged_IW{}'.format(swath), 'fine.cor')
                    configObj.output = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), '{}'.format(subband), 'merged_IW{}'.format(swath), 'fine_look.unw')
                    configObj.nrlks = np.int32(np.around(ionParamUsrObj.ION_numberRangeLooks / ionParamUsrObj.ION_numberRangeLooks0))
                    configObj.nalks = np.int32(np.around(ionParamUsrObj.ION_numberAzimuthLooks / ionParamUsrObj.ION_numberAzimuthLooks0))
                    configObj.lookUnwIon(function)

                swathAll = swath_list
                functionAll = ['[Function-{}]'.format(num+i+1) for i in range(nswath)]
                for swath, function in zip(swathAll, functionAll):
                    configObj.lower = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged_IW{}'.format(swath), 'fine.int')
                    configObj.upper = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'upper', 'merged_IW{}'.format(swath), 'fine.int')
                    configObj.coherence = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged_IW{}'.format(swath), 'fine_look.cor')
                    #configObj.nrlks = np.int32(np.around(ionParamUsrObj.ION_numberRangeLooks / ionParamUsrObj.ION_numberRangeLooks0))
                    #configObj.nalks = np.int32(np.around(ionParamUsrObj.ION_numberAzimuthLooks / ionParamUsrObj.ION_numberAzimuthLooks0))
                    configObj.coherenceIon(function)


            configObj.finalize()

            line_cnt += 1
            #line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
            del configObj


    def computeIon(self, pairs_same_starting_ranges_update, pairs_diff_starting_ranges_update, safe_dict):
        import numpy as np

        swath_list = sorted(self.swath_num.split())
        nswath = len(swath_list)

        ionParamUsrObj = ionParamUsr(self.param_ion)
        ionParamUsrObj.configure()

        if nswath == 1:
            pairs1 = pairs_same_starting_ranges_update + pairs_diff_starting_ranges_update
            pairs2 = []
        else:
            pairs1 = pairs_same_starting_ranges_update
            pairs2 = pairs_diff_starting_ranges_update


        for p in pairs1+pairs2:
            ionParamObj = ionParam(usrInput=ionParamUsrObj, safeObjFirst=safe_dict[p[0]], safeObjSecondary=safe_dict[p[1]])
            ionParamObj.configure()

            #do not use SentinelWrapper.py, as it does not support 2-d list masked_areas
            #configName = os.path.join(self.config_path,'config_mergeBurstsIon_{}-{}'.format(p[0], p[1]))
            #configObj = config(configName)
            #configObj.configure(self)

            if p in pairs1:
                lower = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged', 'fine_look.unw')
                upper = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'upper', 'merged', 'fine_look.unw')
                coherence = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged', 'fine_look.cor')
                ionosphere = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'ion_cal', 'raw_no_projection.ion')
                coherence_output = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'ion_cal', 'raw_no_projection.cor')
                masked_areas = ''
                if ionParamObj.maskedAreas is not None:
                    for m in ionParamObj.maskedAreas:
                        masked_areas += ' --masked_areas {} {} {} {}'.format(m[0], m[1], m[2], m[3])

                cmd = 'computeIon.py --lower {} --upper {} --coherence {} --ionosphere {} --coherence_output {}{}'.format(
                    lower, upper, coherence, ionosphere, coherence_output, masked_areas)

                self.runf.write(self.text_cmd + cmd + '\n')


            if p in pairs2:
                for swath in swath_list:
                    lower = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged_IW{}'.format(swath), 'fine_look.unw')
                    upper = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'upper', 'merged_IW{}'.format(swath), 'fine_look.unw')
                    coherence = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'lower', 'merged_IW{}'.format(swath), 'fine_look.cor')
                    ionosphere = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'ion_cal_IW{}'.format(swath), 'raw_no_projection.ion')
                    coherence_output = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'ion_cal_IW{}'.format(swath), 'raw_no_projection.cor')
                    masked_areas = ''
                    if ionParamObj.maskedAreas is not None:
                        for m in ionParamObj.maskedAreas:
                            masked_areas += ' --masked_areas {} {} {} {}'.format(m[0], m[1], m[2], m[3])

                    cmd = 'computeIon.py --lower {} --upper {} --coherence {} --ionosphere {} --coherence_output {}{}'.format(
                        lower, upper, coherence, ionosphere, coherence_output, masked_areas)

                    self.runf.write(self.text_cmd + cmd + '\n')

                #merge swaths
                reference = os.path.join(self.work_dir, 'reference')
                stack = os.path.join(self.work_dir, 'stack')
                input0 = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]))
                output = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'ion_cal')
                nrlks = ionParamObj.numberRangeLooks
                nalks = ionParamObj.numberAzimuthLooks
                remove_ramp = ionParamObj.rampRemovel

                cmd = 'mergeSwathIon.py --reference {} --stack {} --input {} --output {} --nrlks {} --nalks {} --remove_ramp {}'.format(
                    reference, stack, input0, output, nrlks, nalks, remove_ramp)
                self.runf.write(self.text_cmd + cmd + '\n')


    def filtIon(self, pairs):

        ionParamUsrObj = ionParamUsr(self.param_ion)
        ionParamUsrObj.configure()

        line_cnt = 0
        for p in pairs:
            configName = os.path.join(self.config_path,'config_filtIon_{}_{}'.format(p[0], p[1]))
            configObj = config(configName)
            configObj.configure(self)
            configObj.input = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'ion_cal', 'raw_no_projection.ion')
            configObj.coherence = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'ion_cal', 'raw_no_projection.cor')
            configObj.output = os.path.join(self.work_dir, 'ion', '{}_{}'.format(p[0], p[1]), 'ion_cal', 'filt.ion')
            configObj.win_min = ionParamUsrObj.ION_ionFilteringWinsizeMin
            configObj.win_max = ionParamUsrObj.ION_ionFilteringWinsizeMax
            configObj.filtIon('[Function-1]')
            configObj.finalize()

            line_cnt += 1
            #line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt, self.numProcess)
            line_cnt = configObj.write_wrapper_config2run_file(configName, line_cnt)
            del configObj


    def invertIon(self):

        ionParamUsrObj = ionParamUsr(self.param_ion)
        ionParamUsrObj.configure()
        
        ion_in = os.path.join(self.work_dir,'ion')
        ion_out = os.path.join(self.work_dir,'ion_dates')
        hgt = os.path.join(self.work_dir,'merged/geom_reference/hgt.rdr')

        cmd = 'invertIon.py --idir {} --odir {} --nrlks1 {} --nalks1 {} --nrlks2 {} --nalks2 {} --merged_geom {} --interp --msk_overlap'.format(ion_in,ion_out,ionParamUsrObj.ION_numberRangeLooks, ionParamUsrObj.ION_numberAzimuthLooks, self.rangeLooks, self.azimuthLooks,hgt)

        self.runf.write(self.text_cmd + cmd + '\n')


    def finalize(self):
        self.runf.close()
        #writeJobFile(self.run_outname)

class sentinelSLC(object):
    """
        A Class representing the SLCs
    """
    def __init__(self, safe_file=None, orbit_file=None ,slc=None ):
        self.safe_file = safe_file
        self.orbit = orbit_file
        self.slc = slc

    def get_dates(self):
        datefmt = "%Y%m%dT%H%M%S"
        safe = os.path.basename(self.safe_file)
        fields = safe.split('_')
        self.platform = fields[0]
        self.start_date_time = datetime.datetime.strptime(fields[5], datefmt)
        self.stop_date_time = datetime.datetime.strptime(fields[6], datefmt)
        self.datetime = datetime.datetime.date(self.start_date_time)
        self.date = self.datetime.isoformat().replace('-','')

    def get_lat_lon(self):
        lats=[]
        lons=[]
        for safe in self.safe_file.split():
           from xml.etree import ElementTree as ET

           file=os.path.join(safe,'preview/map-overlay.kml')
           kmlFile = open( file, 'r' ).read(-1)
           kmlFile = kmlFile.replace( 'gx:', 'gx' )

           kmlData = ET.fromstring( kmlFile )
           document = kmlData.find('Document/Folder/GroundOverlay/gxLatLonQuad')
           pnts = document.find('coordinates').text.split()
           for pnt in pnts:
              lons.append(float(pnt.split(',')[0]))
              lats.append(float(pnt.split(',')[1]))
        self.SNWE=[min(lats),max(lats),min(lons),max(lons)]


    def get_starting_ranges(self, safe=None):
        import zipfile
        import xml.etree.ElementTree as ET
        from isceobj.Planet.AstronomicalHandbook import Const

        #if safe file not set, use first slice in the safe list sorted by starting time
        if safe is None:
            safe = sorted(self.safe_file.split(), key=lambda x: x.split('_')[-5], reverse=False)[0]
        zf = zipfile.ZipFile(safe, 'r')
        anna = sorted([item for item in zf.namelist() if '.SAFE/annotation/s1' in item])
        #dual polarization. for the same swath, the slant ranges of two polarizations should be the same.
        if len(anna) == 6:
            anna = anna[1:6:2]

        startingRange = []
        for k in range(3):
            xmlstr = zf.read(anna[k])
            root = ET.fromstring(xmlstr)
            #startingRange computation exactly same as those in components/isceobj/Sensor/TOPS/Sentinel1.py used
            #by topsApp.py and topsStack
            startingRange.append(
                float(root.find('imageAnnotation/imageInformation/slantRangeTime').text)*Const.c/2.0
                )

            if k == 0:
                self.radarWavelength = Const.c / float(root.find('generalAnnotation/productInformation/radarFrequency').text)
                self.passDirection = root.find('generalAnnotation/productInformation/pass').text

        self.startingRanges = startingRange


    def getkmlQUAD(self,safe):
        # The coordinates in pnts must be specified in counter-clockwise order with the first coordinate corresponding to the lower-left corner of the overlayed image.
        # The shape described by these corners must be convex.
        # It appears this does not mean the coordinates are counter-clockwize in lon-lat reference.
        import zipfile
        from xml.etree import ElementTree as ET


        if safe.endswith('.zip'):
            zf = zipfile.ZipFile(safe,'r')
            fname = os.path.join(os.path.basename(safe).replace('zip','SAFE'), 'preview/map-overlay.kml')
            xmlstr = zf.read(fname)
            xmlstr=xmlstr.decode('utf-8')
            start = '<coordinates>'
            end = '</coordinates>'
            pnts = xmlstr[xmlstr.find(start)+len(start):xmlstr.find(end)].split()

        else:
            file=os.path.join(safe,'preview/map-overlay.kml')
            kmlFile = open( file, 'r' ).read(-1)
            kmlFile = kmlFile.replace( 'gx:', 'gx' )
            kmlData = ET.fromstring( kmlFile )
            document = kmlData.find('Document/Folder/GroundOverlay/gxLatLonQuad')
            pnts = document.find('coordinates').text.split()

        # convert the pnts to a list
        from scipy.spatial import distance as dist
        import numpy as np
        import cv2
        lats = []
        lons = []
        for pnt in pnts:
            lons.append(float(pnt.split(',')[0]))
            lats.append(float(pnt.split(',')[1]))
        pts = np.array([[a,b] for a,b in zip(lons,lats)])

        # The two points with most western longitude correspond to the left corners of the rectangle
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        # the top right corner corresponds to the point which has the highest latitude of the two western most corners
        # the second point left will be the bottom left corner
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (bl, tl) = leftMost


        # the two points with the most eastern longitude correspond to the right cornersof the rectangle
        rightMost = xSorted[2:, :]

        '''print("left most")
        print(leftMost)
        print("")
        print("right most")
        print(rightMost)
        print("")'''


        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        temp = np.array([tl, tr, br, bl], dtype="float32")
        #print(temp)
        #print(pnts)
        pnts_new = [str(bl[0])+','+str(bl[1]),str(br[0])+','+str(br[1]) ,str(tr[0])+','+str(tr[1]),str(tl[0])+','+str(tl[1])]
        #print(pnts_new)
        #raise Exception ("STOP")
        return pnts_new

    def get_lat_lon_v2(self):

        import numpy as np
        lats = []
        lons = []

        # track the min lat and max lat in columns with the rows each time a different SAF file
        lat_frame_max = []
        lat_frame_min = []
        for safe in self.safe_file.split():
           safeObj=sentinelSLC(safe)
           pnts = safeObj.getkmlQUAD(safe)
           # The coordinates must be specified in counter-clockwise order with the first coordinate corresponding
           # to the lower-left corner of the overlayed image
           counter=0
           for pnt in pnts:
              lons.append(float(pnt.split(',')[0]))
              lats.append(float(pnt.split(',')[1]))

              # only take the bottom [0] and top [3] left coordinates
              if counter==0:
                  lat_frame_min.append(float(pnt.split(',')[1]))
              elif counter==3:
                  lat_frame_max.append(float(pnt.split(',')[1]))
              counter+=1

        self.SNWE=[min(lats),max(lats),min(lons),max(lons)]

        # checking for missing gaps, by doing a difference between end and start of frames
        # will shift the vectors such that one can diff in rows to compare start and end
        # note need to keep temps seperate as otherwize one is using the other in calculation
        temp1 = max(lat_frame_max)
        temp2 = min(lat_frame_min)
        lat_frame_min.append(temp1)
        lat_frame_min.sort()
        lat_frame_max.append(temp2)
        lat_frame_max.sort()

        # combining the frame north and south left edge
        lat_frame_min = np.transpose(np.array(lat_frame_min))
        lat_frame_max = np.transpose(np.array(lat_frame_max))
        # if the differnce between top and bottom <=0 then there is overlap

        overlap_check = (lat_frame_min-lat_frame_max)<=0
        overlap_check = overlap_check.all()
        """if overlap_check:
            print(lat_frame_min)
            print(lat_frame_max)
            print(lat_frame_min-lat_frame_max)
            print("*********overlap")
        else:
            print(lat_frame_min)
            print(lat_frame_max)
            print(lat_frame_min-lat_frame_max)
            print("gap")"""

        #raise Exception("STOP")
        self.frame_nogap=overlap_check

    def get_lat_lon_v3(self,inps):
        from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1
        lats=[]
        lons=[]

        for swathnum in inps.swath_num.split():
           obj = Sentinel1()
           obj.configure()
           obj.safe = self.safe_file.split()
           obj.swathNumber = int(swathnum)
           print(obj.polarization)
           # add by Minyan
           obj.polarization='vv'
          #obj.output = '{0}-SW{1}'.format(safe,swathnum)
           obj.parse()

           s,n,w,e = obj.product.bursts[0].getBbox()
           lats.append(s);lats.append(n)
           lons.append(w);lons.append(e)

           s,n,w,e = obj.product.bursts[-1].getBbox()
           lats.append(s);lats.append(n)
           lons.append(w);lons.append(e)

        self.SNWE=[min(lats),max(lats),min(lons),max(lons)]

    def get_orbit(self, orbitDir, workDir, margin=60.0):
        margin = datetime.timedelta(seconds=margin)
        datefmt = "%Y%m%dT%H%M%S"
        orbit_files = glob.glob(os.path.join(orbitDir,  self.platform + '*.EOF'))
        if len(orbit_files) == 0:
            orbit_files = glob.glob(os.path.join(orbitDir, '*/{0}*.EOF'.format(self.platform)))

        match = False
        for orbit in orbit_files:
           orbit = os.path.basename(orbit)
           fields = orbit.split('_')
           orbit_start_date_time = datetime.datetime.strptime(fields[6].replace('V',''), datefmt) + margin
           orbit_stop_date_time = datetime.datetime.strptime(fields[7].replace('.EOF',''), datefmt) - margin

           if self.start_date_time >= orbit_start_date_time and self.stop_date_time < orbit_stop_date_time:
               self.orbit = os.path.join(orbitDir,orbit)
               self.orbitType = 'precise'
               match = True
               break
        if not match:
           print ("*****************************************")
           print (self.date)
           print ("orbit was not found in the "+orbitDir) # It should go and look online
           print ("downloading precise or restituted orbits ...")

           restitutedOrbitDir = os.path.join(workDir ,'orbits/' + self.date)
           orbitFiles = glob.glob(os.path.join(restitutedOrbitDir,'*.EOF'))
           if len(orbitFiles) > 0:
              orbitFile = orbitFiles[0]

              #fields = orbitFile.split('_')
              fields = os.path.basename(orbitFile).split('_')
              orbit_start_date_time = datetime.datetime.strptime(fields[6].replace('V',''), datefmt)
              orbit_stop_date_time = datetime.datetime.strptime(fields[7].replace('.EOF',''), datefmt)
              if self.start_date_time >= orbit_start_date_time and self.stop_date_time < orbit_stop_date_time:
                  print ("restituted or precise orbit already exists.")
                  self.orbit =  orbitFile
                  self.orbitType = 'restituted'

           #if not os.path.exists(restitutedOrbitDir):
           else:
              os.makedirs(restitutedOrbitDir, exist_ok=True)

              cmd = 'fetchOrbit.py -i ' + self.safe_file + ' -o ' + restitutedOrbitDir
              print(cmd)
              os.system(cmd)
              orbitFile = glob.glob(os.path.join(restitutedOrbitDir,'*.EOF'))
              self.orbit =  orbitFile[0]
              self.orbitType = 'restituted'

# an example for writing job files when using clusters

"""
def writeJobFile(runFile):

  jobName = runFile + '.job'
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
#PBS -m bae -M hfattahi@gps.caltech.edu

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
  f.write('parallel --sshloginfile $PBS_NODEFILE  -a ' + os.path.basename(runFile) + '\n')
  f.write('')
  f.close()

"""
