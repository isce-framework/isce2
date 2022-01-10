#!/usr/bin/env python3
#
#Copyright 2010, by the California Institute of Technology.
#ALL RIGHTS RESERVED.
#United States Government Sponsorship acknowledged.
#Any commercial use must be negotiated with the Office of
#Technology Transfer at the California Institute of Technology.
#
#This software may be subject to U.S. export control laws. By
#accepting this software, the user agrees to comply with all applicable
#U.S. export laws and regulations. User has the responsibility to obtain
#export licenses, or other export authority as may be required before
#exporting such information to foreign countries or providing access
#to foreign persons.
#
import isce
import sys
import os
from sys import float_info
import logging
import datetime
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import Orbit
from isceobj.Attitude.Attitude import Attitude
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from isceobj.Util.decorators import type_check, logged, pickled
import isceobj
import tempfile

@pickled
class Track(object):
    """A class to represent a collection of temporally continuous radar frame
    objects"""

    logging_name = "isce.Scene.Track"

    @logged
    def __init__(self):
        # These are attributes representing the starting time and stopping
        # time of the track
        # As well as the early and late times (range times) of the track
        self._startTime = datetime.datetime(year=datetime.MAXYEAR,month=1,day=1)
        self._stopTime = datetime.datetime(year=datetime.MINYEAR,month=1,day=1)
        # Hopefully this number is large
        # enough, Python doesn't appear to have a MAX_FLT variable
        self._nearRange = float_info.max
        self._farRange = 0.0
        self._frames = []
        self._frame = Frame()
        self._lastFile = ''
        return None

    def combineFrames(self, output, frames):
        attitudeOk = True
        for frame in frames:
            self.addFrame(frame)
            if hasattr(frame,'_attitude'):
                att = frame.getAttitude()
                if not att:
                    attitudeOk = False
        self.createInstrument()
        self.createTrack(output)
        self.createOrbit()
        if attitudeOk:
            self.createAttitude()
        return self._frame

    def createAuxFile(self, fileList, output):
        import struct
        from operator import itemgetter
        import os
        import array
        import copy
        dateIndx = []
        cnt = 0
        #first sort the files from earlier to latest. use the first element
        for name in fileList:
            with open(name,'rb') as fp: date = fp.read(16)
            day, musec = struct.unpack('<dd',date)
            dateIndx.append([day,musec,name])
            cnt += 1
        sortedDate = sorted(dateIndx, key=itemgetter(0,1))

        #we need to make sure that there are not duplicate points in the orbit since some frames overlap
        allL = array.array('d')
        allL1 = array.array('d')
        name = sortedDate[0][2]
        size = os.path.getsize(name)//8
        with open(name,'rb') as fp1: allL.fromfile(fp1,size)
        lastDay = allL[-2]
        lastMusec = allL[-1]
        for j in range(1, len(sortedDate)):
            name = sortedDate[j][2]
            size = os.path.getsize(name)//8
            with open(name,'rb') as fp1: allL1.fromfile(fp1, size)
            indxFound = None
            avgPRI = 0
            cnt = 0
            for i in range(len(allL1)//2):
                if i > 0:
                    avgPRI += allL1[2*i+1] - allL1[2*i-1]
                    cnt += 1
                if allL1[2*i] >= lastDay and allL1[2*i+1] > lastMusec:
                    avgPRI //= (cnt-1)
                    if (allL1[2*i+1] - lastMusec) > avgPRI/2:# make sure that the distance in pulse is atleast 1/2 PRI
                        indxFound = 2*i
                    else:#if not take the next
                        indxFound = 2*(i+1)
                        pass
                    break
            if not indxFound is None:
                allL.extend(allL1[indxFound:])
                lastDay = allL[-2]
                lastMusec = allL[-1]
                pass
            pass
        with open(output,'wb') as fp: allL.tofile(fp)
        return

    # Add an additional Frame object to the track
    @type_check(Frame)
    def addFrame(self, frame):
        self.logger.info("Adding Frame to Track")
        self._updateTrackTimes(frame)
        self._frames.append(frame)
        return None

    def createOrbit(self):
        orbitAll = Orbit()
        for i in range(len(self._frames)):
            orbit = self._frames[i].getOrbit()
            #remember that everything is by reference, so the changes applied to orbitAll will be made to the Orbit
            #object in self.frame
            for sv in orbit._stateVectors:
                orbitAll.addStateVector(sv)
            # sort the orbit state vecotrs according to time
            orbitAll._stateVectors.sort(key=lambda sv: sv.time)
        self.removeDuplicateVectors(orbitAll._stateVectors)
        self._frame.setOrbit(orbitAll)

    def removeDuplicateVectors(self,stateVectors):
        i1 = 0
        #remove duplicate state vectors
        while True:
            if i1 >= len(stateVectors) - 1:
                break
            if stateVectors[i1].time == stateVectors[i1+1].time:
                stateVectors.pop(i1+1)
            #since is sorted by time if is not equal we can pass to the next
            else:
                i1 += 1


    def createAttitude(self):
        attitudeAll = Attitude()
        for i in range(len(self._frames)):
            attitude = self._frames[i].getAttitude()
            #remember that everything is by reference, so the changes applied to attitudeAll will be made to the Attitude object in self.frame
            for sv in attitude._stateVectors:
                attitudeAll.addStateVector(sv)
            # sort the attitude state vecotrs according to time
            attitudeAll._stateVectors.sort(key=lambda sv: sv.time)
        self.removeDuplicateVectors(attitudeAll._stateVectors)
        self._frame.setAttitude(attitudeAll)

    def createInstrument(self):
        # the platform is already part of the instrument
        ins = self._frames[0].getInstrument()
        self._frame.setInstrument(ins)

    # sometime the startLine computed below from the sensingStart is not
    #precise and the image are missaligned.
    #for each pair do an exact mach by comparing the lines around lineStart
    #file1,2 input files, startLine1 is the estimated start line in the first file
    #line1 = last line used in the first file
    #width = width of the files
    #frameNum1,2 number of the frames in the sequence of frames to stitch
    #returns a more accurate line1
    def findOverlapLine(self, file1, file2, line1,width,frameNum1,frameNum2):
        import numpy as np
        import array
        fin2 = open(file2,'rb')
        arr2 = array.array('b')
        #read full line at the beginning of second file
        arr2.fromfile(fin2,width)
        buf2 = np.array(arr2,dtype = np.int8)
        numTries = 30
        # start around line1 and try numTries around line1
        # see searchlist to see which lines it searches
        searchNumLines = 2
        #make a sliding window that search for the searchSize samples inside buf2
        searchSize = 500
        max = 0
        indx = None
        fin1 = open(file1,'rb')
        for i in range(numTries):
            # example line1 = 0,searchNumLine = 2 and i = 0 search = [-2,-1,0,1], i = 1, serach =  [-4,-3,2,3]
            search = list(range(line1 - (i+1)*searchNumLines,line1 - i*searchNumLines))
            search.extend(list(range(line1 + i*searchNumLines,line1 + (i+1)*searchNumLines)))
            for k in search:
                arr1 = array.array('b')
                #seek to the line k and read +- searchSize/2 samples from the middle of the line
                fin1.seek(k*width + (width - searchSize)//2,0)
                arr1.fromfile(fin1,searchSize)
                buf1 = np.array(arr1,dtype = np.int8)
                found = False
                for i in np.arange(width-searchSize):
                    lenSame =len(np.nonzero(buf1 == buf2[i:i+searchSize])[0])
                    if  lenSame > max:
                        max = lenSame
                        indx = k
                        if(lenSame == searchSize):
                            found = True
                            break
                if(found):
                    break
            if(found):
                break
        if not found:
            self.logger.warning("Cannot find perfect overlap between frame %d and frame %d. Using acquisition time to find overlap position."%(frameNum1,frameNum2))
        fin1.close()
        fin2.close()
        print('Match found: ', indx)
        return indx

    def reAdjustStartLine(self, sortedList, width):
        """ Computed the adjusted starting lines based on matching in overlapping regions """
        from operator import itemgetter
        import os

        #first one always starts from zero
        startLine =  [sortedList[0][0]]
        outputs =  [sortedList[0][1]]
        for i in range(1,len(sortedList)):
            # endLine of the first file. we use all the lines of the first file up to endLine
            endLine = sortedList[i][0] - sortedList[i-1][0]
            indx = self.findOverlapLine(sortedList[i-1][1],sortedList[i][1],endLine,width,i-1,i)
            #if indx is not None than indx is the new start line
            #otherwise we use startLine  computed from acquisition time
            #no need to do this for ALOS; otherwise there will be problems when there are multiple prfs and the data are interpolated. C. Liang, 20-dec-2021
            if (self._frames[0].instrument.platform._mission != 'ALOS') and (indx is not None) and (indx + sortedList[i-1][0] != sortedList[i][0]):
                startLine.append(indx + sortedList[i-1][0])
                outputs.append(sortedList[i][1])
                self.logger.info("Changing starting line for frame %d from %d to %d"%(i,endLine,indx))
            else:
                startLine.append(sortedList[i][0])
                outputs.append(sortedList[i][1])

        return startLine,outputs



    # Create the actual Track data by concatenating data from
    # all of the Frames objects together
    def createTrack(self,output):
        import os
        from operator import itemgetter
        from isceobj import Constants as CN
        from ctypes import cdll, c_char_p, c_int, c_ubyte,byref
        lib = cdll.LoadLibrary(os.path.dirname(__file__)+'/concatenate.so')
        # Perhaps we should check to see if Xmin is 0, if it is not, strip off the header
        self.logger.info("Adjusting Sampling Window Start Times for all Frames")
        # Iterate over each frame object, and calculate the number of samples with which to pad it on the left and right
        outputs = []
        totalWidth = 0
        auxList = []
        for frame in self._frames:
            # Calculate the amount of padding
            thisNearRange = frame.getStartingRange()
            thisFarRange = frame.getFarRange()
            left_pad = int(round(
                (thisNearRange - self._nearRange)*
                frame.getInstrument().getRangeSamplingRate()/(CN.SPEED_OF_LIGHT/2.0)))*2
            right_pad = int(round((self._farRange - thisFarRange)*frame.getInstrument().getRangeSamplingRate()/(CN.SPEED_OF_LIGHT/2.0)))*2
            width = frame.getImage().getXmax()
            if width - int(width) != 0:
                raise ValueError("frame Xmax is not an integer")
            else:
                width = int(width)

            input = frame.getImage().getFilename()
#            tempOutput = os.path.basename(os.tmpnam()) # Some temporary filename
            with tempfile.NamedTemporaryFile(dir='.') as f:
                tempOutput = f.name

            pad_value = int(frame.getInstrument().getInPhaseValue())

            if totalWidth < left_pad + width + right_pad:
                totalWidth = left_pad + width + right_pad
            # Resample this frame with swst_resample
            input_c = c_char_p(bytes(input,'utf-8'))
            output_c = c_char_p(bytes(tempOutput,'utf-8'))
            width_c = c_int(width)
            left_pad_c = c_int(left_pad)
            right_pad_c = c_int(right_pad)
            pad_value_c = c_ubyte(pad_value)
            lib.swst_resample(input_c,output_c,byref(width_c),byref(left_pad_c),byref(right_pad_c),byref(pad_value_c))
            outputs.append(tempOutput)
            auxList.append(frame.auxFile)

        #this step construct the aux file withe the pulsetime info for the all set of frames
        self.createAuxFile(auxList,output + '.aux')
        # This assumes that all of the frames to be concatenated are sampled at the same PRI
        prf = self._frames[0].getInstrument().getPulseRepetitionFrequency()
        # Calculate the starting output line of each scene
        i = 0
        lineSort = []
        # the listSort has 2 elements: a line start number which is the position of that specific frame
        # computed from acquisition time and the  corresponding file name
        for frame in self._frames:
            startLine = int(round(DTU.timeDeltaToSeconds(frame.getSensingStart()-self._startTime)*prf))
            lineSort.append([startLine,outputs[i]])
            i += 1

        sortedList = sorted(lineSort, key=itemgetter(0)) # sort by line number i.e. acquisition time
        startLines, outputs = self.reAdjustStartLine(sortedList,totalWidth)


        self.logger.info("Concatenating Frames along Track")
        # this is a hack since the length of the file could be actually different from the one computed using start and stop time. it only matters the last frame added
        import os

        fileSize = os.path.getsize(outputs[-1])

        numLines = fileSize//totalWidth + startLines[-1]
        totalLines_c = c_int(numLines)
        # Next, call frame_concatenate
        width_c = c_int(totalWidth) # Width of each frame (with the padding added in swst_resample)
        numberOfFrames_c = c_int(len(self._frames))
        inputs_c = (c_char_p * len(outputs))() # These are the inputs to frame_concatenate, but the outputs from swst_resample
        for kk in range(len(outputs)):
            inputs_c[kk] = bytes(outputs[kk],'utf-8')
        output_c = c_char_p(bytes(output,'utf-8'))
        startLines_c = (c_int * len(startLines))()
        startLines_c[:] = startLines
        lib.frame_concatenate(output_c,byref(width_c),byref(totalLines_c),byref(numberOfFrames_c),inputs_c,startLines_c)

        # Clean up the temporary output files from swst_resample
        for file in outputs:
            os.unlink(file)

        orbitNum = self._frames[0].getOrbitNumber()
        first_line_utc = self._startTime
        last_line_utc = self._stopTime
        centerTime = DTU.timeDeltaToSeconds(last_line_utc-first_line_utc)/2.0
        center_line_utc = first_line_utc + datetime.timedelta(microseconds=int(centerTime*1e6))
        procFac = self._frames[0].getProcessingFacility()
        procSys = self._frames[0].getProcessingSystem()
        procSoft = self._frames[0].getProcessingSoftwareVersion()
        pol = self._frames[0].getPolarization()
        xmin = self._frames[0].getImage().getXmin()


        self._frame.setOrbitNumber(orbitNum)
        self._frame.setSensingStart(first_line_utc)
        self._frame.setSensingMid(center_line_utc)
        self._frame.setSensingStop(last_line_utc)
        self._frame.setStartingRange(self._nearRange)
        self._frame.setFarRange(self._farRange)
        self._frame.setProcessingFacility(procFac)
        self._frame.setProcessingSystem(procSys)
        self._frame.setProcessingSoftwareVersion(procSoft)
        self._frame.setPolarization(pol)
        self._frame.setNumberOfLines(numLines)
        self._frame.setNumberOfSamples(width)
        # add image to frame
        rawImage = isceobj.createRawImage()
        rawImage.setByteOrder('l')
        rawImage.setFilename(output)
        rawImage.setAccessMode('r')
        rawImage.setWidth(totalWidth)
        rawImage.setXmax(totalWidth)
        rawImage.setXmin(xmin)
        self._frame.setImage(rawImage)


    # Extract the early, late, start and stop times from a Frame object
    # And use this information to update
    def _updateTrackTimes(self,frame):

        if (frame.getSensingStart() < self._startTime):
            self._startTime = frame.getSensingStart()
        if (frame.getSensingStop() > self._stopTime):
            self._stopTime = frame.getSensingStop()
        if (frame.getStartingRange() < self._nearRange):
            self._nearRange = frame.getStartingRange()
        if (frame.getFarRange() > self._farRange):
            self._farRange = frame.getFarRange()
            pass
        pass
    pass

def main():

    tr = Track()
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    line1 = 17731
    width = 21100
    indx = tr.findOverlapLine(file1, file2, line1,width,0,1)

if __name__ == '__main__':
    sys.exit(main())
