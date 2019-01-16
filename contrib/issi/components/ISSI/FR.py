import os
import logging
import tempfile
from isceobj.Filter.Filter import Filter
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from isceobj.Util.mathModule import MathModule as MM
# Because of Fortran 77 quirks, this string needs to end with a $
# This allows the path to be passed into the code
# and then 'trimmed' to the correct size
# There are better ways to do this, but this works for now.
dataPath = os.path.join(os.path.split(os.path.abspath(__file__))[0],
    'igrf_data$')

class FR(object):

    def __init__(self,hhFile=None,hvFile=None,vhFile=None,vvFile=None,lines=None,samples=None,
                 frOutput=None, tecOutput=None, phaseOutput=None):
        """
        Constructor

        @param hhFile (\a string) the file name containing the HH polarity data
        @param hvFile (\a string) the file name containing the HV polarity data
        @param vhFile (\a string) the file name containing the VH polarity data
        @param vvFile (\a string) the file name containing the VV polarity data
        @param lines (\a int) the number of ranges lines in each of the data files
        @param samples (\a int) the number of range bins in each line of each data file
        @param frOutput (\a string) the output file name for the Faraday rotation image
        @param tecOutput (\a string) the output file name for the Total Electron Count (TEC) image
        @param phaseOutput (\a string) the output file name for the phase delay image
        """
        self.hhFile = hhFile
        self.hvFile = hvFile
        self.vhFile = vhFile
        self.vvFile = vvFile
        self.frOutput = frOutput
        self.tecOutput = tecOutput
        self.phaseOutput = phaseOutput
        self.lines = lines
        self.samples = samples
        self.averageFaradayRotation = None
        # The ionospheric layer parameters
        self.top = 691.65 # top of the ionosphere in km
        self.bottom = 100.0 # bottom of the ionosphere in km
        self.step = 10.0 # height increment in km
        self.logger = logging.getLogger("contrib.ISSI")

    def getAverageFaradayRotation(self):
        return self.averageFaradayRotation

    def polarimetricCorrection(self,transmit,receive):
        """
        Apply the polarimetic calibration.

        @param transmit (\a isceobj.Sensor.Polarimetry.Distortion) The transmission distortion parameters
        @param receive (\a isceobj.Sensor.Polarimetry.Distortion) The reception distortion parameters
        """
        from ctypes import cdll, c_int, c_char_p, c_float
        lib = cdll.LoadLibrary(os.path.dirname(__file__) + '/issi.so')

        hhOutFile = self.hhFile.replace('.slc','_cal.slc')
        hvOutFile = self.hvFile.replace('.slc','_cal.slc')
        vhOutFile = self.vhFile.replace('.slc','_cal.slc')
        vvOutFile = self.vvFile.replace('.slc','_cal.slc')

        hhFile_c = c_char_p(self.hhFile.encode("utf-8"))
        hvFile_c = c_char_p(self.hvFile.encode("utf-8"))
        vhFile_c = c_char_p(self.vhFile.encode("utf-8"))
        vvFile_c = c_char_p(self.vvFile.encode("utf-8"))
        hhOutFile_c = c_char_p(hhOutFile.encode("utf-8"))
        hvOutFile_c = c_char_p(hvOutFile.encode("utf-8"))
        vhOutFile_c = c_char_p(vhOutFile.encode("utf-8"))
        vvOutFile_c = c_char_p(vvOutFile.encode("utf-8"))
        # Unpack the transmit and receive distortion matrices
        transmitCrossTalk1Real_c = c_float(transmit.getCrossTalk1().real)
        transmitCrossTalk1Imag_c = c_float(transmit.getCrossTalk1().imag)
        transmitCrossTalk2Real_c = c_float(transmit.getCrossTalk2().real)
        transmitCrossTalk2Imag_c = c_float(transmit.getCrossTalk2().imag)
        transmitChannelImbalanceReal_c = c_float(transmit.getChannelImbalance().real)
        transmitChannelImbalanceImag_c = c_float(transmit.getChannelImbalance().imag)
        receiveCrossTalk1Real_c = c_float(receive.getCrossTalk1().real)
        receiveCrossTalk1Imag_c = c_float(receive.getCrossTalk1().imag)
        receiveCrossTalk2Real_c = c_float(receive.getCrossTalk2().real)
        receiveCrossTalk2Imag_c = c_float(receive.getCrossTalk2().imag)
        receiveChannelImbalanceReal_c = c_float(receive.getChannelImbalance().real)
        receiveChannelImbalanceImag_c = c_float(receive.getChannelImbalance().imag)
        samples_c = c_int(self.samples)
        lines_c = c_int(self.lines)

        self.logger.info("Applying polarimetric correction")
        lib.polcal(hhFile_c,hvFile_c,vhFile_c,vvFile_c,hhOutFile_c,hvOutFile_c,vhOutFile_c,vvOutFile_c,
                   transmitCrossTalk1Real_c, transmitCrossTalk2Real_c, transmitChannelImbalanceReal_c,
                   transmitCrossTalk1Imag_c, transmitCrossTalk2Imag_c, transmitChannelImbalanceImag_c,
                   receiveCrossTalk1Real_c, receiveCrossTalk2Real_c, receiveChannelImbalanceReal_c,
                   receiveCrossTalk1Imag_c, receiveCrossTalk2Imag_c, receiveChannelImbalanceImag_c,
                   self.samples,self.lines)

        # Move change the reference files to the calibrated files
        self.hhFile = hhOutFile
        self.hvFile = hvOutFile
        self.vhFile = vhOutFile
        self.vvFile = vvOutFile

    def calculateFaradayRotation(self,filter=False,filterSize=None,swap=True):
        """
        Create a map of Faraday Rotation from quad-pol SAR data

        @param filter (\a boolean) True if spatial filtering is desired, default is False
        @param filterSize (\a tuple) a tuple containing the filter size in the range and azimuth direction specified by (range size, azimuth
        size)
        @param swap (\a boolean) enable byte-swapping, default is True
        """
        from ctypes import cdll, c_int, c_float, c_char_p
        lib = cdll.LoadLibrary(os.path.dirname(__file__) + '/issi.so')
        input = self._combinePolarizations(swap=swap)
        if (filter):
            input = self._filterFaradayRotation(input,filter,filterSize)
        self.logger.info("Calculating Faraday Rotation")

        input_c = c_char_p(input.name)
        output_c = c_char_p(self.frOutput)
        samples_c = c_int(self.samples)
        lines_c = c_int(self.lines)

        lib.cfrToFr.restype = c_float
        self.averageFaradayRotation = lib.cfrToFr(input_c,output_c,samples_c,lines_c)

        # Create a resource file for the Faraday Rotation output file
        rsc = ResourceFile(self.frOutput + '.rsc')
        rsc.write('WIDTH',self.samples)
        rsc.write('FILE_LENGTH',self.lines)
        if (filter):
            rsc.write('FILTER_SIZE_RANGE',filterSize[0])
            rsc.write('FILTER_SIZE_AZIMUTH',filterSize[1])
        rsc.close()

    def frToTEC(self,date,corners,lookAngle,lookDirection,fc):
        """
        Given a list of geodetic coordinates, a list of lookAngles and a list of lookDirections,
        calculate the average value of the B-field in the radar line-of-sight. Look angles are
        calculated in degrees from the nadir and look directions are calculated in degrees from
        the perpendicular to the flight direction.

        @param date (\a datetime.datetime) the date on which to calculate the B-field
        @param corners (\a list) a list of Location.Coordinate objects specifying the corners of the radar image
        @param lookAngle (\a list) a list of the look angles (in degrees) to each corner of the radar image
        @param lookDirection (\a list) a list of the look directions (in degrees) to each corner of the radar image
        @param fc (\a float) the radar carrier frequency in Hz
        @return (\a float) the mean value of the B-field in the look direction of the radar in gauss
        """
        kdotb = []
        # Calculate the integrated B vector for each of the four corners of the interferogram
        # Need to get the date from any of the Frame objects associated with one of the polarities
        for i,coordinate in enumerate(corners):
            k = self._calculateLookVector(lookAngle[i],lookDirection[i])
            kdotb.append(self._integrateBVector(date,coordinate,k))

        # Use this value to convert from Faraday rotation to TEC
        meankdotb = MM.mean(kdotb)
        self.logger.info("Creating TEC Map")
        self._scaleFRToTEC(meankdotb,fc)

        # Create a resource file for the TEC output file
        rsc = ResourceFile(self.tecOutput + '.rsc')
        rsc.write('WIDTH',self.samples)
        rsc.write('FILE_LENGTH',self.lines)
        rsc.write('MEAN_K_DOT_B',meankdotb)
        rsc.write('LOOK_DIRECTION',lookDirection[0])
        for i in range(len(corners)):
            lattag = 'LAT_CORNER_' + str((i+1))
            lontag = 'LON_CORNER_' + str((i+1))
            looktag = 'LOOK_ANGLE_' + str((i+1))
            rsc.write(lattag,corners[i].getLatitude())
            rsc.write(lontag,corners[i].getLongitude())
            rsc.write(looktag,lookAngle[i])
        rsc.close()

        return meankdotb

    def tecToPhase(self,fc):
        """
        Apply a scalar value to convert from Total Electron Count (TEC) to Phase in radians.

        @param fc (\a float) the carrier frequency of the radar
        """
        from ctypes import cdll, c_float, c_int,c_char_p
        lib = cdll.LoadLibrary(os.path.dirname(__file__) + '/issi.so')

        inFile_c = c_char_p(self.tecOutput)
        outFile_c = c_char_p(self.phaseOutput)
        width_c = c_int(self.samples)
        fc_c = c_float(fc)

        lib.convertToPhase(inFile_c,outFile_c,width_c,fc_c)

        # Create a resource file for the phase output
        rsc = ResourceFile(self.phaseOutput + '.rsc')
        rsc.write('WIDTH',self.samples)
        rsc.write('FILE_LENGTH',self.lines)
        rsc.close()

    def _combinePolarizations(self,swap=True):
        """
        Combine the four polarizations using the method of Bickel & Bates (1965).
        @note: Bickel, S. H., and R. H. T. Bates (1965), Effects of magneto-ionic propagation on the polarization scattering matrix,
        pp. 1089--1091.

        @param swap (\a boolean) enable byte-swapping, default is True
        @return (\a string) the temporary file name containing the combined polarization channels
        """
        from ctypes import cdll, c_int, c_char_p
        self.logger.info("Combining Polarizations")
        lib = cdll.LoadLibrary(os.path.dirname(__file__) + '/issi.so')

        output = tempfile.NamedTemporaryFile()
        output_c = c_char_p(output.name)
        hhFile_c = c_char_p(self.hhFile)
        hvFile_c = c_char_p(self.hvFile)
        vhFile_c = c_char_p(self.vhFile)
        vvFile_c = c_char_p(self.vvFile)
        samples_c = c_int(self.samples)
        lines_c = c_int(self.lines)
        swap_c = None
        if (swap):
            self.logger.debug("Byte swapping")
            swap_c = c_int(0) # 0 for byte swapping, 1 for no byte swapping
        else:
            self.logger.debug("Not Byte swapping")
            swap_c = c_int(1)
        lib.cfr(hhFile_c,hvFile_c,vhFile_c,vvFile_c,output_c,samples_c,lines_c,swap_c)

        return output

    def _filterFaradayRotation(self,infile,filterType,filterSize):
        """
        Apply a filter to the intermediate Faraday rotation product.

        @param infile (\a string) the file name containing the complex*8 data to be filtered
        @param filterType (\a string) the filter type, may be 'median', 'gaussian', or 'mean'
        @param filterSize (\a list) a list containing the range and azimuth filter sizes
        @return (\a string) a file name containing the filtered complex*8 data
        @throws NotImplementedError: if filterType is not implemented
        """
        outfile = tempfile.NamedTemporaryFile()
        filter = Filter(inFile=infile.name, outFile=outfile.name, width=self.samples, length=self.lines)

        #2013-06-04 Kosal
        filterType = filterType.title()
        filterWidth, filterHeight = filterSize
        if (filterType == 'Median'):
            filter.medianFilter(filterWidth, filterHeight)
        elif (filterType == 'Gaussian'):
            width = filterWidth
            sigma = ( ( (width - 1) / 2 ) / 3.0 )**2 # Thus "stretches" the Gaussian so that the 3-sigma level occurs at the edge of the filter
            filter.gaussianFilter(filterWidth, filterHeight, sigma)
        elif (filterType == 'Mean'):
            filter.meanFilter(filterWidth, filterHeight)
        else:
            self.logger.error("Filter type %s is not currently supported" % (filterType))
            raise NotImplementedError()

        self.logger.info("%s Filtering with a %dx%d filter" % (filterType, filterWidth, filterHeight))
        #Kosal

        return outfile

    def _scaleFRToTEC(self,meankdotb,fc):
        """
        Apply a scalar value to convert from Faraday Rotation to Total Electron Count.

        @param meankdotb (\a float) the mean value of the B-field in the look direction of the radar
        @param fc (\a float) the carrier frequency of the radar in Hz
        """
        from ctypes import cdll, c_float, c_int,c_char_p
        lib = cdll.LoadLibrary(os.path.dirname(__file__) + '/issi.so')

        inFile_c = c_char_p(self.frOutput)
        outFile_c = c_char_p(self.tecOutput)
        width_c = c_int(self.samples)
        bdotk_c = c_float(meankdotb)
        fc_c = c_float(fc)

        lib.convertToTec(inFile_c,outFile_c,width_c,bdotk_c,fc_c)

    def _integrateBVector(self,date,coordinate,k):
        """
        Integrate the B-field estimates through the ionosphere at the specified date and location

        @param date (\a datetime.datetime) date at which to calculate the B-field
        @param coordinate (\a isceobj.Location.Coordinate) the coordinate at which to calculate the B-field.
        @param k (\a list) the look vector of the radar
        @return (\a float) the integrated value of the B-field at the specified date and location in gauss
        """

        kdotb = []
        n_altitude = int((self.top - self.bottom)/self.step) + 1
        altitude = [self.bottom + i*self.step for i  in range(n_altitude)]
        for h in altitude:
            coordinate.setHeight(h)
            bvector = self._calculateBVector(date,coordinate)
            kdotb.append(MM.dotProduct(k,bvector))

        meankdotb = MM.mean(kdotb)

        return meankdotb

    def _calculateBVector(self,date,coordinate):
        """
        Given a date, and a coordinate, calculate the value of the B-field.

        @param date (\a float) the decimal year at which to calulate the B-field
        @param coordinate (\a isceobj.Location.Coordinate) the location at which to calculate the B-field
        @return (\a list) the north, east and down values of the B-field in gauss
        """
        from ctypes import cdll, c_float, c_char_p
        lib = cdll.LoadLibrary(os.path.dirname(__file__) + '/issi.so')

        year = DTU.dateTimeToDecimalYear(date)
        year_c = c_float(year)
        lat_c = c_float(coordinate.getLatitude())
        lon_c = c_float(coordinate.getLongitude())
        alt_c = c_float(coordinate.getHeight())
        beast_c  = (c_float*1)(*[0.0])
        bnorth_c = (c_float*1)(*[0.0])
        bdown_c  = (c_float*1)(*[0.0])
        babs_c   = (c_float*1)(*[0.0])
        dataPath_c = c_char_p(dataPath) # Point to the library directory

        lib.calculateBVector(year_c,lat_c,lon_c,alt_c,beast_c,bnorth_c,bdown_c,babs_c,dataPath_c)

        beast = beast_c[0]
        bnorth = bnorth_c[0]
        bdown = bdown_c[0]

        return [beast, bnorth, bdown]

    def _calculateLookVector(self,lookAngle,lookDirection):
        """
        Calculate the look vector of the radar from the look direction and look angle.

        @param lookAngle (\a float) the look angle of the radar measured from the nadir direction in degrees
        @param lookDirection (\a float) the look direction of the radar measured from the direction perpendicular to the flight direction in
        degrees
        @return (\a list) the cartesian look vector
        """
        import math
        x = math.sin(math.radians(lookAngle))*math.sin(math.radians(lookDirection))
        y = math.sin(math.radians(lookAngle))*math.cos(math.radians(lookDirection))
        z = -math.cos(math.radians(lookAngle))

        return [x,y,z]

class ResourceFile(object):
    """A simple resource file generator"""

    def __init__(self,filename):
        """
        Constructor

        @param filename (\a string) the resource file name
        """
        self.file = open(filename,'w')

    def close(self):
        """
        Explicitly close the resource file
        """
        self.file.close()

    def write(self,keyword,value):
        """
        Write a keyword-value pair into the resource file

        @param keyword (\a string) a resource file keyword
        @param value (\a string) a resource file value
        """
        keyword = keyword.upper()
        keyword = keyword.replace(' ','_')
        value = str(value)
        self.file.write(keyword + ' ' + value + "\n")
