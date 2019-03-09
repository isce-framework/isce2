#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
import sys
import os
import math
import logging
import contextlib
from iscesys.Dumpers.XmlDumper import XmlDumper
from iscesys.Component.Configurable import Configurable
from iscesys.ImageApi.DataAccessorPy import DataAccessor
from iscesys.ImageApi import CasterFactory as CF
from iscesys.DictUtils.DictUtils import DictUtils as DU
from isceobj.Util import key_of_same_content
from isceobj.Util.decorators import pickled, logged
from iscesys.Component.Component import Component
import numpy as np
from isceobj.Util.decorators import use_api

# # \namespace ::isce.components.isceobj.Image Base class for Image API


# # This is the default copy list-- it is not a class attribute because the
# # I decided the class wwas too big-- but that's strictly subjective.
ATTRIBUTES = ('bands', 'scheme', 'caster', 'width', 'filename', 'byteOrder',
              'dataType', 'xmin', 'xmax', 'numberGoodBytes', 'firstLatitude',
              'firstLongitude', 'deltaLatitude', 'deltaLongitude')

# # Map various byte order codes to Image's.
ENDIAN = {'l':'l', 'L':'l', '<':'l', 'little':'l', 'Little':'l',
          'b':'b', 'B':'b', '>':'b', 'big':'b', 'Big':'b'}

# long could be machine dependent
sizeLong = DataAccessor.getTypeSizeS('LONG')
TO_NUMPY = {'BYTE':'i1', 'SHORT':'i2', 'INT':'i4', 'LONG':'i' + str(sizeLong), 'FLOAT':'f4', 'DOUBLE':'f8',
            'CFLOAT':'c8', 'CDOUBLE':'c16'}


BYTE_ORDER = Component.Parameter('byteOrder',
                           public_name='BYTE_ORDER',
                           default=sys.byteorder[0].lower(),
                           type=str,
                           mandatory=False,
                           doc='Endianness of the image.')
WIDTH = Component.Parameter('width',
                           public_name='WIDTH',
                           default=None,
                           type=int,
                           mandatory=False,
                           private=True,
                           doc='Image width')
LENGTH = Component.Parameter('length',
                           public_name='LENGTH',
                           default=None,
                           type=int,
                           mandatory=False,
                           private=True,
                           doc='Image length')
SCHEME = Component.Parameter('scheme',
                        public_name='SCHEME',
                        default='BIP',
                        type=str,
                        mandatory=False,
                        doc='Interleaving scheme of the image.')
CASTER = Component.Parameter('caster',
                        public_name='CASTER',
                        default='',
                        type=str,
                        mandatory=False,
                        private=True,
                        doc='Type of conversion to be performed from input '
                        + 'source to output source. Being input or output source will depend on the type of operations performed (read or write)')
NUMBER_BANDS = Component.Parameter('bands',
                       public_name='NUMBER_BANDS',
                       default=1,
                       type=int,
                       mandatory=False,
                       doc='Number of image bands.')

'''
COORD1 = Component.Parameter('coord1',
                       public_name='COORD1',
                       default=None,
                       type=int,
                       mandatory=True,
                       doc='Horizontal coordinate.')

COORD2 = Component.Parameter('coord2',
                        public_name='COORD2',
                        default=None,
                        type=int,
                        mandatory=True,
                        doc='Vertical coordinate.')
'''
DATA_TYPE = Component.Parameter('dataType',
                          public_name='DATA_TYPE',
                          default='',
                          type=str,
                          mandatory=True,
                          doc='Image data type.')
IMAGE_TYPE = Component.Parameter('imageType',
                           public_name='IMAGE_TYPE',
                           default='',
                           type=str,
                           mandatory=False,
                           private=True,
                           doc='Image type used for displaying.')
FILE_NAME = Component.Parameter('filename',
                          public_name='FILE_NAME',
                          default='',
                          type=str,
                          mandatory=True,
                          doc='Name of the image file.')
EXTRA_FILE_NAME = Component.Parameter('_extraFilename',
                          public_name='EXTRA_FILE_NAME',
                          default='',
                          type=str,
                          private=True,
                          mandatory=False,
                          doc='For example name of vrt metadata.')
ACCESS_MODE = Component.Parameter('accessMode',
                            public_name='ACCESS_MODE',
                            default='',
                            type=str,
                            mandatory=True,
                            doc='Image access mode.')
DESCRIPTION = Component.Parameter('description',
                              public_name='DESCRIPTION',
                              default='',
                              type=str,
                              mandatory=False,
                              private=True,
                              doc='Image description')
XMIN = Component.Parameter('xmin',
                       public_name='XMIN',
                       default=None,
                       type=float,
                       mandatory=False,
                       private=True,
                       doc='Minimum range value')
XMAX = Component.Parameter('xmax',
                       public_name='XMAX',
                       default=None,
                       type=float,
                       mandatory=False,
                       private=True,
                       doc='Maximum range value')
ISCE_VERSION = Component.Parameter('isce_version',
                               public_name='ISCE_VERSION',
                               default=None,
                               type=str,
                               mandatory=False,
                               private=True,
                               doc='Information about the isce release version.')


COORD1 = Component.Facility(
    'coord1',
    public_name='Coordinate1',
    module='isceobj.Image',
    factory='createCoordinate',
    args=(),
    mandatory=True,
    doc='First coordinate of a 2D image (width).'
)
COORD2 = Component.Facility(
    'coord2',
    public_name='Coordinate2',
    module='isceobj.Image',
    factory='createCoordinate',
    args=(),
    mandatory=True,
    doc='Second coordinate of a 2D image (length).'
)

@pickled
class Image(DataAccessor, Configurable):

    logging_name = 'isce.isceobj.Image.Image'
    parameter_list = (
                      BYTE_ORDER,
                      SCHEME,
                      CASTER,
                      NUMBER_BANDS,
                      WIDTH,
                      LENGTH,
                      DATA_TYPE,
                      IMAGE_TYPE,
                      FILE_NAME,
                      EXTRA_FILE_NAME,
                      ACCESS_MODE,
                      DESCRIPTION,
                      XMIN,
                      XMAX,
                      ISCE_VERSION
                      )
    facility_list = (
                     COORD1,
                     COORD2
                     )
    family = 'image'
    def __init__(self, family='', name=''):
        # There is an hack to set the first latitude and longitude (see setters) so coord1 and 2
        # need to be defined when calling Configurable.__init__ which will try to call the setters
        self.catalog = {}
        self.descriptionOfVariables = {}
        self.descriptionOfFacilities = {}
        self._dictionaryOfFacilities = {}

        self.typeOfVariables = {}
        self.unitsOfVariables = {}
        self.dictionaryOfOutputVariables = {}
        self.dictionaryOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []

        # since we hacked the with to call coord1 the facilities need to be defined when calling
        # Configurable.__init__
        self._facilities()

        self.updateParameters()
        DataAccessor.__init__(self)
        Configurable.__init__(self, family if family else  self.__class__.family, name=name)
        self._instanceInit()
        self._isFinalized = False
        return None
    # To facilitate the use of numpy to manipulate isce images
    def toNumpyDataType(self):
        return TO_NUMPY[self.dataType.upper()]

    def updateParameters(self):
        self.extendParameterList(Configurable, Image)
        super(Image, self).updateParameters()

    # # New usage is: image.copy_attribute(image', *args), replacing:
    # # ImageUtil.ImageUtil.ImageUtil.copyAttributes(image, image', *args)
    def copy_attributes(self, other, *args):
        for item in args or ATTRIBUTES:
            try:
                setattr(other, item, getattr(self, item))
            except AttributeError:
                pass
        return other

    # Why reinventing the wheel when there is deepcopy
    # # This method makes a new image sub-class object that are copies of
    # # existing ones.
    def copy(self, access_mode=None):
        obj_new = self.copy_attributes(self.__class__())
        if access_mode:
            obj_new.setAccessMode(access_mode)
        obj_new.createImage()
        return obj_new

    def clone(self, access_mode=None):
        import copy
        obj_new = copy.deepcopy(self)
        if access_mode:
            obj_new.setAccessMode(access_mode)
        return obj_new
    # # Call the copy method, as a context manager
    @contextlib.contextmanager
    def ccopy(self, access_mode=None):
        result = self.copy(access_mode=access_mode)
        yield result
        result.finalizeImage()
        pass

    # # creates a DataAccessor.DataAccessor instance. If the parameters tagged
    # # as mandatory are not set, an exception is thrown.
    def createImage(self):
        self.createAccessor()
        da = self.getAccessor()

        ###Intercept for GDAL
        if self.methodSelector() != 'api':
            return None

        try:
            fsize = os.path.getsize(self.filename)
        except OSError:
            print("File", self.filename, "not found")
            raise OSError
        size = self.getTypeSize()
        if(fsize != self.width * self.length * size * self.bands):
            print("Image.py::createImage():Size on disk and  size computed from metadata for file", \
                  self.filename, "do not match")
            sys.exit(1)
        self._isFinalized = False
        return None

    def memMap(self, mode='r', band=None):
        if self.scheme.lower() == 'bil':
            immap = np.memmap(self.filename, self.toNumpyDataType(), mode,
                            shape=(self.coord2.coordSize , self.bands, self.coord1.coordSize))
            if band is not None:
                immap = immap[:, band, :]
        elif self.scheme.lower() == 'bip':
            immap = np.memmap(self.filename, self.toNumpyDataType(), mode,
                              shape=(self.coord2.coordSize, self.coord1.coordSize, self.bands))
            if band is not None:
                immap = immap[:, :, band]
        elif self.scheme.lower() == 'bsq':
            immap = np.memmap(self.filename, self.toNumpyDataType(), mode,
                        shape=(self.bands, self.coord2.coordSize, self.coord1.coordSize))
            if band is not None:
                immap = immap[band, :, :]
        return immap

    def asMemMap(self, filename):
        if self.scheme.lower() == 'bil':
            immap = np.memmap(filename, self.toNumpyDataType(), 'w+',
                        shape=(self.coord2.coordSize , self.bands, self.coord1.coordSize))
        elif self.scheme.lower() == 'bip':
            immap = np.memmap(filename, self.toNumpyDataType(), 'w+',
                        shape=(self.coord2.coordSize, self.coord1.coordSize, self.bands))
        elif self.scheme.lower() == 'bsq':
            immap = np.memmap(filename, self.toNumpyDataType(), 'w+',
                        shape=(self.bands, self.coord2.coordSize, self.coord1.coordSize))
        return immap


    # intercept the dump method and the adaptToRender to make sure the the coor2.coordSize is set.
    # the assignment does the trick

    @use_api
    def dump(self, filename):
        self.length = self.length
        super(Image, self).dump(filename)
        self.renderVRT()

    @use_api
    def adaptToRender(self):
        self.length = self.length

    '''
    ##
    # Initialize the image instance from an xml file
    def load(self,filename):
        from iscesys.Parsers.FileParserFactory import createFileParser
        parser = createFileParser('xml')
        #get the properties from the file
        prop, fac, misc = parser.parse(filename)
        self.init(prop,fac,misc)
    '''
    @use_api
    def renderHdr(self, outfile=None):
        from datetime import datetime
        from isceobj.XmlUtil import xmlUtils as xml
        from isce import release_version, release_svn_revision, release_date, svn_revision
        odProp = xml.OrderedDict()
        odFact = xml.OrderedDict()
        odMisc = xml.OrderedDict()
        # hack since the length is normally not set but obtained from the file
        # size, before rendering  make sure that coord1.size is set to length
        self.coord2.coordSize = self.length
        self.renderToDictionary(self, odProp, odFact, odMisc)
        # remove key,value pair with empty value (except if value is zero)
        DU.cleanDictionary(odProp)
        DU.cleanDictionary(odFact)
        DU.cleanDictionary(odMisc)
        odProp['ISCE_VERSION'] = "Release: %s, svn-%s, %s. Current: svn-%s." % \
                             (release_version, release_svn_revision, release_date, svn_revision)
        outfile = outfile if outfile else self.getFilename() + '.xml'
        firstTag = 'imageFile'
        XD = XmlDumper()
        XD.dump(outfile, odProp, odFact, odMisc, firstTag)
        self.renderVRT()
        return None

    # This method renders an ENVI HDR file similar to the XML file.
    def renderEnviHDR(self):
        '''
        Renders a bare minimum ENVI HDR file, that can be used to directly ingest the outputs into
        a GIS package.
        '''

        typeMap = { 'BYTE'   : 1,
                    'SHORT'  : 2,
                    'INT'    : 3,
                    'LONG'   : 14,
                    'FLOAT'  : 4,
                    'DOUBLE' : 5,
                    'CFLOAT' : 6,
                    'CDOUBLE': 9 }

        orderMap = {'L' : 0,
                    'B' : 1}

        tempstring = """ENVI
description = {{Data product generated using ISCE}}
samples = {0}
lines   = {1}
bands   = {2}
header offset = 0
file type = ENVI Standard
data type = {3}
interleave = {4}
byte order = {5}
"""
        map_infostr = """coordinate system string = {{GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137, 298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]]}}
map_info = {{Geographic Lat/Lon, 1.0, 1.0, {0}, {1}, {2}, {3}, WGS-84, units=Degrees}}"""

        flag = False
        try:
            if (self.coord1.coordStart == 0.) and \
               (self.coord2.coordStart == 0.) and \
               (self.coord1.coordDelta == 1.) and \
               (self.coord2.coordDelta == 1.):
                    flag = True
        except:
            pass


        outfile = self.getFilename() + '.hdr'
        outstr = tempstring.format(self.width, self.length,
                self.bands, typeMap[self.dataType.upper()],
                self.scheme.lower(),
                orderMap[ENDIAN[self.byteOrder].upper()])

        if not flag:
            outstr += map_infostr.format(self.coord1.coordStart,
                                         self.coord2.coordStart,
                                         self.coord1.coordDelta,
                                         -self.coord2.coordDelta)

        with open(outfile, 'w') as f:
            f.write(outstr)

        return


    # This method renders and ENVI HDR file similar to the XML file.
    def renderVRT(self, outfile=None):
        '''
        Renders a bare minimum ENVI HDR file, that can be used to directly ingest the outputs into a GIS package.
        '''
        import xml.etree.ElementTree as ET

        typeMap = { 'BYTE'   : 'Byte',
                    'SHORT'  : 'Int16',
                    'CIQBYTE': 'Int16',
                    'INT'    : 'Int32',
                    'FLOAT'  : 'Float32',
                    'DOUBLE' : 'Float64',
                    'CFLOAT' : 'CFloat32',
                    'CDOUBLE': 'CFloat64'}

        sizeMap = {'BYTE' : 1,
                   'SHORT' : 2,
                   'CIQBYTE': 2,
                   'INT'   : 4,
                   'FLOAT' : 4,
                   'DOUBLE': 8,
                   'CFLOAT' : 8,
                   'CDOUBLE' : 16}

        orderMap = {'L' : 'LSB',
                    'B' : 'MSB'}


        def indentXML(elem, depth=None, last=None):
            if depth == None:
                depth = [0]
            if last == None:
                last = False
            tab = ' ' * 4
            if(len(elem)):
                depth[0] += 1
                elem.text = '\n' + (depth[0]) * tab
                lenEl = len(elem)
                lastCp = False
                for i in range(lenEl):
                    if(i == lenEl - 1):
                        lastCp = True
                    indentXML(elem[i], depth, lastCp)
                if(not last):
                    elem.tail = '\n' + (depth[0]) * tab
                else:
                    depth[0] -= 1
                    elem.tail = '\n' + (depth[0]) * tab
            else:
                if(not last):
                    elem.tail = '\n' + (depth[0]) * tab
                else:
                    depth[0] -= 1
                    elem.tail = '\n' + (depth[0]) * tab

            return


        srs = "EPSG:4326"
        flag = False
        try:
            if (self.coord1.coordStart == 0.) and \
               (self.coord2.coordStart == 0.) and \
               (self.coord1.coordDelta == 1.) and \
               (self.coord2.coordDelta == 1.):
                    flag = True
        except:
            pass



        if not outfile:
            outfile = self.getFilename() + '.vrt'

        root = ET.Element('VRTDataset')
        root.attrib['rasterXSize'] = str(self.width)
        root.attrib['rasterYSize'] = str(self.length)

        if not flag:
            print('Writing geotrans to VRT for {0}'.format(self.filename))
            ET.SubElement(root, 'SRS').text = "EPSG:4326"
            gtstr = "{0}, {1}, 0.0, {2}, 0.0, {3}".format(self.coord1.coordStart,
                    self.coord1.coordDelta,
                    self.coord2.coordStart,
                    self.coord2.coordDelta)
            ET.SubElement(root, 'GeoTransform').text = gtstr

        nbytes = sizeMap[self.dataType.upper()]

        for band in range(self.bands):
            broot = ET.Element('VRTRasterBand')
            broot.attrib['dataType'] = typeMap[self.dataType.upper()]
            broot.attrib['band'] = str(band + 1)
            broot.attrib['subClass'] = "VRTRawRasterBand"

            elem = ET.SubElement(broot, 'SourceFilename')
            elem.attrib['relativeToVRT'] = "1"
            elem.text = os.path.basename(self.getFilename())

            ET.SubElement(broot, 'ByteOrder').text = orderMap[ENDIAN[self.byteOrder].upper()]
            if self.scheme.upper() == 'BIL':
                ET.SubElement(broot, 'ImageOffset').text = str(band * self.width * nbytes)
                ET.SubElement(broot, 'PixelOffset').text = str(nbytes)
                ET.SubElement(broot, 'LineOffset').text = str(self.bands * self.width * nbytes)
            elif self.scheme.upper() == 'BIP':
                ET.SubElement(broot, 'ImageOffset').text = str(band * nbytes)
                ET.SubElement(broot, 'PixelOffset').text = str(self.bands * nbytes)
                ET.SubElement(broot, 'LineOffset').text = str(self.bands * self.width * nbytes)
            elif self.scheme.upper() == 'BSQ':
                ET.SubElement(broot, 'ImageOffset').text = str(band * self.width * self.length * nbytes)
                ET.SubElement(broot, 'PixelOffset').text = str(nbytes)
                ET.SubElement(broot, 'LineOffset').text = str(self.width * nbytes)

            root.append(broot)

        indentXML(root)
        tree = ET.ElementTree(root)
        tree.write(outfile, encoding='unicode')


        return



    # #
    # This method initialize the Image.
    # @param filename \c string the file name associated with the image.
    # @param accessmode \c string access mode of the  file.
    # @param bands \c int number of bands of the interleaving scheme.
    # @param type \c string data type used to store the data.
    # @param width \c int width of the image.
    # @param scheme \c string interleaving scheme.
    # @param caster \c string type of caster (ex. 'DoubleToFloat').
    def initImage(self, filename, accessmode, width,
                  type=None, bands=None, scheme=None, caster=None):

        self.initAccessor(filename, accessmode, width, type, bands, scheme, caster)
    # # This method gets the pointer associated to the DataAccessor.DataAccessor
    # # object created.
    # @return \c pointer pointer to the underlying DataAccessor.DataAccessor
    # # object.
    def getImagePointer(self):
        return self.getAccessor()

    # # gets the string describing the image for the user
    # #@return \c text description string describing the image in English for
    # # the user
    def getDescription(self):
        return self.description

    # # This method appends the string describing the image for the user create
    # # a list.
    # #@param doc \c text description string describing the image in English for
    # #  the user
    def addDescription(self, doc):
        if self.description == '':
            self.description = [doc]
        elif isinstance(self.description, list):
            self.description.append(doc)

    # # This method gets the length associated to the DataAccessor.DataAccessor
    # # object created.
    # # @return \c int length of the underlying DataAccessor.DataAccessor object.
    @use_api
    def getLength(self):
        if not self.coord2.coordSize:
            self.coord2.coordSize = self.getFileLength()
        return self.coord2.coordSize

    # Always call this function if  createImage() was previously invoked.
    # It deletes the pointer to the object, closes the file associated with
    # the object, frees memory.
    def finalizeImage(self):
        if not self._isFinalized:
            self.finalizeAccessor()
            self._isFinalized = True

    def setImageType(self, val):
        self.imageType = str(val)

    def setLength(self, val):
        # needed because the __init__ calls self.lenth = None which calls this
        # function and the casting would fail. with time possibly need to
        # refactor all the image API with better inheritance
        if val is not None:
            self.coord2.coordSize = int(val)

    def getWidth(self):
        return self.coord1.coordSize

    def setWidth(self, val):
        # see getLength
        if val is not None:
            width = int(val)
            self.coord1.coordSize = width
#            self.width = width
#            DataAccessor.setWidth(self, width)

    def setXmin(self, val):
        # see getLength
        if not val is None:
            xmin = val
            self.coord1.coordStart = xmin
    def getXmin(self):
        return  self.coord1.coordStart

    def setXmax(self, val):
        # see getLength
        if not val is None:
            xmax = val
            self.coord1.coordEnd = xmax

    def getXmax(self):
        return self.coord1.coordEnd

    def setByteOrder(self, byteOrder):
        try:
            b0 = ENDIAN[byteOrder]
        except KeyError:
            self.logger.error(
                self.__class__.__name__ +
                ".setByteOorder got a bad argument:" +
                str(byteOrder)
                )
            raise ValueError(str(byteOrder) +
                             " is not a valid byte ordering, e.g.\n" +
                             str(ENDIAN.keys()))
        self.byteOrder = b0
        return None

    # # Set the caster type if needed
    # @param accessMode \c string access mode of the file. Can be 'read' or 'write'
    # @param dataType \c string is the dataType from or to the caster writes or reads.
    def setCaster(self, accessMode, dataType):
        self.accessMode = accessMode
        if(accessMode == 'read'):
            self.caster = CF.getCaster(self.dataType, dataType)
        elif(accessMode == 'write'):
            self.caster = CF.getCaster(dataType, self.dataType)
        else:
            print('Unrecorgnized access mode', accessMode)
            raise ValueError
    @property
    def extraFilename(self):
        return self._extraFilename

    @extraFilename.setter
    def extraFilename(self,val):
        self._extraFilename = val

    def setFirstLatitude(self, val):
        self.coord2.coordStart = val

    def setFirstLongitude(self, val):
        self.coord1.coordStart = val

    def setDeltaLatitude(self, val):
        self.coord2.coordDelta = val

    def setDeltaLongitude(self, val):
        self.coord1.coordDelta = val

    def getFirstLatitude(self):
        return self.coord2.coordStart

    def getFirstLongitude(self):
        return self.coord1.coordStart

    def getDeltaLatitude(self):
        return self.coord2.coordDelta

    def getDeltaLongitude(self):
        return self.coord1.coordDelta
    def getImageType(self):
        return self.imageType

    def getByteOrder(self):
        return self.byteOrder

    def getProduct(self):
        return self.product

    def setProduct(self, val):
        self.product = val
    '''
    def _facilities(self):
        self.coord1 = self.facility('coord1',public_name='Coordinate1',module='isceobj.Image',factory='createCoordinate',mandatory=True,doc='First coordinate of a 2D image (witdh).')
        self.coord2 = self.facility('coord2',public_name='Coordinate2',module='isceobj.Image',factory='createCoordinate',mandatory=True,doc='Second coordinate of a 2D image (length).')
    '''


    firstLatitude = property(getFirstLatitude, setFirstLatitude)
    firstLongitude = property(getFirstLongitude, setFirstLongitude)
    deltaLatitude = property(getDeltaLatitude, setDeltaLatitude)
    deltaLongitude = property(getDeltaLongitude, setDeltaLongitude)
    width = property(getWidth, setWidth)
    length = property(getLength, setLength)
    xmin = property(getXmin, setXmin)
    xmax = property(getXmax, setXmax)
    pass


class ImageCoordinate(Configurable):
    family = 'imagecoordinate'

    def __init__(self, family='', name=''):
        # # Call super with class name
        Configurable.__init__(self, family if family else  self.__class__.family, name=name)
        self.coordDescription = ''
        self._parameters()

        return None

    @property
    def coordStart(self):
        return self._coordStart
    @coordStart.setter
    def coordStart(self, val):
        self._coordStart = val
    @property
    def coordEnd(self):
        if self._coordEnd is None and self._coordSize is not None:
            self._coordEnd = self._coordStart + self._coordSize * self._coordDelta
        return self._coordEnd
    @coordEnd.setter
    def coordEnd(self, val):
        self._coordEnd = val
    @property
    def coordSize(self):
        return self._coordSize
    @coordSize.setter
    def coordSize(self, val):
        self._coordSize = val
    @property
    def coordDelta(self):
        return self._coordDelta
    @coordDelta.setter
    def coordDelta(self, val):
        self._coordDelta = val

    def _parameters(self):
        self._coordStart = self.parameter('coordStart', public_name='startingValue', default=0, units='',
                                         type=float, mandatory=False,
                                         doc="Starting value of the coordinate.")
        self._coordEnd = self.parameter('coordEnd', public_name='endingValue', default=None, units='',
                                         type=float, mandatory=False,
                                         doc="Ending value of the coordinate.")
        self._coordDelta = self.parameter('coordDelta', public_name='delta', default=1, units='',
                                         type=float, mandatory=False,
                                         doc="Coordinate quantization.")

        self._coordSize = self.parameter('coordSize', public_name='size', default=None,
                                         type=int,
                                         mandatory=False,
                                         private=True,
                                         doc="Coordinate size.")


    pass
