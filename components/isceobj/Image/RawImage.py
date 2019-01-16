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
import logging
from .Image import Image
from iscesys.Component.Component import Component

##
# This class allows the creation of a RawImage object. The parameters that need to be set are
#\verbatim
#WIDTH: width of the image in units of the DATA_TYPE. Mandatory.
#FILE_NAME: name of the file containing the image. Mandatory.
#DATA_TYPE: data  type used to store the image. The naming convention is the one adopted by numpy (see LineAccessor class). Optional. Default value 'BYTE'.
#ACCESS_MODE: access mode of the file such as 'read', 'write' etc. See LineAccessor class for all possible values. Mandatory.
#NUMBER_GOOD_BYTES: number of bytes cosidered good for computation. Must be less or equal WIDTH. Optional. Default value WIDTH.
#SCHEME: the interleaving scheme adopted for the image. Could be BIL (band interleaved by line), BIP (band intereleaved by pixel) and BSQ (band sequential). Optional. BIP set by default.
#CASTER: define the type of caster. For example DoubleToFloat reads the image data as double but puts it into a buffer that is of float type. Optional. If not provided casting is not performed.
#\endverbatim
#Since the RawImage class inherits the Image.Image, the methods of initialization described in the Component package can be used.
#Moreover each parameter can be set with the corresponding accessor method setParameter() (see the class member methods).
#@see DataAccessor.Image.
#@see Component.Component.
DATA_TYPE = Component.Parameter('dataType',
                      public_name='DATA_TYPE',
                      default='byte',
                      type=str,
                      mandatory=True,
                      doc='Image data type.')
IMAGE_TYPE = Component.Parameter('imageType',
                       public_name='IMAGE_TYPE',
                       default='raw',
                       type=str,
                       mandatory=False,
                       private=True,
                       doc='Image type used for displaying.')
NUMBER_GOOD_BYTES = Component.Parameter('numberGoodBytes',
                       public_name='NUMBER_GOOD_BYTES',
                       default=0,
                       type=int,
                       mandatory=False,
                       private=True,
                       doc='Number of bytes used for the image width')
class RawImage(Image):
    
    parameter_list = (
                  DATA_TYPE,
                  IMAGE_TYPE,
                  NUMBER_GOOD_BYTES
                  ) 
    def updateParameters(self):
        self.extendParameterList(Image,RawImage)
        super(RawImage,self).updateParameters()
    
    family ='rawimage'
    def __init__(self,family = '', name = ''):

        self.updateParameters()
        super(RawImage, self).__init__(family if family else  self.__class__.family, name=name)

        self.initOptionalAndMandatoryLists()
        self.logger = logging.getLogger('isce.Image.RawImage')
        return None


    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d
    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.Image.RawImage')
        return

##
# This method creates a DataAccessor instance. The method also runs Component.InitComponent.checkIntialization().
# If the parameters tagged as mandatory are not set, an exception is thrown.
    def createImage(self):


        if self.xmin == None:
            self.xmin = 0
        if self.xmax == None:
            self.xmax = self.width
        if self.numberGoodBytes == None:
            self.logger.info('Variable NUMBER_GOOD_BYTES of the raw image %s set equal to (xmax - xmin)   = %i in RawImageBase.py' % (self.filename,self.xmax - self.xmin))
            self.numberGoodBytes = self.xmax - self.xmin

        self.checkInitialization()
        Image.createImage(self)
        return None

    def setNumberGoodBytes(self,num):
        self.numberGoodBytes = int(num)
    def getNumberGoodBytes(self):
        return self.numberGoodBytes
    pass

