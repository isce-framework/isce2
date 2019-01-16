#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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



import logging
import isce
from contrib.demUtils.SWBDStitcher import SWBDStitcher
from iscesys.DataManager import createManager
import os
import math
logger = logging.getLogger('isce.insar.createWbdMask')

def runCreateWbdMask(self, info):
    if self.insar.applyWaterMask:
        sw = createManager('wbd')
        sw.configure()
        ####If the user has requested a bounding box
        if self.geocode_bbox:
            latMax = math.ceil(self.geocode_bbox[1])
            latMin = math.floor(self.geocode_bbox[0])
            lonMin = math.floor(self.geocode_bbox[2])
            lonMax = math.ceil(self.geocode_bbox[3])
        else:
            extremes = info.getExtremes(.2)
            latMax = extremes[1]
            latMin = extremes[0]
            lonMax = extremes[3]
            lonMin = extremes[2]

        #get the name of the swbd image
        name = sw.defaultName([latMin,latMax,lonMin,lonMax])
        #form the name of the corresponding xml file
        nameXml = name + '.xml'

        #Check if the swbd file exists on disk to load from
        #either in the local directory
        if os.path.exists(nameXml) and os.path.exists(name):
            from isceobj import createImage
            image  = createImage()
            image.load(nameXml)
            image.metadatalocation = nameXml

        #or in the DEMDB directory
        elif ( "DEMDB" in os.environ and
               os.path.isfile(os.path.join(os.environ["DEMDB"],
                                           nameXml))
        ):
            from isceobj import createImage
            image  = createImage()
            image.load(os.path.join(os.environ["DEMDB"],nameXml))
            image.metadatalocation = os.path.join(os.environ["DEMDB"],nameXml)


        #or finally, have the stitcher download and stitch a new one.
        else:
            sw.noFilling = False
            sw.stitch([latMin,latMax],[lonMin,lonMax])
            image = sw.image

            #if there is a global store, move the swbd files to it
            if "DEMDB" in os.environ and os.path.exists(os.environ["DEMDB"]):
                #modify the filename in the meta data to include
                #path to the global store
                from isceobj import createImage
                image = createImage()
                image.load(nameXml)
                image.filename = os.path.join(os.environ["DEMDB"],
                    image.filename)
                image._extraFilename = os.path.join(os.environ["DEMDB"],
                    image._extraFilename)
                image.metadatalocation = os.path.join(os.environ["DEMDB"],nameXml)
                image.dump(nameXml)

                #remove the swbdLat*.vrt file from the local directory because
                #a side effect of the demImage.dump() above was to create the
                #vrt in the location indicated by the path in the xml file.
                os.remove(nameXml.replace('.xml','.vrt'))

                #make list of swbdLat file names to be moved to the global store
                import glob
                dwlist = glob.glob(name+"*")
                import shutil
                #move the dem files to the global store
                for dwfile in dwlist:
                    shutil.move(dwfile, os.environ["DEMDB"])

        #put the wbdImage in the InsarProc object
        self.insar.wbdImage = image
