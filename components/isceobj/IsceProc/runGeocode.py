#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Kosal Khun, Marco Lavalle
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Comment: Adapted from InsarProc/runGeocode.py
import logging
import stdproc
from stdproc.rectify.geocode.Geocodable import Geocodable
import isceobj

#from contextlib import nested
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from iscesys.StdOEL.StdOELPy import create_writer

import os
logger = logging.getLogger('isce.isceProc.runGeocode')
posIndx = 1


def runGeocode(self, prodlist, unwrapflag, bbox):
    '''Generalized geocoding of all the files listed above (in prodlist).'''
    if isinstance(prodlist, str):
        from isceobj.Util.StringUtils import StringUtils as SU
        tobeGeocoded = SU.listify(prodlist)
    else:
        tobeGeocoded = prodlist

    #####Remove the unwrapped interferogram if no unwrapping is done
    if not unwrapflag:
        try:
            tobeGeocoded.remove(self._isce.unwrappedIntFilename)
        except ValueError:
            pass

    print('Number of products to geocode: ', len(tobeGeocoded))

    stdWriter = create_writer("log", "", True, filename="geo.log")

    velocity, height = self._isce.vh()

    if bbox is not None:
        snwe = list(bbox)
        if len(snwe) != 4:
            raise valueError('Bounding box should be a list/tuple of length 4')
    else:
        snwe = self._isce.topo.snwe

    infos = {}
    for attribute in ['demCropFilename', 'numberRangeLooks', 'numberAzimuthLooks',
                      'is_mocomp', 'demImage', 'peg', 'dopplerCentroid']:
        infos[attribute] = getattr(self._isce, attribute)


    for sceneid1, sceneid2 in self._isce.selectedPairs:
        pair = (sceneid1, sceneid2)
        for pol in self._isce.selectedPols:
            frame1 = self._isce.frames[sceneid1][pol]
            formSLC1 = self._isce.formSLCs[sceneid1][pol]
            sid = self._isce.formatname(pair, pol)            
            infos['outputPath'] = os.path.join(self.getoutputdir(sceneid1, sceneid2), sid)
            catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
            run(tobeGeocoded, frame1, formSLC1, velocity, height, snwe, infos, catalog=catalog, sceneid=sid)
            self._isce.procDoc.addAllFromCatalog(catalog)

def run(tobeGeocoded, frame1, formSLC1, velocity, height, snwe, infos, catalog=None, sceneid='NO_ID'):
    logger.info("Geocoding Image: %s" % sceneid)
    stdWriter = create_writer("log", "", True, filename=infos['outputPath'] + ".geo.log")

    planet = frame1.getInstrument().getPlatform().getPlanet()
    doppler = infos['dopplerCentroid'].getDopplerCoefficients(inHz=False)[0]

    #####Geocode one by one
    for prod in tobeGeocoded:
        prodPath = infos['outputPath'] + '.' + prod
        if not os.path.isfile(prodPath):
            logger.info("File not found. Skipping %s" % prodPath) #KK some prods are only in refScene folder! (tbd)
            continue
        #else:
        objGeo = stdproc.createGeocode('insarapp_geocode_' + os.path.basename(prod).replace('.',''))
        objGeo.configure()
        objGeo.referenceOrbit = formSLC1.getMocompPosition(posIndx)

    ####IF statements to check for user configuration
        if objGeo.minimumLatitude is None:
            objGeo.minimumLatitude = snwe[0]

        if objGeo.maximumLatitude is None:
            objGeo.maximumLatitude = snwe[1]

        if objGeo.minimumLongitude is None:
            objGeo.minimumLongitude = snwe[2]

        if objGeo.maximumLongitude is None:
            objGeo.maximumLongitude = snwe[3]

        if objGeo.demCropFilename is None:
            objGeo.demCropFilename = infos['outputPath'] + '.' + infos['demCropFilename']

        if objGeo.dopplerCentroidConstantTerm is None:
            objGeo.dopplerCentroidConstantTerm = doppler

        if objGeo.bodyFixedVelocity is None:
            objGeo.bodyFixedVelocity = velocity

        if objGeo.spacecraftHeight is None:
            objGeo.spacecraftHeight = height

        if objGeo.numberRangeLooks is None:
            objGeo.numberRangeLooks = infos['numberRangeLooks']

        if objGeo.numberAzimuthLooks is None:
            objGeo.numberAzimuthLooks = infos['numberAzimuthLooks']

        if objGeo.isMocomp is None:
            objGeo.isMocomp = infos['is_mocomp']

        objGeo.stdWriter = stdWriter

        #create the instance of the image and return the method is supposed to use
        ge = Geocodable()
        inImage, objGeo.method = ge.create(prodPath)
        if objGeo.method is None:
            objGeo.method = method

        if inImage:
            demImage = isceobj.createDemImage()
            IU.copyAttributes(infos['demImage'], demImage)
            objGeo(peg=infos['peg'], frame=frame1,
                           planet=planet, dem=demImage, tobegeocoded=inImage,
                           geoPosting=None, referenceslc=formSLC1)

            if catalog is not None:
                isceobj.Catalog.recordInputsAndOutputs(catalog, objGeo,
                                                       "runGeocode.%s.%s" % (sceneid, prodPath),
                                                       logger,
                                                       "runGeocode.%s.%s" % (sceneid, prodPath))

    stdWriter.finalize()
