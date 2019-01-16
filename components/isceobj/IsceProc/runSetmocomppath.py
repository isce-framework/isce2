#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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



# Comment: Adapted from InsarProc/runSetMocomppath.py
import logging
import stdproc
import isceobj

logger = logging.getLogger('isce.isceProc.runSetmocomppath')


def runSetmocomppath(self, peg=None):
    getpegs = {}
    stdWriter = self._stdWriter
    for sceneid in self._isce.selectedScenes:
        getpegs[sceneid] = {}
        self._isce.pegAverageHeights[sceneid] = {}
        self._isce.pegProcVelocities[sceneid] = {}
        for pol in self._isce.selectedPols:
            frame = self._isce.frames[sceneid][pol]
            planet = frame.getInstrument().getPlatform().getPlanet()
            orbit = self._isce.orbits[sceneid][pol]
            catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
            sid = self._isce.formatname(sceneid, pol)
            objGetpeg = run(orbit, planet, stdWriter, peg=peg, catalog=catalog, sceneid=sid)
            self._isce.pegAverageHeights[sceneid][pol] = objGetpeg.getAverageHeight()
            self._isce.pegProcVelocities[sceneid][pol] = objGetpeg.getProcVelocity()
            self._isce.procDoc.addAllFromCatalog(catalog)
            getpegs[sceneid][pol] = objGetpeg
    objpegpts = []
    for pol in self._isce.selectedPols:
        objpegpts.extend(self._isce.getAllFromPol(pol, getpegs))

    catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
    peg = averageObjPeg(objpegpts, planet, catalog=catalog, sceneid='ALL') ##planet is the last one from the loop
    self._isce.procDoc.addAllFromCatalog(catalog)
    self._isce.peg = peg


def run(orbit, planet, stdWriter, peg=None, catalog=None, sceneid='NO_ID'):
    """
    Get peg point from orbit.
    """
    logger.info("Selecting individual peg points: %s" % sceneid)

    objGetpeg = stdproc.createGetpeg()
    if peg is not None:
        objGetpeg.setPeg(peg)

    objGetpeg.wireInputPort(name='planet', object=planet)
    objGetpeg.wireInputPort(name='Orbit', object=orbit)
    objGetpeg.stdWriter = stdWriter.set_file_tags("getpeg",
                                                 "log",
                                                 "err",
                                                 "log")
    logger.info('Peg points are computed for individual SAR scenes.')
    objGetpeg.estimatePeg()
    if catalog is not None:
        isceobj.Catalog.recordInputsAndOutputs(catalog, objGetpeg,
                                               "runSetmocomppath.Getpeg.%s" % sceneid,
                                               logger,
                                               "runSetmocomppath.Getpeg.%s" % sceneid)
    return objGetpeg


def averageObjPeg(objpegpts, planet, catalog=None, sceneid='NO_POL'):
    """
    Average peg points.
    """
    logger.info('Combining individual peg points: %s' % sceneid)
    peg = stdproc.orbit.pegManipulator.averagePeg([gp.getPeg() for gp in objpegpts], planet)
    pegheights = [gp.getAverageHeight() for gp in objpegpts]
    pegvelocities = [gp.getProcVelocity() for gp in objpegpts]
    peg.averageheight = float(sum(pegheights)) / len(pegheights)
    peg.averagevelocity = float(sum(pegvelocities)) / len(pegvelocities)
    if catalog is not None:
        isceobj.Catalog.recordInputsAndOutputs(catalog, peg,
                                               "runSetmocomppath.averagePeg.%s" % sceneid,
                                               logger,
                                               "runSetmocomppath.averagePeg.%s" % sceneid)
    return peg
