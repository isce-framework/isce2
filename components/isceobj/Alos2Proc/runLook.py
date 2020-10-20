#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from contrib.alos2proc.alos2proc import look
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar

logger = logging.getLogger('isce.alos2insar.runLook')

def runLook(self):
    '''take looks
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    #referenceTrack = self._insar.loadTrack(reference=True)
    #secondaryTrack = self._insar.loadTrack(reference=False)
    wbdFile = os.path.abspath(self._insar.wbd)

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)


    amp = isceobj.createImage()
    amp.load(self._insar.amplitude+'.xml')
    width = amp.width
    length = amp.length
    width2 = int(width / self._insar.numberRangeLooks2)
    length2 = int(length / self._insar.numberAzimuthLooks2)

    if not ((self._insar.numberRangeLooks2 == 1) and (self._insar.numberAzimuthLooks2 == 1)):
        #take looks
        look(self._insar.differentialInterferogram, self._insar.multilookDifferentialInterferogram, width, self._insar.numberRangeLooks2, self._insar.numberAzimuthLooks2, 4, 0, 1)
        look(self._insar.amplitude, self._insar.multilookAmplitude, width, self._insar.numberRangeLooks2, self._insar.numberAzimuthLooks2, 4, 1, 1)
        look(self._insar.latitude, self._insar.multilookLatitude, width, self._insar.numberRangeLooks2, self._insar.numberAzimuthLooks2, 3, 0, 1)
        look(self._insar.longitude, self._insar.multilookLongitude, width, self._insar.numberRangeLooks2, self._insar.numberAzimuthLooks2, 3, 0, 1)
        look(self._insar.height, self._insar.multilookHeight, width, self._insar.numberRangeLooks2, self._insar.numberAzimuthLooks2, 3, 0, 1)
        #creat xml
        create_xml(self._insar.multilookDifferentialInterferogram, width2, length2, 'int')
        create_xml(self._insar.multilookAmplitude, width2, length2, 'amp')
        create_xml(self._insar.multilookLatitude, width2, length2, 'double')
        create_xml(self._insar.multilookLongitude, width2, length2, 'double')
        create_xml(self._insar.multilookHeight, width2, length2, 'double')
        #los has two bands, use look program in isce instead
        #cmd = "looks.py -i {} -o {} -r {} -a {}".format(self._insar.los, self._insar.multilookLos, self._insar.numberRangeLooks2, self._insar.numberAzimuthLooks2)
        #runCmd(cmd)

        #replace the above system call with function call
        from mroipac.looks.Looks import Looks
        from isceobj.Image import createImage
        inImage = createImage()
        inImage.load(self._insar.los+'.xml')

        lkObj = Looks()
        lkObj.setDownLooks(self._insar.numberAzimuthLooks2)
        lkObj.setAcrossLooks(self._insar.numberRangeLooks2)
        lkObj.setInputImage(inImage)
        lkObj.setOutputFilename(self._insar.multilookLos)
        lkObj.looks()

        #water body
        #this looking operation has no problems where there is only water and land, but there is also possible no-data area
        #look(self._insar.wbdOut, self._insar.multilookWbdOut, width, self._insar.numberRangeLooks2, self._insar.numberAzimuthLooks2, 0, 0, 1)
        #create_xml(self._insar.multilookWbdOut, width2, length2, 'byte')
        #use waterBodyRadar instead to avoid the problems of no-data pixels in water body
        waterBodyRadar(self._insar.multilookLatitude, self._insar.multilookLongitude, wbdFile, self._insar.multilookWbdOut)


    os.chdir('../')

    catalog.printToLog(logger, "runLook")
    self._insar.procDoc.addAllFromCatalog(catalog)


