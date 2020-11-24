import os
import logging
import numpy as np
import numpy.matlib

import isceobj

logger = logging.getLogger('isce.alos2insar.runIonCorrect')

def runIonCorrect(self):
    '''resample original ionosphere and ionospheric correction
    '''
    if hasattr(self, 'doInSAR'):
        if not self.doInSAR:
            return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    if not self.doIon:
        catalog.printToLog(logger, "runIonCorrect")
        self._insar.procDoc.addAllFromCatalog(catalog)
        return

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    from isceobj.Alos2Proc.runIonSubband import defineIonDir
    ionDir = defineIonDir()
    subbandPrefix = ['lower', 'upper']

    ionCalDir = os.path.join(ionDir['ion'], ionDir['ionCal'])
    os.makedirs(ionCalDir, exist_ok=True)
    os.chdir(ionCalDir)


    ############################################################
    # STEP 3. resample ionospheric phase
    ############################################################
    from contrib.alos2proc_f.alos2proc_f import rect
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
    from scipy.interpolate import interp1d
    import shutil

    #################################################
    #SET PARAMETERS HERE
    #interpolation method
    interpolationMethod = 1
    #################################################

    print('\ninterpolate ionosphere')

    ml2 = '_{}rlks_{}alks'.format(self._insar.numberRangeLooks1*self._insar.numberRangeLooksIon, 
                              self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooksIon)

    ml3 = '_{}rlks_{}alks'.format(self._insar.numberRangeLooks1*self._insar.numberRangeLooks2, 
                              self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooks2)

    ionfiltfile = 'filt_ion'+ml2+'.ion'
    #ionrectfile = 'filt_ion'+ml3+'.ion'
    ionrectfile = self._insar.multilookIon

    img = isceobj.createImage()
    img.load(ionfiltfile + '.xml')
    width2 = img.width
    length2 = img.length

    img = isceobj.createImage()
    img.load(os.path.join('../../', ionDir['insar'], self._insar.multilookDifferentialInterferogram) + '.xml')
    width3 = img.width
    length3 = img.length

    #number of range looks output
    nrlo = self._insar.numberRangeLooks1*self._insar.numberRangeLooks2
    #number of range looks input
    nrli = self._insar.numberRangeLooks1*self._insar.numberRangeLooksIon
    #number of azimuth looks output
    nalo = self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooks2
    #number of azimuth looks input
    nali = self._insar.numberAzimuthLooks1*self._insar.numberAzimuthLooksIon

    if (self._insar.numberRangeLooks2 != self._insar.numberRangeLooksIon) or \
       (self._insar.numberAzimuthLooks2 != self._insar.numberAzimuthLooksIon):
        #this should be faster using fortran
        if interpolationMethod == 0:
            rect(ionfiltfile, ionrectfile,
                width2,length2,
                width3,length3,
                nrlo/nrli, 0.0,
                0.0, nalo/nali,
                (nrlo-nrli)/(2.0*nrli),
                (nalo-nali)/(2.0*nali),
                'REAL','Bilinear')
        #finer, but slower method
        else:
            ionfilt = np.fromfile(ionfiltfile, dtype=np.float32).reshape(length2, width2)
            index2 = np.linspace(0, width2-1, num=width2, endpoint=True)
            index3 = np.linspace(0, width3-1, num=width3, endpoint=True) * nrlo/nrli + (nrlo-nrli)/(2.0*nrli)
            ionrect = np.zeros((length3, width3), dtype=np.float32)
            for i in range(length2):
                f = interp1d(index2, ionfilt[i,:], kind='cubic', fill_value="extrapolate")
                ionrect[i, :] = f(index3)
            
            index2 = np.linspace(0, length2-1, num=length2, endpoint=True)
            index3 = np.linspace(0, length3-1, num=length3, endpoint=True) * nalo/nali + (nalo-nali)/(2.0*nali)
            for j in range(width3):
                f = interp1d(index2, ionrect[0:length2, j], kind='cubic', fill_value="extrapolate")
                ionrect[:, j] = f(index3)
            ionrect.astype(np.float32).tofile(ionrectfile)
            del ionrect
        create_xml(ionrectfile, width3, length3, 'float')

        os.rename(ionrectfile, os.path.join('../../insar', ionrectfile))
        os.rename(ionrectfile+'.vrt', os.path.join('../../insar', ionrectfile)+'.vrt')
        os.rename(ionrectfile+'.xml', os.path.join('../../insar', ionrectfile)+'.xml')
        os.chdir('../../insar')
    else:
        shutil.copyfile(ionfiltfile, os.path.join('../../insar', ionrectfile))
        os.chdir('../../insar')
        create_xml(ionrectfile, width3, length3, 'float')
    #now we are in 'insar'


    ############################################################
    # STEP 4. correct interferogram
    ############################################################
    from isceobj.Alos2Proc.Alos2ProcPublic import renameFile
    from isceobj.Alos2Proc.Alos2ProcPublic import runCmd

    if self.applyIon:
        print('\ncorrect interferogram')
        if os.path.isfile(self._insar.multilookDifferentialInterferogramOriginal):
            print('original interferogram: {} is already here, do not rename: {}'.format(self._insar.multilookDifferentialInterferogramOriginal, self._insar.multilookDifferentialInterferogram))
        else:
            print('renaming {} to {}'.format(self._insar.multilookDifferentialInterferogram, self._insar.multilookDifferentialInterferogramOriginal))
            renameFile(self._insar.multilookDifferentialInterferogram, self._insar.multilookDifferentialInterferogramOriginal)

        cmd = "imageMath.py -e='a*exp(-1.0*J*b)' --a={} --b={} -s BIP -t cfloat -o {}".format(
            self._insar.multilookDifferentialInterferogramOriginal,
            self._insar.multilookIon,
            self._insar.multilookDifferentialInterferogram)
        runCmd(cmd)
    else:
        print('\nionospheric phase estimation finished, but correction of interfeorgram not requested')

    os.chdir('../')

    catalog.printToLog(logger, "runIonCorrect")
    self._insar.procDoc.addAllFromCatalog(catalog)

