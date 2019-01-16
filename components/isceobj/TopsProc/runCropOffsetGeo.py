#
# Author: Joshua Cohen
# Copyright 2016
#

import os
import isceobj
import logging
import numpy as np
from imageMath import IML

def runCropOffsetGeo(self):
    '''
    Crops and resamples lat/lon/los/z images created by topsApp to the
    same grid as the offset field image.
    '''
    print('\n====================================')
    print('Cropping topo products to offset grid...')
    print('====================================')
    
    suffix = '.full'
    if (self.numberRangeLooks == 1) and (self.numberAzimuthLooks == 1):
        suffix=''
    flist1b = ['lat.rdr'+suffix, 'lon.rdr'+suffix, 'z.rdr'+suffix]
    flist2b = [self._insar.mergedLosName+suffix]

    wend = (self.offset_width*self.skipwidth) + self.offset_left
    lend = (self.offset_length*self.skiphgt) + self.offset_top

    for filename in flist1b:
        print('\nCropping %s to %s ...\n' % (filename,filename+'.crop'))
        f = os.path.join(self._insar.mergedDirname, filename)
        outArr = []
        mmap = IML.mmapFromISCE(f,logging)
        '''
        for i in range(self.offset_top, mmap.length, self.skiphgt):
            outArr.append(mmap.bands[0][i][self.offset_left::self.skipwidth])
        '''
        for i in range(self.offset_top, lend, self.skiphgt):
            outArr.append(mmap.bands[0][i][self.offset_left:wend:self.skipwidth])

        outFile = os.path.join(self._insar.mergedDirname, filename+'.crop')
        outImg = isceobj.createImage()
        outImg.bands = 1
        outImg.scheme = 'BIP'
        outImg.dataType = 'DOUBLE'
        outImg.setWidth(len(outArr[0]))
        outImg.setLength(len(outArr))
        outImg.setFilename(outFile)
        with open(outFile,'wb') as fid:
            for i in range(len(outArr)):
                np.array(outArr[i]).astype(np.double).tofile(fid)   ### WAY easier to write to file like this
        outImg.renderHdr()
        print('Cropped %s' % (filename))

    for filename in flist2b:
        print('\nCropping %s to %s ...\n' % (filename,filename+'.crop'))
        f = os.path.join(self._insar.mergedDirname, filename)
        outArrCh1 = []
        outArrCh2 = []
        mmap = IML.mmapFromISCE(f,logging)
        '''
        for i in range(self.offset_top, mmap.length, self.skiphgt):
            outArrCh1.append(mmap.bands[0][i][self.offset_left::self.skipwidth])
            outArrCh2.append(mmap.bands[1][i][self.offset_left::self.skipwidth])
        '''
        for i in range(self.offset_top, lend, self.skiphgt):
            outArrCh1.append(mmap.bands[0][i][self.offset_left:wend:self.skipwidth])
            outArrCh2.append(mmap.bands[1][i][self.offset_left:wend:self.skipwidth])

        outFile = os.path.join(self._insar.mergedDirname, filename+'.crop')
        outImg = isceobj.createImage()
        outImg.bands = 2
        outImg.scheme = 'BIL'
        outImg.dataType = 'FLOAT'
        outImg.setWidth(len(outArrCh1[0]))
        outImg.setLength(len(outArrCh1))
        outImg.setFilename(outFile)
        with open(outFile,'wb') as fid:
            for i in range(len(outArrCh1)):
                np.array(outArrCh1[i]).astype(np.float32).tofile(fid)
                np.array(outArrCh2[i]).astype(np.float32).tofile(fid)
        outImg.renderHdr()
        print('Cropped %s' % (filename))


if __name__ == "__main__":
    '''
    Default run method for runCropOffsetGeo.
    '''
    main()
