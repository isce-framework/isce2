#
# Author: Joshua Cohen
# Copyright 2016
#

import numpy as np
import isce
import isceobj
import os
import logging

logger = logging.getLogger('isce.insar.OffsetFilter')

def runOffsetFilter(self):
    '''
    Filter the resulting offset field images.
    '''

    if not self.doDenseOffsets:
        return

    from scipy.ndimage.filters import median_filter

    offsetfile = os.path.join(self._insar.mergedDirname, self._insar.offsetfile) 
    snrfile = os.path.join(self._insar.mergedDirname, self._insar.snrfile)
    print('\n======================================')
    print('Filtering dense offset field image...')
    print('Offset field filename: %s\n' % (self._insar.offsetfile ))
    
    ### Open images as numpy arrays (easier to mask/filter)
    if offsetfile.endswith('.bip'):
        with open(offsetfile) as fid:
            offsetArr = np.fromfile(fid,dtype='float32').reshape(self._insar.offset_length, self._insar.offset_width * 2)
            
        downOffsets = offsetArr[:,0::2].flatten()
        acrossOffsets = offsetArr[:,1::2].flatten()

    else:
        with open(offsetfile) as fid:
            offsetArr = np.fromfile(fid,dtype='float32').reshape(2*self._insar.offset_length,self._insar.offset_width)

        downOffsets = offsetArr[0::2,:].flatten()
        acrossOffsets = offsetArr[1::2,:].flatten()

    del offsetArr   ### Save virtual space

    with open(snrfile) as fid:
        snr = np.fromfile(fid,dtype='float32')[:self._insar.offset_length*self._insar.offset_width]

    ### Filter out bad SNR elements (determined by user-configured threshold)
    if self.dense_offset_snr_thresh is not None:
        snrBad = snr < self.dense_offset_snr_thresh
        acrossOffsets[snrBad] = self.filt_null
        downOffsets[snrBad] = self.filt_null
    del snr     ### Don't need it again, save the space!

    ### Set median_filter window size (user-configurable, int-only, minimum 1)
    if self.filt_size < 1:
        print('ERROR: Filter window size must be a positive integer. Setting to default of 1.')
    window = np.max([1,np.int32(self.filt_size)])
    self.filt_size = window

    ### Reshape the offsets back into their original form (they were flattened above)
    downOffsets = downOffsets.reshape(-1,self._insar.offset_width)
    acrossOffsets = acrossOffsets.reshape(-1,self._insar.offset_width)

    ### Avoid NaNs getting "smeared" by the median_filter (only happens if user converted NULL values
    ### in offset images to be np.nan)
    if np.isnan(self.filt_null):
        nanmask = np.isnan(downOffsets)
        downOffsets[nanmask] = -9999.
        acrossOffsets[nanmask] = -9999.

    ### Filter the offsets
    downOffsets = median_filter(downOffsets,int(window))
    acrossOffsets = median_filter(acrossOffsets,int(window))

    ### If the NULL value was set by the user to np.nan, replace them as they were switched above
    if np.isnan(self.filt_null):
        nanmask = np.abs(downOffsets+9999.) < np.finfo(np.float32).eps
        downOffsets[nanmask] = np.nan
        acrossOffsets[nanmask] = np.nan

    ### Flatten the arrays to make them writeable to the .bil
    ### NOTE: can we remove this now and change it back to just writing row by row?
    #downOffsets = downOffsets.flatten()
    #acrossOffsets = acrossOffsets.flatten()

    ### Write the offsets to the .bil file
    ### Channel 1: Azimuth offsets
    ### Channel 2: Range offsets
    filt_offsetfile = os.path.join(self._insar.mergedDirname, self._insar.filt_offsetfile)
    filtImg = isceobj.createImage()
    filtImg.bands = 2
    filtImg.scheme = 'BIL'
    filtImg.dataType = 'FLOAT'
    filtImg.setWidth(self._insar.offset_width)
    filtImg.setLength(self._insar.offset_length)
    filtImg.setFilename(filt_offsetfile)
    with open(filt_offsetfile,'wb') as fid:
        for i in range(self._insar.offset_length):
            downOffsets[i].astype(np.float32).tofile(fid)
            acrossOffsets[i].astype(np.float32).tofile(fid)
            #start = i * self.offset_width
            #np.array(downOffsets[start:start+self.offset_width]).astype(np.float32).tofile(fid)
            #np.array(acrossOffsets[start:start+self.offset_width]).astype(np.float32).tofile(fid)
    filtImg.renderHdr()

if __name__ == '__main__' :
    '''
    Default routine to filter offset field images.
    '''
    
    main()
