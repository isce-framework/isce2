#
# Author: Joshua Cohen
# Copyright 2016
#

import os
import isceobj
from isceobj.TopsProc.runGeocode import runGeocode

def runOffsetGeocode(self):
    '''
    Fast wrapper for topsApp's runGeocode to properly set the file list.
    '''

    print('\n=====================================')
    print('Geocoding filtered offset and LOS images...')
    print('=====================================\n')
    if self.off_geocode_list is None:
        offset = os.path.join(self._insar.mergedDirname, (self.filt_offsetfile + '.bil'))
        suffix = '.full'
        if (self.numberRangeLooks == 1) and (self.numberAzimuthLooks == 1):
            suffix = ''
        los = os.path.join(self._insar.mergedDirname, self._insar.mergedLosName+suffix+'.crop')
        self.off_geocode_list = [offset,los]

    print('File list to geocode:')
    for f in self.off_geocode_list:
        print(f)
    print()
    self.runGeocode(self.off_geocode_list, self.do_unwrap, self.geocode_bbox, is_offset_mode = True)

if __name__ == "__main__":
    '''
    Default run method for runOffsetGecode.
    '''
    main()
