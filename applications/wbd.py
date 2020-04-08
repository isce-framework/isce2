#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2020
#


import sys
import isce
from isceobj.Alos2Proc.runDownloadDem import download_wbd


def download_wbd_old(snwe):
    '''
    for keeping the option of the old wbd.py
    '''

    from isceobj.InsarProc.runCreateWbdMask import runCreateWbdMask

    class INSAR:
        def __init__(self):
            self.applyWaterMask = True
            self.wbdImage = None

    class SELF:
        def __init__(me, snwe):
            me.geocode_bbox = snwe
            me.insar = INSAR()

    class INFO:
        def __init__(self, snwe):
            self.extremes = snwe
        def getExtremes(x):
            return self.extremes

    self = SELF(snwe)
    info = INFO(None)
    runCreateWbdMask(self,info)


if __name__=="__main__":

    if len(sys.argv) < 5:
        print()
        print("usage: wbd.py s n w e [c]")
        print("  s: south latitude bounds in degrees")
        print("  n: north latitude bounds in degrees")
        print("  w: west longitude bounds in degrees")
        print("  e: east longitude bounds in degrees")
        print("  c: whether correct missing water body tiles problem")
        print("       0: False")
        print("       1: True (default)")
        sys.exit(0)

    doCorrection = True
    if len(sys.argv) >= 6:
    	if int(sys.argv[5]) == 0:
    		doCorrection = False
 
    snwe = list(map(float,sys.argv[1:5]))

    if doCorrection:
        download_wbd(snwe[0], snwe[1], snwe[2], snwe[3])
    else:
    	download_wbd_old(snwe)
