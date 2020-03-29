#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2020
#


import sys
import isce
from isceobj.Alos2Proc.runDownloadDem import download_wbd


if __name__=="__main__":

    if len(sys.argv) < 5:
        print("usage: wbd_with_correction.py s n w e")
        print("where s, n, w, e are latitude, longitude bounds in degrees")
        sys.exit(0)

    snwe = list(map(float,sys.argv[1:]))

    download_wbd(snwe[0], snwe[1], snwe[2], snwe[3])
