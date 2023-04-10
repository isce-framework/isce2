#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2020
#


import argparse
import sys
import isce
from isceobj.Alos2Proc.runDownloadDem import download_wbd


EXAMPLE = """Usage examples:
  wbd.py -1 0 -92 -91

  # do not correct missing tiles
  wbd.py -1 0 -92 -91 0

  # use a different url to download tile files
  wbd.py -1 0 -92 -91 -u https://e4ftl01.cr.usgs.gov/DP133/SRTM/SRTMSWBD.003/2000.02.11
"""


def cmd_line_parse(iargs=None):
    parser = argparse.ArgumentParser(description='Create water body file from SRTMSWBD.003 database.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('s', type=float, help='south bounds in latitude in degrees')
    parser.add_argument('n', type=float, help='north bounds in latitude in degrees')
    parser.add_argument('w', type=float, help='west bounds in longitude in degrees')
    parser.add_argument('e', type=float, help='east bounds in longitude in degrees')
    parser.add_argument('correct_missing_tiles', type=int, nargs='?', choices=[0, 1], default=1,
                        help='whether correct missing water body tiles problem:\n'
                             '    0: False\n'
                             '    1: True (default)')
    parser.add_argument('-u', '--url', dest='url', type=str,
                        help='Change the (default) url in full path for where water body files are located.\n'
                             'E.g.: https://e4ftl01.cr.usgs.gov/DP133/SRTM/SRTMSWBD.003/2000.02.11')

    inps = parser.parse_args(args=iargs)
    return inps


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


def main(iargs=None):
    inps = cmd_line_parse(iargs)

    if inps.correct_missing_tiles:
        download_wbd(inps.s, inps.n, inps.w, inps.e, url=inps.url)
    else:
        snwe = [inps.s, inps.n, inps.w, inps.e]
        download_wbd_old(snwe)


if __name__ == '__main__':
    main(sys.argv[1:])

