#!/usr/bin/env python3

import sys
from mroipac.dopiq.DopIQ import DopIQ

def main():
    rawFilename = sys.argv[1]
    obj = DopIQ()
    obj.setRawfilename(rawFilename)
    obj.setPRF(1679.87845453499)
    obj.setMean(15.5)
    obj.setLineLength(11812)
    obj.setLineHeaderLength(412)
    obj.setLineSuffixLength(11812)
    obj.setNumberOfLines(28550)
    obj.setStartLine(1)
    obj.calculateDoppler()
    acc = obj.getAcc()
    for val in acc:
        print val


if __name__ == "__main__":
    main()
