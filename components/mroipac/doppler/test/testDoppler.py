#!/usr/bin/env python3

import sys
from mroipac.doppler.Doppler import Doppler

def main():
    rawFilename = sys.argv[1]
    obj = Doppler()
    obj.setSLCfilename(rawFilename)
    obj.setSamples(15328)
    obj.setLines(32710)
    obj.setStartLine(1)
    obj.calculateDoppler()
    acc = obj.getAcc()
    for val in acc:
        print val


if __name__ == "__main__":
    main()
