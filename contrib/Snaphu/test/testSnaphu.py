#!/usr/bin/env python3

import sys
from contrib.Snaphu.Snaphu import Snaphu

def main():
    inputFilename = sys.argv[1]
    outputFilename = sys.argv[2]

    snaphu = Snaphu()
    snaphu.setInput(inputFilename)
    snaphu.setOutput(outputFilename)
    snaphu.setWidth(710)
    snaphu.setCostMode('DEFO')
    snaphu.setEarthRadius(6356236.24233467)
    snaphu.setWavelength(0.0562356424)
    snaphu.setAltitude(788151.7928135)
    
    print "Preparing"
    snaphu.prepare()
    snaphu.unwrap()

if __name__ == "__main__":
    main()
