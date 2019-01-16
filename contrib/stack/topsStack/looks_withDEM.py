#!/usr/bin/env python3
import isce
import argparse
import os
import looks

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Take integer number of looks.',
            epilog = '''

Example: 

looks.py -i input.file -o output.file -r 4  -a 4
            
''')
    parser.add_argument('-i','--input', type=str, required=True, help='Input ISCEproduct with a corresponding .xml file.', dest='infile')
    parser.add_argument('-o','--output',type=str, default=None, help='Output ISCE DEproduct with a corresponding .xml file.', dest='outfile')
    parser.add_argument('-r', '--range', type=int, default=1, help='Number of range looks. Default: 1', dest='rglooks')
    parser.add_argument('-a', '--azimuth', type=int, default=1, help='Number of azimuth looks. Default: 1', dest='azlooks')

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):

    inps = cmdLineParse(iargs)

    if (inps.rglooks == 1) and (inps.azlooks == 1):
        print('Nothing to do. One look requested in each direction. Exiting ...')
        sys.exit(0)

    looks.main(inps)

if __name__ == '__main__':

    main()

