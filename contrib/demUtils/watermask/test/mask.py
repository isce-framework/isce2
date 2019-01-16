#!/usr/bin/env python3
from __future__ import print_function
import isce
import sys
import os
import argparse
from contrib.demUtils.WaterMask import MaskStitcher
import isceobj
def main():
    #if not argument provided force the --help flag
    if(len(sys.argv) == 1):
        sys.argv.append('-h')

    # Use the epilog to add usege eamples
    epilog = 'Usage examples:\n\n'
    epilog += 'mask.py -a stitch -i dem.xml -r -n your_username -w your_password  -u https://aria-dav.jpl.nasa.gov/repository/products \n\n'
    epilog += 'mask.py -a download -i dem.xml \n\n'
    epilog += 'mask.py -a stitch -i dem.xml -k  -r -l\n'
    #set the formatter_class=argparse.RawDescriptionHelpFormatter othewise it splits the epilog lines with its own default format
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,epilog=epilog)
    
    parser.add_argument('-a', '--action', type = str, default = 'stitch', dest = 'action', help = 'Possible actions: stitch or download (default: %(default)s). ')
    parser.add_argument('-m', '--meta', type = str, default = 'xml', dest = 'meta', help = 'What type of metadata file is created. Possible values: \
                        xml or rsc (default: %(default)s)')
    parser.add_argument('-i', '--input', type=str, required=True, dest='indem', help='Input DEM for which the land water mask is desired.')
    parser.add_argument('-k', '--keep', action = 'store_true', dest = 'keep', help = 'If the option is present then the single files used for stitching are kept. If -l or --local is specified than the flag is automatically set (default: %(default)s)')
    parser.add_argument('-r', '--report', action = 'store_true', dest = 'report', help = 'If the option is present then failed and succeeded downloads are printed (default: %(default)s)')
    parser.add_argument('-l', '--local', action = 'store_true', dest = 'local', help = 'If the option is present then use the files that are in the location \
                        specified by --dir. If not present --dir indicates the directory where the files are downloaded (default: %(default)s)')
    parser.add_argument('-d', '--dir', type = str, dest = 'dir', default = './', help = 'If used in conjunction with --local it specifies the location where the DEMs are located \
                        otherwise it specifies the directory where the DEMs are downloaded and the stitched DEM is generated (default: %(default)s)')

    parser.add_argument('-o', '--output', type = str, dest = 'output', default = None, help = 'Name of the output file to be created in --dir. If not provided the system generates one based on the bbox extremes') 
    parser.add_argument('-n', '--uname', type = str, dest = 'uname', default = None, help = 'User name if using a server that requires authentication') 
    parser.add_argument('-w', '--password', type = str, dest = 'password', default = None, help = 'Password if using a server that requires authentication') 
    parser.add_argument('-u', '--url', type = str, dest = 'url', default = None, help = 'Part of the url where the DEM files are located. The actual location must be \
                        the one specified by --url plus /srtm/version2_1/SRTM(1,3)') 


    args = parser.parse_args()
    #first get the url,uname and password since are needed in the constructor

    
    ds = MaskStitcher()
    ds.configure()
    if(args.url):
        ds.setUrl(args.url)
    ds.setUsername(args.uname)
    ds.setPassword(args.password)
    ds._keepAfterFailed = True
    #avoid to accidentally remove local file if -k is forgotten
    #if one wants can remove them manually
    if(args.local):
        args.keep = True
    if(args.meta == 'xml'):
        ds.setCreateXmlMetadata(True)
    elif(args.meta == 'rsc'):
        ds.setCreateRscMetadata(True)

    ds.setUseLocalDirectory(args.local)


    ####Parse input DEM xml to get bbox
    inimg = isceobj.createDemImage()
    inimg.load(args.indem + '.xml')

    north = inimg.coord2.coordStart
    south = north + inimg.coord2.coordDelta * (inimg.length-1)

    west = inimg.coord1.coordStart
    east = west + inimg.coord1.coordDelta * (inimg.width-1)

    bbox = [south,north,west,east]
    

    ds.setWidth(inimg.width)
    ds.setLength(inimg.length)
    ds.setFirstLatitude(north)
    ds.setFirstLongitude(west)
    ds.setLastLatitude(south)
    ds.setLastLongitude(east)

    if(args.action == 'stitch'):
        lat = bbox[0:2]
        lon = bbox[2:4]
        if (args.output is None):
            args.output = ds.defaultName(bbox)

        if not(ds.stitchMasks(lat,lon,args.output,args.dir,keep=args.keep)):
            print('Some tiles are missing. Maybe ok')
    
    elif(args.action == 'download'):
        lat = bbox[0:2]
        lon = bbox[2:4]
        ds.getMasksInBox(lat,lon,args.dir)
    
    else:
        print('Unrecognized action -a or --action',args.action)
        return

    if(args.report):
        for k,v in ds._downloadReport.items():
            print(k,'=',v)


if __name__ == '__main__':
    sys.exit(main())
