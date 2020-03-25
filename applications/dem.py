#!/usr/bin/env python3


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright  2010 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import isce
import sys
import os
import argparse
from contrib.demUtils import createDemStitcher


def main():
    #if not argument provided force the --help flag
    if(len(sys.argv) == 1):
        sys.argv.append('-h')

    # Use the epilog to add usage examples
    epilog = 'Usage examples:\n\n'
    epilog += 'Stitch (-a stitch) 1 arcsec dems (-s 1) in the bounding region 31 33 -114 -112 using the url (-u) and the log in credentials provided (-n,-w).\n'
    epilog += 'Create a rsc metadata file (-m) and report the download results (-r)\n'
    epilog += 'dem.py -a stitch -b 31 33 -114 -112 -s 1 -m rsc -r -n your_username -w your_password  -u https://aria-alt-dav.jpl.nasa.gov/repository/products/SRTM1_v3/ \n\n'
    epilog += 'Download (-a download) the 3 arcsec (-s 3) whose lat/lon are 31 -114 and 31 -115 (-p)\n'
    epilog += 'dem.py -a download -p 31 -114 31 -115 -s 3 \n\n'
    epilog += 'Stitch the requested files and apply EGM96 -> WGS84 correction (-c)\n'
    epilog += 'dem.py -a stitch -b 31 33 -114 -113 -r -s 1 -c\n\n'
    epilog += 'Download from bounding boxes (-b)\n'
    epilog += 'dem.py -a download -b 31 33 -114 -113  -r  -s 1\n\n'
    epilog += 'Stitch the files in the local directory (-l) in the bounding region provided keeping the\n'
    epilog += 'zip files after stitching (-k)\n'
    epilog += 'dem.py -a stitch -b 31 33 -114 -113 -k  -r -l  -s 1\n\n'

    #set the formatter_class=argparse.RawDescriptionHelpFormatter otherwise it splits the epilog lines with its own default format
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,epilog=epilog)

    parser.add_argument('-a', '--action', type = str, default = 'stitch', dest = 'action', help = 'Possible actions: stitch or download (default: %(default)s). ')
    parser.add_argument('-c', '--correct', action = 'store_true', dest = 'correct', help = 'Apply correction  EGM96 -> WGS84 (default: %(default)s). The output metadata is in xml format only')
    parser.add_argument('-m', '--meta', type = str, default = 'xml', dest = 'meta', help = 'What type of metadata file is created. Possible values: \
                        xml or rsc (default: %(default)s)')
    parser.add_argument('-s', '--source', type = int, default = 1, dest = 'source', help = 'Dem SRTM source. Possible values 1  or 3 (default: %(default)s)')
    parser.add_argument('-f', '--filling', action = 'store_true', dest = 'filling', help = 'Flag to instruct to fill missing Dems with null values \
                        (default null value -32768. Use -v or --filling_value option to change it)')
    parser.add_argument('-v', '--filling_value', type = int, default = -32768, dest = 'fillingValue', help = 'Value used to fill missing Dems (default: %(default)s)')
    parser.add_argument('-b', '--bbox', type = int, default = None, nargs = '+', dest = 'bbox', help = 'Defines the spatial region in the format south north west east.\
                        The values should be integers from (-90,90) for latitudes and (0,360) or (-180,180) for longitudes.')
    parser.add_argument('-p', '--pairs', type = int, default = None, nargs = '+', dest = 'pairs', help = 'Set of latitude and longitude pairs for which --action = download is performed.\
                        The values should be integers from (-90,90) for latitudes and (0,360) or (-180,180) for longitudes')
    parser.add_argument('-k', '--keep', action = 'store_true', dest = 'keep', help = 'If the option is present then the single files used for stitching are kept. If -l or --local is specified than the flag is automatically set (default: %(default)s)')
    parser.add_argument('-r', '--report', action = 'store_true', dest = 'report', help = 'If the option is present then failed and succeeded downloads are printed (default: %(default)s)')
    parser.add_argument('-l', '--local', action = 'store_true', dest = 'local', help = 'If the option is present then use the files that are in the location \
                        specified by --dir. If not present --dir indicates the directory where the files are downloaded (default: %(default)s)')
    parser.add_argument('-d', '--dir', type = str, dest = 'dir', default = './', help = 'If used in conjunction with --local it specifies the location where the DEMs are located \
                        otherwise it specifies the directory where the DEMs are downloaded and the stitched DEM is generated (default: %(default)s)')

    parser.add_argument('-o', '--output', type = str, dest = 'output', default = None, help = 'Name of the output file to be created in --dir. If not provided the system generates one based on the bbox extremes')
    parser.add_argument('-n', '--uname', type = str, dest = 'uname', default = None, help = 'User name if using a server that requires authentication')
    parser.add_argument('-w', '--password', type = str, dest = 'password', default = None, help = 'Password if using a server that requires authentication')
    parser.add_argument('-t', '--type', type = str, dest = 'type', default = 'version3', help = \
                        'Use version 3 or version 2 SRTM')
    parser.add_argument('-x', '--noextras', action = 'store_true', dest = 'noextras', help = 'Use this flag if the filenames do not have extra part')
    parser.add_argument('-u', '--url', type = str, dest = 'url', default = None, help = \
                        'If --type=version2 then this is part of the url where the DEM files are located. The actual location must be' + \
                        'the one specified by --url plus /srtm/version2_1/SRTM(1,3).'  \
                        +'If --type=version3 then it represents the full path url')
    args = parser.parse_args()
    #first get the url,uname and password since are needed in the constructor


    ds = createDemStitcher(args.type)
    ds.configure()

    if(args.url):
        if(args.type == 'version3'):
            if(args.source == 1):
                ds._url1 = args.url
            elif(args.source == 3):
                ds._url3 = args.url
            else:
                print('Unrecognized source')
                raise ValueError

        else:
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
    if(args.noextras):
        ds._hasExtras = False
    ds.setUseLocalDirectory(args.local)
    ds.setFillingValue(args.fillingValue)
    ds.setFilling() if args.filling else ds.setNoFilling()
    if(args.action == 'stitch'):
        if(args.bbox):
            lat = args.bbox[0:2]
            lon = args.bbox[2:4]
            if (args.output is None):
                args.output = ds.defaultName(args.bbox)

            if not(ds.stitchDems(lat,lon,args.source,args.output,args.dir,keep=args.keep)):
                print('Could not create a stitched DEM. Some tiles are missing')
            else:
                if(args.correct):
                    #ds.correct(args.output,args.source,width,min(lat[0],lat[1]),min(lon[0],lon[1]))
                    demImg = ds.correct()
                    # replace filename with full path including dir in which file is located
                    demImg.filename = os.path.abspath(os.path.join(args.dir, demImg.filename))
                    demImg.setAccessMode('READ')
                    demImg.renderHdr()
        else:
            print('Error. The --bbox (or -b) option must be specified when --action stitch is used')
            raise ValueError
    elif(args.action == 'download'):
        if(args.bbox):
            lat = args.bbox[0:2]
            lon = args.bbox[2:4]
            ds.getDemsInBox(lat,lon,args.source,args.dir)
        #can make the bbox and pairs mutually esclusive if replace the if below with elif
        if(args.pairs):
            ds.downloadFilesFromList(args.pairs[::2],args.pairs[1::2],args.source,args.dir)
        if(not (args.bbox or args.pairs)):
            print('Error. Either the --bbox (-b) or the --pairs (-p) options must be specified when --action download is used')
            raise ValueError

    else:
        print('Unrecognized action -a or --action',args.action)
        return

    if(args.report):
        for k,v in list(ds._downloadReport.items()):
            print(k,'=',v)


if __name__ == '__main__':
    sys.exit(main())
