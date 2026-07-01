#!/usr/bin/env python3

import os
import glob
import argparse
import json


def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Prepare Capella SLC processing: organize TIFF files into date folders and generate unpack script.')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help='directory with the Capella SLC TIFF files')
    parser.add_argument('-o', '--output', dest='output', type=str, required=False,
                        help='output directory where SLCs will be unpacked into ISCE format')
    parser.add_argument('-t', '--text_cmd', dest='text_cmd', type=str, default='',
                        help='text command to be added to the beginning of each line of the run files')

    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args=iargs)


def get_date_from_tiff(tiffFile):
    '''
    Extract acquisition date from Capella TIFF ImageDescription tag.
    '''
    try:
        from osgeo import gdal
    except ImportError:
        raise Exception('GDAL python bindings not found. Need this for Capella data.')

    ds = gdal.Open(tiffFile, gdal.GA_ReadOnly)
    if ds is None:
        return False, 'FAIL'

    # Get ImageDescription from TIFF metadata
    image_desc = ds.GetMetadataItem('TIFFTAG_IMAGEDESCRIPTION')
    ds = None

    if not image_desc:
        return False, 'FAIL'

    try:
        metadata = json.loads(image_desc)
        collect = metadata.get('collect', {})
        timestamp = collect.get('start_timestamp', '')

        if timestamp:
            # Parse timestamp like "2025-10-31T19:11:04.123456Z"
            acquisitionDate = timestamp[0:4] + timestamp[5:7] + timestamp[8:10]
            if len(acquisitionDate) == 8:
                return True, acquisitionDate
    except (json.JSONDecodeError, KeyError) as e:
        print(f'Error reading metadata from {tiffFile}: {e}')

    return False, 'FAIL'


def main(iargs=None):
    '''
    The main driver.
    '''

    inps = cmdLineParse(iargs)
    inputDir = os.path.abspath(inps.input)

    if inps.output:
        outputDir = os.path.abspath(inps.output)
    else:
        outputDir = None

    # filename of the runfile
    run_unPack = 'run_unPackCapella'

    # Find all Capella TIFF files
    # File naming: CAPELLA_<sat>_<mode>_SLC_<pol>_<start>_<stop>.tif
    tiffFiles = glob.glob(os.path.join(inputDir, 'CAPELLA*.tif'))
    if not tiffFiles:
        tiffFiles = glob.glob(os.path.join(inputDir, '*.tif'))

    # Organize TIFF files into date folders
    for tiffFile in tiffFiles:
        successflag, imgDate = get_date_from_tiff(tiffFile)

        if successflag:
            # Create date folder
            dateDir = os.path.join(inputDir, imgDate)
            os.makedirs(dateDir, exist_ok=True)

            # Move TIFF file to date folder
            destFile = os.path.join(dateDir, os.path.basename(tiffFile))
            if tiffFile != destFile:
                os.rename(tiffFile, destFile)
                print(f'Moved {os.path.basename(tiffFile)} -> {imgDate}/')
        else:
            print(f'Failed to get date from: {tiffFile}')

    # Generate the unpacking script for all date dirs
    dateDirs = sorted(glob.glob(os.path.join(inputDir, '2*')))

    if outputDir is not None:
        with open(run_unPack, 'w') as f:
            for dateDir in dateDirs:
                if not os.path.isdir(dateDir):
                    continue

                # Find Capella TIFF file in this date directory
                tiffFiles = glob.glob(os.path.join(dateDir, 'CAPELLA*.tif'))
                if not tiffFiles:
                    tiffFiles = glob.glob(os.path.join(dateDir, '*.tif'))

                if tiffFiles:
                    tiffFile = tiffFiles[0]
                    acquisitionDate = os.path.basename(dateDir)
                    slcDir = os.path.join(outputDir, acquisitionDate)
                    os.makedirs(slcDir, exist_ok=True)

                    cmd = f'unpackFrame_Capella.py -i {tiffFile} -o {slcDir}'
                    print(cmd)
                    if inps.text_cmd:
                        f.write(f'{inps.text_cmd} {cmd}\n')
                    else:
                        f.write(f'{cmd}\n')

        print(f'\nGenerated run file: {run_unPack}')


if __name__ == '__main__':

    main()
