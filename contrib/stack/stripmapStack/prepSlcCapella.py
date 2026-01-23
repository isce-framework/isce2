#!/usr/bin/env python3

import os
import glob
import argparse
import json
from uncompressFile import uncompressfile


def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Prepare Capella SLC processing (unzip/untar files, organize in date folders, generate script to unpack into ISCE formats).')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help='directory with the Capella SLC data')
    parser.add_argument('-rmfile', '--rmfile', dest='rmfile', action='store_true', default=False,
                        help='Optional: remove zip/tar/compressed files after unpacking into date structure (default is to keep in archive folder)')
    parser.add_argument('-o', '--output', dest='output', type=str, required=False,
                        help='output directory where data needs to be unpacked into ISCE format (for script generation).')
    parser.add_argument('-t', '--text_cmd', dest='text_cmd', type=str, default='source ~/.bash_profile;',
                        help='text command to be added to the beginning of each line of the run files. Default: source ~/.bash_profile;')

    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args=iargs)


def get_Date(capellaFolder):
    '''
    Extract acquisition date from Capella metadata.
    '''

    # Look for extended JSON metadata file
    jsonfiles = glob.glob(os.path.join(capellaFolder, '*_extended.json'))
    if not jsonfiles:
        jsonfiles = glob.glob(os.path.join(capellaFolder, '*.json'))

    if len(jsonfiles) > 0:
        jsonfile = jsonfiles[0]
        try:
            with open(jsonfile, 'r') as fp:
                metadata = json.load(fp)

            # Get timestamp from collect.start_timestamp
            collect = metadata.get('collect', {})
            timestamp = collect.get('start_timestamp', '')

            if timestamp:
                # Parse timestamp like "2025-10-31T19:11:04.123456Z"
                acquisitionDate = timestamp[0:4] + timestamp[5:7] + timestamp[8:10]
                if len(acquisitionDate) == 8:
                    return True, acquisitionDate
        except (json.JSONDecodeError, IOError) as e:
            print(f'Error reading {jsonfile}: {e}')

    # Could not find acquisition date
    return False, 'FAIL'


def main(iargs=None):
    '''
    The main driver.
    '''

    inps = cmdLineParse(iargs)
    # parsing required inputs
    inputDir = os.path.abspath(inps.input)
    # parsing optional inputs
    if inps.output:
        outputDir = os.path.abspath(inps.output)
    else:
        outputDir = None
    rmfile = inps.rmfile

    # filename of the runfile
    run_unPack = 'run_unPackCapella'

    # Capella file patterns - handles various archive formats
    # File naming: CAPELLA_<sat>_<mode>_SLC_<pol>_<start>_<stop>.<ext>
    capella_extensions = (
        os.path.join(inputDir, 'CAPELLA*.zip'),
        os.path.join(inputDir, 'CAPELLA*.tar'),
        os.path.join(inputDir, 'CAPELLA*.tar.gz'),
        os.path.join(inputDir, 'CAPELLA*.tgz'),
    )

    for capella_extension in capella_extensions:
        capella_filesfolders = glob.glob(capella_extension)
        for capella_infilefolder in capella_filesfolders:
            # the path to the folder/zip
            workdir = os.path.dirname(capella_infilefolder)

            # get the output name folder without any extensions
            temp = os.path.basename(capella_infilefolder)
            # trim extensions
            parts = temp.split('.')
            capella_outfolder = parts[0]
            # add the path back in
            capella_outfolder = os.path.join(workdir, capella_outfolder)

            # this is a file, try to unzip/untar it
            if os.path.isfile(capella_infilefolder):
                # unzip the file in the outfolder
                successflag_unzip = uncompressfile(capella_infilefolder, capella_outfolder)

                # put failed files in a separate directory
                if not successflag_unzip:
                    os.makedirs(os.path.join(workdir, 'FAILED_FILES'), exist_ok=True)
                    os.rename(capella_infilefolder, os.path.join(workdir, 'FAILED_FILES', '.'))
                else:
                    # check if file needs to be removed or put in archive folder
                    if rmfile:
                        os.remove(capella_infilefolder)
                        print('Deleting: ' + capella_infilefolder)
                    else:
                        os.makedirs(os.path.join(workdir, 'ARCHIVED_FILES'), exist_ok=True)
                        cmd = 'mv ' + capella_infilefolder + ' ' + os.path.join(workdir, 'ARCHIVED_FILES', '.')
                        os.system(cmd)

    # loop over the different Capella folders and organize in date folders
    # Look for folders containing Capella data (has TIFF and JSON files)
    capella_folders = glob.glob(os.path.join(inputDir, 'CAPELLA*'))
    for capella_folder in capella_folders:
        if not os.path.isdir(capella_folder):
            continue

        # Verify it has the expected Capella files
        tiffiles = glob.glob(os.path.join(capella_folder, '*.tif'))
        jsonfiles = glob.glob(os.path.join(capella_folder, '*.json'))
        if not tiffiles or not jsonfiles:
            continue

        # get the date
        successflag, imgDate = get_Date(capella_folder)

        workdir = os.path.dirname(capella_folder)
        if successflag:
            # move the folder into the date folder
            SLC_dir = os.path.join(workdir, imgDate, '')
            if os.path.isdir(SLC_dir):
                import shutil
                shutil.rmtree(SLC_dir)

            cmd = 'mv ' + capella_folder + ' ' + SLC_dir
            os.system(cmd)

            print('Success: ' + imgDate)
        else:
            print('Failed: ' + capella_folder)

    # now generate the unpacking script for all the date dirs
    dateDirs = glob.glob(os.path.join(inputDir, '2*'))
    if outputDir is not None:
        f = open(run_unPack, 'w')
        for dateDir in dateDirs:
            # Check for Capella TIFF files
            capellaFiles = glob.glob(os.path.join(dateDir, 'CAPELLA*.tif'))
            if not capellaFiles:
                capellaFiles = glob.glob(os.path.join(dateDir, '*.tif'))
            if len(capellaFiles) > 0:
                acquisitionDate = os.path.basename(dateDir)
                slcDir = os.path.join(outputDir, acquisitionDate)
                os.makedirs(slcDir, exist_ok=True)
                cmd = 'unpackFrame_Capella.py -i ' + os.path.abspath(dateDir) + ' -o ' + slcDir
                print(cmd)
                f.write(inps.text_cmd + cmd + '\n')
        f.close()


if __name__ == '__main__':

    main()
