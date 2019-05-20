#!/usr/bin/env python3
# Author: David Bekaert
# Zhang Yunjun, adopted from prepRawALOS.py for ALOS2 SM SLC


import os
import glob
import argparse
import shutil
import tarfile
import zipfile
from uncompressFile import uncompressfile


def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Prepare ALOS2 slc for processing (unzip/untar files, '
                                     'organize in date folders, generate script to unpack into isce formats).')
    parser.add_argument('-i', '--input', dest='inputDir', type=str, required=True,
            help='directory with the downloaded SLC data')
    parser.add_argument('-rmfile', '--rmfile', dest='rmfile',action='store_true', default=False,
            help='Optional: remove zip/tar/compressed files after unpacking into date structure '
                 '(default is to keep in archive fo  lder)')
    parser.add_argument('-o', '--output', dest='outputDir', type=str, required=False,
            help='output directory where data needs to be unpacked into isce format (for script generation).')
    parser.add_argument('-t', '--text_cmd', dest='text_cmd', type=str, default='source ~/.bash_profile;',
            help='text command to be added to the beginning of each line of the run files. Default: source ~/.bash_profile;')
    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args = iargs)

    # parsing required inputs
    inps.inputDir = os.path.abspath(inps.inputDir)

    # parsing optional inputs
    if inps.outputDir:
        inps.outputDir = os.path.abspath(inps.outputDir)
    return inps


def get_Date(ALOS_folder):
    """Grab acquisition date"""
    # will search for different version of workreport to be compatible with ASf, WInSAR etc
    workreport_files = ('*workreport','summary.txt')
    for workreport_file in workreport_files:
        workreports = glob.glob(os.path.join(ALOS_folder,workreport_file))

        # if nothing is found return a failure
        if len(workreports) > 0:
            for workreport in workreports:
                template_dict = {}
                with open(workreport) as openfile:
                    for line in openfile:
                        c = line.split("=")
                        template_dict[c[0].strip()] = c[1].strip()
                acquisitionDate = (str(template_dict['Img_SceneCenterDateTime'][1:9]))
                if acquisitionDate:
                    successflag = True
                    return successflag, acquisitionDate

    # if it reached here it could not find the acqusiitionDate
    successflag = False                                                                                                                                                    
    acquisitionDate = 'FAIL'
    return successflag, acquisitionDate


def get_ALOS2_name(infile):
    """Get the ALOS2210402970 name from compress file in various format."""
    outname = None
    fbase = os.path.basename(infile)
    if 'ALOS2' in fbase:
        fbase = fbase.replace('_','-')
        outname = [i for i in fbase.split('-') if 'ALOS2' in i][0]
    else:
        fext = os.path.splitext(infile)[1]
        if fext in ['.tar', '.gz']:
            with tarfile.open(infile, 'r') as tar:
                file_list = tar.getnames()
        elif fext in ['.zip']:
            with zipfile.ZipFile(infile, 'r') as z:
                file_list = z.namelist()
        else:
            raise ValueError('unrecognized file extension: {}'.format(fext))
        led_file = [i for i in file_list if 'LED' in i][0]
        led_file = os.path.basename(led_file)
        outname = [i for i in led_file.split('-') if 'ALOS2' in i][0]
    return outname


def main(iargs=None):
    '''
    The main driver.
    '''

    inps = cmdLineParse(iargs)

    # filename of the runfile
    run_unPack = 'run_unPackALOS2'

    # loop over the different folder of ALOS2 zip/tar files and unzip them, make the names consistent
    file_exts = (os.path.join(inps.inputDir, '*.zip'),
                 os.path.join(inps.inputDir, '*.tar'),
                 os.path.join(inps.inputDir, '*.gz'))
    for file_ext in file_exts:
        # loop over zip/tar files
        for fname in sorted(glob.glob(file_ext)):
            ## the path to the folder/zip
            workdir = os.path.dirname(fname)

            ## get the output name folder without any extensions
            dir_unzip = get_ALOS2_name(fname)
            dir_unzip = os.path.join(workdir, dir_unzip)

            # loop over two cases (either file or folder): 
            # if this is a file, try to unzip/untar it
            if os.path.isfile(fname):
                # unzip the file in the outfolder
                successflag_unzip = uncompressfile(fname, dir_unzip)

                # put failed files in a seperate directory
                if not successflag_unzip:
                    dir_failed = os.path.join(workdir,'FAILED_FILES')
                    if not os.path.isdir(dir_failed):
                        os.makedirs(dir_failed)
                    cmd = 'mv {} {}'.format(fname, dir_failed)
                    os.system(cmd)
                else:
                    # check if file needs to be removed or put in archive folder
                    if inps.rmfile:
                        os.remove(fname)
                        print('Deleting: ' + fname)
                    else:
                        dir_archive = os.path.join(workdir,'ARCHIVED_FILES')
                        if not os.path.isdir(dir_archive):
                            os.makedirs(dir_archive)
                        cmd = 'mv {} {}'.format(fname, dir_archive)
                        os.system(cmd)


        # loop over the different ALOS folders and make sure the folder names are consistent.
        # this step is not needed unless the user has manually unzipped data before.
        ALOS_folders = glob.glob(os.path.join(inps.inputDir, 'ALOS2*'))
        for ALOS_folder in ALOS_folders:
            # in case the user has already unzipped some files
            # make sure they are unzipped similar like the uncompressfile code
            temp = os.path.basename(ALOS_folder)
            parts = temp.split(".")
            parts = parts[0].split('-')
            ALOS_outfolder_temp = parts[0]
            ALOS_outfolder_temp = os.path.join(os.path.dirname(ALOS_folder),ALOS_outfolder_temp)
            # check if the folder (ALOS_folder) has a different filename as generated from uncompressFile (ALOS_outfolder_temp)
            if not (ALOS_outfolder_temp == ALOS_folder):
                # it is different, check if the ALOS_outfolder_temp already exists, if yes, delete the current folder
                if os.path.isdir(ALOS_outfolder_temp):
                    print('Remove ' + ALOS_folder + ' as ' + ALOS_outfolder_temp + ' exists...')
                    # check if this folder already exist, if so overwrite it
                    shutil.rmtree(ALOS_folder)


    # loop over the different ALOS folders and organize in date folders
    ALOS_folders = glob.glob(os.path.join(inps.inputDir, 'ALOS2*'))                        
    for ALOS_folder in ALOS_folders:
        # get the date
        successflag, imgDate = get_Date(ALOS_folder)       

        workdir = os.path.dirname(ALOS_folder)
        if successflag:
            # move the file into the date folder
            SLC_dir = os.path.join(workdir,imgDate,'')
            if not os.path.isdir(SLC_dir):
                os.makedirs(SLC_dir)
                
            # check if the folder already exist in that case overwrite it
            ALOS_folder_out = os.path.join(SLC_dir,os.path.basename(ALOS_folder))
            if os.path.isdir(ALOS_folder_out):
                shutil.rmtree(ALOS_folder_out)
            # move the ALOS acqusition folder in the date folder
            cmd  = 'mv ' + ALOS_folder + ' ' + SLC_dir + '.' 
            os.system(cmd)

            print ('Succes: ' + imgDate)
        else:
            print('Failed: ' + ALOS_folder)
        

    # now generate the unpacking script for all the date dirs
    dateDirs = sorted(glob.glob(os.path.join(inps.inputDir,'2*')))
    if inps.outputDir is not None:
        f = open(run_unPack,'w')
        for dateDir in dateDirs:
            AlosFiles = glob.glob(os.path.join(dateDir, 'ALOS2*'))
            # if there is at least one frame
            if len(AlosFiles)>0:
                acquisitionDate = os.path.basename(dateDir)
                slcDir = os.path.join(inps.outputDir, acquisitionDate)
                if not os.path.exists(slcDir):
                    os.makedirs(slcDir)     
                cmd = 'unpackFrame_ALOS2.py -i ' + os.path.abspath(dateDir) + ' -o ' + slcDir      
                print (cmd)
                f.write(inps.text_cmd + cmd+'\n')
        f.close()
    return


if __name__ == '__main__':

    main()
