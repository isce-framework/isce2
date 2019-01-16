#!/usr/bin/env python3
# David Bekaert
import os
import glob
import argparse
from uncompressFile import uncompressfile
import shutil

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Prepare ALOS raw processing (unzip/untar files, organize in date folders, generate script to unpack into isce formats).')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='directory with the raw data')
    parser.add_argument('-rmfile', '--rmfile', dest='rmfile',action='store_true', default=False,
            help='Optional: remove zip/tar/compressed files after unpacking into date structure (default is to keep in archive fo  lder)')
    parser.add_argument('-o', '--output', dest='output', type=str, required=False,
            help='output directory where data needs to be unpacked into isce format (for script generation).')
    parser.add_argument('-t', '--text_cmd', dest='text_cmd', type=str, default='source ~/.bash_profile;'
       , help='text command to be added to the beginning of each line of the run files. Default: source ~/.bash_profile;')
    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args = iargs)

def get_Date(ALOSfolder):

    # will search for different version of workreport to be compatible with ASf, WInSAR etc
    workreport_files = ('*workreport','summary.txt')
    for workreport_file in workreport_files:
        workreports = glob.glob(os.path.join(ALOSfolder,workreport_file))

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
    run_unPack = 'run_unPackALOS'   

    # loop over the different folder, ALOS zip/tar files and unzip them, make the names consistent
    ALOS_extensions = (os.path.join(inputDir, 'ALP*.zip'),os.path.join(inputDir, 'ALP*.tar'),os.path.join(inputDir, 'ALP*.gz'))
    for ALOS_extension in ALOS_extensions:
        ALOS_filesfolders = glob.glob(ALOS_extension)
        for ALOS_infilefolder in ALOS_filesfolders:
            ## the path to the folder/zip
            workdir = os.path.dirname(ALOS_infilefolder)
            
            ## get the output name folder without any extensions
            temp = os.path.basename(ALOS_infilefolder)
            # trim the extensions and keep only very first part
            parts = temp.split(".")
            parts = parts[0].split('-')
            ALOS_outfolder = parts[0]
            # add the path back in
            ALOS_outfolder = os.path.join(workdir,ALOS_outfolder)
            
            # loop over two cases (either file or folder): 
            ### this is a file, try to unzip/untar it
            if os.path.isfile(ALOS_infilefolder):
                # unzip the file in the outfolder
                successflag_unzip = uncompressfile(ALOS_infilefolder,ALOS_outfolder)

                # put failed files in a seperate directory
                if not successflag_unzip:
                    if not os.path.isdir(os.path.join(workdir,'FAILED_FILES')):
                        os.makedirs(os.path.join(workdir,'FAILED_FILES'))
                    os.rename(ALOS_infilefolder,os.path.join(workdir,'FAILED_FILES','.'))
                else:
                    # check if file needs to be removed or put in archive folder
                    if rmfile:
                        os.remove(ALOS_infilefolder)
                        print('Deleting: ' + ALOS_infilefolder)
                    else:
                        if not os.path.isdir(os.path.join(workdir,'ARCHIVED_FILES')):
                            os.makedirs(os.path.join(workdir,'ARCHIVED_FILES'))
                        cmd  = 'mv ' + ALOS_infilefolder + ' ' + os.path.join(workdir,'ARCHIVED_FILES','.')
                        os.system(cmd)


        # loop over the different ALOS folders and make sure the folder names are consistent.
        # this step is not needed unless the user has manually unzipped data before.
        ALOS_folders = glob.glob(os.path.join(inputDir, 'ALP*'))
        for ALOS_folder in ALOS_folders:
            # in case the user has already unzipped some files, make sure they are unzipped similar like the uncompressfile code
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
    ALOS_folders = glob.glob(os.path.join(inputDir, 'ALP*'))                        
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
    dateDirs = glob.glob(os.path.join(inputDir,'2*'))
    if outputDir is not None:
        f = open(run_unPack,'w')
        for dataDir in dateDirs:
            AlosFiles = glob.glob(os.path.join(dataDir, 'ALP*'))
            if len(AlosFiles)>0:
                acquisitionDate = os.path.basename(dataDir)
                slcDir = os.path.join(outputDir, acquisitionDate)
                if not os.path.exists(slcDir):
                    os.makedirs(slcDir)     
                cmd = 'unpackFrame_ALOS_raw.py -i ' + os.path.abspath(dataDir) + ' -o ' + slcDir      
                IMG_files = glob.glob(os.path.join(AlosFiles[0],'IMG*'))
                if len(IMG_files)==1:
                    cmd = cmd + ' -f  fbs2fbd ' 
                if len(AlosFiles) > 1:
                    cmd = cmd + ' -m' 
                print (cmd)
                f.write(inps.text_cmd + cmd+'\n')
        f.close()

if __name__ == '__main__':

    main()


