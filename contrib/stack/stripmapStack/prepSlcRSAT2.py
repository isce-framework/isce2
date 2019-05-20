#!/usr/bin/env python3
# David Bekaert
import os
import glob
import argparse
from uncompressFile import uncompressfile
import shutil
import xml.etree.ElementTree as etree

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Prepare RSAT2 SLC processing (unzip/untar files, organize in date folders, generate script to unpack into isce formats). For now, it cannot merge multiple scenes')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='directory with the slc data')
    parser.add_argument('-rmfile', '--rmfile', dest='rmfile',action='store_true', default=False,
            help='Optional: remove zip/tar/compressed files after unpacking into date structure (default is to keep in archive folder)')
    parser.add_argument('-o', '--output', dest='output', type=str, required=False,
            help='output directory where data needs to be unpacked into isce format (for script generation).')
    parser.add_argument('-t', '--text_cmd', dest='text_cmd', type=str, default='source ~/.bash_profile;', 
            help='text command to be added to the beginning of each line of the run files. Default: source ~/.bash_profile;')

    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args = iargs)

def get_Date(RSAT2folder):

    # will search for different version of workreport to be compatible with ASf, WInSAR etc
    RSAT2file = glob.glob(os.path.join(RSAT2folder,'product.xml'))
    # if nothing is found return a failure
    if len(RSAT2file) > 0:
        RSAT2file = RSAT2file[0] 
        # loading the date information from the product.xml file
        tree = etree.parse(RSAT2file)
        root = tree.getroot()
        for attributes in root.iter('{http://www.rsi.ca/rs2/prod/xml/schemas}sourceAttributes'):
            attribute_list = attributes.getchildren()
        for attribute in attribute_list:
            if attribute.tag=='{http://www.rsi.ca/rs2/prod/xml/schemas}rawDataStartTime':
                date = attribute.text
                UTC = date[11:16]
                acquisitionDate = date[0:4]+date[5:7]+date[8:10]

        if len(acquisitionDate)==8:
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
    run_unPack = 'run_unPackRSAT2'


    # loop over the different folder, RSAT2 zip/tar files and unzip them, make the names consistent
    RSAT2_extensions = (os.path.join(inputDir, 'RS2*SLC*.zip'),os.path.join(inputDir, 'RS2*SLC*.tar'),os.path.join(inputDir, 'RS2*SLC*.gz'))
    for RSAT2_extension in RSAT2_extensions:
        RSAT2_filesfolders = glob.glob(RSAT2_extension)
        for RSAT2_infilefolder in RSAT2_filesfolders:
            ## the path to the folder/zip
            workdir = os.path.dirname(RSAT2_infilefolder)
            
            ## get the output name folder without any extensions
            temp = os.path.basename(RSAT2_infilefolder)
            # trim the extensions and keep only very first part
            parts = temp.split(".")
            parts = parts[0].split('-')
            RSAT2_outfolder = parts[0]
            # add the path back in
            RSAT2_outfolder = os.path.join(workdir,RSAT2_outfolder)
            
            # loop over two cases (either file or folder): 
            ### this is a file, try to unzip/untar it
            if os.path.isfile(RSAT2_infilefolder):
                # unzip the file in the outfolder
                successflag_unzip = uncompressfile(RSAT2_infilefolder,RSAT2_outfolder)

                # put failed files in a seperate directory
                if not successflag_unzip:
                    if not os.path.isdir(os.path.join(workdir,'FAILED_FILES')):
                        os.makedirs(os.path.join(workdir,'FAILED_FILES'))
                    os.rename(RSAT2_infilefolder,os.path.join(workdir,'FAILED_FILES','.'))
                else:
                    # check if file needs to be removed or put in archive folder
                    if rmfile:                                                                                                                                   
                        os.remove(RSAT2_infilefolder)
                        print('Deleting: ' + RSAT2_infilefolder)
                    else:
                        if not os.path.isdir(os.path.join(workdir,'ARCHIVED_FILES')):
                            os.makedirs(os.path.join(workdir,'ARCHIVED_FILES'))
                        cmd  = 'mv ' + RSAT2_infilefolder + ' ' + os.path.join(workdir,'ARCHIVED_FILES','.')
                        os.system(cmd)

        # loop over the different RSAT2 folders and make sure the folder names are consistent.
        # this step is not needed unless the user has manually unzipped data before.
        RSAT2_folders = glob.glob(os.path.join(inputDir, 'RS2*SLC*'))
        for RSAT2_folder in RSAT2_folders:
            # in case the user has already unzipped some files, make sure they are unzipped similar like the uncompressfile code
            temp = os.path.basename(RSAT2_folder)
            parts = temp.split(".")
            parts = parts[0].split('-')
            RSAT2_outfolder_temp = parts[0]
            RSAT2_outfolder_temp = os.path.join(os.path.dirname(RSAT2_folder),RSAT2_outfolder_temp)
            # check if the folder (RSAT2_folder) has a different filename as generated from the uncompressFile code (RSAT2_outfolder_temp)
            if not (RSAT2_outfolder_temp == RSAT2_folder):
                # it is different, check if the RSAT2_outfolder_temp already exists, if yes, delete the current folder
                if os.path.isdir(RSAT2_outfolder_temp):
                    print('Remove ' + RSAT2_folder + ' as ' + RSAT2_outfolder_temp + ' exists...')
                    # check if this folder already exist, if so overwrite it
                    shutil.rmtree(RSAT2_folder)

    # loop over the different RSAT2 folders and organize in date folders
    RSAT2_folders = glob.glob(os.path.join(inputDir, 'RS2*SLC*'))     
    for RSAT2_folder in RSAT2_folders:
        # get the date
        successflag, imgDate = get_Date(RSAT2_folder)       
            
        workdir = os.path.dirname(RSAT2_folder)
        if successflag:
            # move the file into the date folder
            SLC_dir = os.path.join(workdir,imgDate,'')
            if os.path.isdir(SLC_dir):
                shutil.rmtree(SLC_dir)

            cmd = 'mv ' + RSAT2_folder + ' ' + SLC_dir 
            os.system(cmd)

            print ('Succes: ' + imgDate)
        else:
            print('Failed: ' + RSAT2_folder)
        

    # now generate the unpacking script for all the date dirs
    dateDirs = glob.glob(os.path.join(inputDir,'2*'))
    if outputDir is not None:
        f = open(run_unPack,'w')
        for dateDir in dateDirs:
            RSAT2Files = glob.glob(os.path.join(dateDir, 'imagery_HH.tif'))
            if len(RSAT2Files)>0:
                acquisitionDate = os.path.basename(dateDir)
                slcDir = os.path.join(outputDir, acquisitionDate)
                if not os.path.exists(slcDir):
                    os.makedirs(slcDir)     
                cmd = 'unpackFrame_RSAT2.py -i ' + os.path.abspath(dateDir) + ' -o ' + slcDir      
                print (cmd)
                f.write(inps.text_cmd + cmd+'\n')
        f.close()

if __name__ == '__main__':

    main()


