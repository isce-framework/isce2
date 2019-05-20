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

    parser = argparse.ArgumentParser(description='Prepare CSK raw processing (unzip/untar files, organize in date folders, generate script to unpack into isce formats).')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='directory with the raw data')
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

def get_Date(CSKfolder):

    # will search for different version of workreport to be compatible with ASf, WInSAR etc
    CSKfile = glob.glob(os.path.join(CSKfolder,'CSK*.h5'))
    # if nothing is found return a failure
    if len(CSKfile) > 0:
        CSKfile = os.path.basename(CSKfile[0])
        parts = CSKfile.split('_')
        if len(parts)>8:
            if len(parts[8])>8:
                acquisitionDate = parts[8]
                acquisitionDate = acquisitionDate[0:8]
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
    run_unPack = 'run_unPackCSK'


    # loop over the different folder, CSK zip/tar files and unzip them, make the names consistent
    CSK_extensions = (os.path.join(inputDir, 'EL*.zip'),os.path.join(inputDir, 'EL*.tar'),os.path.join(inputDir, 'EL*.gz'))
    for CSK_extension in CSK_extensions:
        CSK_filesfolders = glob.glob(CSK_extension)
        for CSK_infilefolder in CSK_filesfolders:
            ## the path to the folder/zip
            workdir = os.path.dirname(CSK_infilefolder)
            
            ## get the output name folder without any extensions
            temp = os.path.basename(CSK_infilefolder)
            # trim the extensions and keep only very first part
            parts = temp.split(".")
            parts = parts[0].split('-')
            CSK_outfolder = parts[0]
            # add the path back in
            CSK_outfolder = os.path.join(workdir,CSK_outfolder)
            
            # loop over two cases (either file or folder): 
            ### this is a file, try to unzip/untar it
            if os.path.isfile(CSK_infilefolder):
                # unzip the file in the outfolder
                successflag_unzip = uncompressfile(CSK_infilefolder,CSK_outfolder)

                # put failed files in a seperate directory
                if not successflag_unzip:
                    if not os.path.isdir(os.path.join(workdir,'FAILED_FILES')):
                        os.makedirs(os.path.join(workdir,'FAILED_FILES'))
                    os.rename(CSK_infilefolder,os.path.join(workdir,'FAILED_FILES','.'))
                else:
                    # check if file needs to be removed or put in archive folder
                    if rmfile:                                                                                                                                   
                        os.remove(CSK_infilefolder)
                        print('Deleting: ' + CSL_infilefolder)
                    else:
                        if not os.path.isdir(os.path.join(workdir,'ARCHIVED_FILES')):
                            os.makedirs(os.path.join(workdir,'ARCHIVED_FILES'))
                        cmd  = 'mv ' + CSK_infilefolder + ' ' + os.path.join(workdir,'ARCHIVED_FILES','.')
                        os.system(cmd)
                                                                                                                                      
        # loop over the different CSK folders and make sure the folder names are consistent.
        # this step is not needed unless the user has manually unzipped data before.
        CSK_folders = glob.glob(os.path.join(inputDir, 'EL*'))
        for CSK_folder in CSK_folders:
            # in case the user has already unzipped some files, make sure they are unzipped similar like the uncompressfile code
            temp = os.path.basename(CSK_folder)
            parts = temp.split(".")
            parts = parts[0].split('-')
            CSK_outfolder_temp = parts[0]
            CSK_outfolder_temp = os.path.join(os.path.dirname(CSK_folder),CSK_outfolder_temp)
            # check if the folder (CSK_folder) has a different filename as generated from the uncompressFile code (CSK_outfolder_temp)
            if not (CSK_outfolder_temp == CSK_folder):
                # it is different, check if the CSK_outfolder_temp already exists, if yes, delete the current folder
                if os.path.isdir(CSK_outfolder_temp):
                    print('Remove ' + CSK_folder + ' as ' + CSK_outfolder_temp + ' exists...')
                    # check if this folder already exist, if so overwrite it
                    shutil.rmtree(CSK_folder)

    # loop over the different CSK folders and organize in date folders
    CSK_folders = glob.glob(os.path.join(inputDir, 'EL*'))     
    for CSK_folder in CSK_folders:
        # get the date
        successflag, imgDate = get_Date(CSK_folder)       
            
        workdir = os.path.dirname(CSK_folder)
        if successflag:
            # move the file into the date folder
            SLC_dir = os.path.join(workdir,imgDate,'')
            if not os.path.isdir(SLC_dir):
                os.makedirs(SLC_dir)
                
            # check if the folder already exist in that case overwrite it
            CSK_folder_out = os.path.join(SLC_dir,os.path.basename(CSK_folder))
            if os.path.isdir(CSK_folder_out):
                shutil.rmtree(CSK_folder_out)

            ### FOR NOW TO MAKE MERGING WORK OF MULTIPLE SCENES
            ### In future would be better to have a -m option for CSK unpack like ALOS unpack?
            cmd  = 'mv ' + CSK_folder + '/* ' + SLC_dir + '.' 
            os.system(cmd)
            cmd = 'rmdir ' + CSK_folder 
            os.system(cmd)
            ###
            ###

            # # move the CSK acqusition folder in the date folder
            # cmd  = 'mv ' + CSK_folder + ' ' + SLC_dir + '.' 
            # os.system(cmd)




            print ('Succes: ' + imgDate)
        else:
            print('Failed: ' + CSK_folder)
        

    # now generate the unpacking script for all the date dirs
    dateDirs = glob.glob(os.path.join(inputDir,'2*'))
    if outputDir is not None:
        f = open(run_unPack,'w')
        for dateDir in dateDirs:
            CSKFiles = glob.glob(os.path.join(dateDir, 'CSK*.h5'))
            if len(CSKFiles)>0:
                acquisitionDate = os.path.basename(dateDir)
                slcDir = os.path.join(outputDir, acquisitionDate)
                if not os.path.exists(slcDir):
                    os.makedirs(slcDir)     
                cmd = 'unpackFrame_CSK_raw.py -i ' + os.path.abspath(dateDir) + ' -o ' + slcDir      
                print (cmd)
                f.write(inps.text_cmd + cmd+'\n')
    
            """ 
            ##### FOR now lets ptu all scences in single folder
            CSKFiles = glob.glob(os.path.join(dateDir, 'EL*'))
            if len(CSKFiles)>0:
               acquisitionDate = os.path.basename(dateDir)
               slcDir = os.path.join(outputDir, acquisitionDate)
               if not os.path.exists(slcDir):
                  os.makedirs(slcDir)     
               cmd = 'unpackFrame_CSK_raw.py -i ' + os.path.abspath(dateDir) + ' -o ' + slcDir      

               if len(CSKFiles) > 1:
                  cmd = cmd + ' -m' 
               print (cmd)
               f.write(inps.text_cmd + cmd+'\n')
            """
        f.close()

if __name__ == '__main__':

    main()


