#!/usr/bin/env python3

# David Bekaert

import zipfile
import os
import glob
import argparse
import tarfile
import shutil

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Script to uncompress tar and zip files.')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='File to be uncompressed')
    parser.add_argument('-o', '--output', dest='output', type=str, required=False,
            help='Directory to where the file needs to be uncompressed to (default is input name without extension).')
    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args = iargs)


def main(iargs=None):
    '''
    The main driver.
    '''

    # getting the input file and the output dir
    inps = cmdLineParse(iargs)
#    inputFile = inps.input
#    if inps.output:
#        outputDir = inps.output
#    else:
#        outputDir = None

    completeFlag = uncompressfile(inps.input,inps.output)

    if completeFlag == True:
        print('Done')
    elif completeFlag == False:
        print('Failed')

def uncompressfile(inputFile,outputDir):
    
    # keeping track of succesfull unzipping/untarring
    completeFlag = False

    # check if the file exists
    if not os.path.isfile(inputFile):
        print('File not found: ' + inputFile)
        completeFlag = None
        return completeFlag

    # defining the filenames
    if not outputDir:
        # strip the extension(s) of the name. avoid .tar to remain for tar.gz
        parts = inputFile.split(".")
        outputDir = parts[0]

    # make sure the path is absolute
    outputDir= os.path.abspath(outputDir)
    inputFile = os.path.abspath(inputFile)
    workdir = os.path.dirname(outputDir)

    # raize an exception if the input and outputdir names are the same
    if inputFile == outputDir:
        print('Input file and extraction directory are the same, abord...')
        return completeFlag


    # make the output directory if it does not exist 
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)


    ## loop over the different options, and if fail try the second one
    # see if the file has a .zip extension
    temp, extension = os.path.splitext(inputFile)    

    # File update
    print('File: ', inputFile, ' to ', outputDir)                                                                                                            
    if extension == '.zip':
        ZIP = zipfile.ZipFile(inputFile)

        # first test if the zip is in good condition
        test = ZIP.testzip()
        if test is not None:
            print('Zip file seems to be corrupted, abord...')
            return completeFlag
        else:
            ZIP.extractall(outputDir)
            ZIP.close()
            completeFlag = True

            # Check if the data is unpacked in its own folder
            folderfiles = glob.glob(os.path.join(outputDir,'*'))
            if len(folderfiles)==1:
                # get the sub-folder name only
                tempdir = os.path.basename(folderfiles[0])
                if os.path.isdir(folderfiles[0]):
                    # it seems there is a subfolder, will copy the content in the parent
                    tempdir2=os.path.join(workdir,tempdir + '.temp')
                    os.rename(folderfiles[0],tempdir2)
                    os.rmdir(outputDir)
                    os.rename(tempdir2,outputDir)
            return completeFlag
    elif extension == '.tar' or extension == '.gz':
        TAR = tarfile.open(inputFile)
            
        # first test the tar is in good condition
        try:
            TAR.extractall(outputDir)
            TAR.close()
            completeFlag = True

            # Check if the data is unpacked in its own folder
            folderfiles = glob.glob(os.path.join(outputDir,'*'))
            if len(folderfiles)==1:
                # get the sub-folder name only                   
                tempdir = os.path.basename(folderfiles[0])
                if os.path.isdir(folderfiles[0]):
                    # it seems there is a subfolder, will copy the content in the parent
                    tempdir2=os.path.join(workdir,tempdir + '.temp')                                                                                
                    os.rename(folderfiles[0],tempdir2)
                    os.rmdir(outputDir)
                    os.rename(tempdir2,outputDir)
            return completeFlag
        except:
            print('Tar file seems to be corrupted, abord...')
            return completeFlag
    else:
        print('Do not recognize as zip/tar file, abord...')
        return completeFlag

if __name__ == '__main__':

    main()


