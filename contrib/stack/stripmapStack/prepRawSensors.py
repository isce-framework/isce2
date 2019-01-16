#!/usr/bin/env python3

# Bekaert David


import os
import glob
import argparse


def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Script that attempts to recognize the sensor automatically and then call the correspodning unzips/unpacks command.')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='directory which has all the raw data.')
    parser.add_argument('-rmfile', '--rmfile', dest='rmfile',action='store_true', default=False,
            help='Optional: remove zip/tar/compressed files after unpacking into date structure (default is to keep in archive folder)')
    parser.add_argument('-o', '--output', dest='output', type=str, required=False,
            help='Optional: output directory which will be used for unpacking into isce format (run file generation only).')
    parser.add_argument('-t', '--text_cmd', dest='text_cmd', type=str, default='source ~/.bash_profile;'
            , help='Optional: text command to be added to the beginning of each line of the run files. Default: source ~/.bash_profile;')

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

    inps = cmdLineParse(iargs)
    ## parsing of the required input arguments
    inputDir = os.path.abspath(inps.input)
    ## parsing of the optional input arguments
    # output dir for generating runfile
    if inps.output:
        outputDir = os.path.abspath(inps.output)
        outputDir_str = ' -o ' + outputDir
    else:
        outputDir_str = ''
    # textcommand to be added to the runfile start
    text_str = ' -t "' + inps.text_cmd + '"'
    # delete zip file option
    if inps.rmfile: 
        rmfile_str = ' -rmfile'
    else:
        rmfile_str = ''

    # search criteria for the different sensors
    ENV_str = 'ASA*'		# Envisat
    ERS_CEOS_str = 'ER*CEOS*'	# ERS in CEOS format
    ERS_ENV_str = 'ER*ESA*'	# ERS in Envisat format
    ALOS1_str = 'ALPSRP*'	# ALOS-1 Palsar, zip files and extracted files
    CSK_str = 'EL*'		# CSK, zip files
    CSK_str2 = 'CSK*.h5'        # CSK, extracted files
    # combine together
    sensor_str_list = (ENV_str,ERS_CEOS_str,ERS_ENV_str,ALOS1_str,CSK_str,CSK_str2)
    sensor_list = ('Envisat','ERS_CEOS','ERS_ENV','ALOS1','CSK','CSK')
    sensor_unpackcommand = ('TODO','TODO','TODO','prepRawALOS.py','prepRawCSK.py','prepRawCSK.py')
    Sensors = dict(zip(sensor_str_list,sensor_list))
    Sensors_unpack = dict(zip(sensor_str_list,sensor_unpackcommand))

    # Loop over the different sensor strings and try to find them
    sensor_found = False
    for sensor_str in Sensors:
        for file in glob.iglob(os.path.join(inputDir,'**',sensor_str),recursive=True):
            sensor_found = True
            sensor_str_keep = sensor_str
            break

    # report back to user
    if sensor_found:
        print("Looks like " + Sensors[sensor_str_keep])
        cmd  = Sensors_unpack[sensor_str_keep] + ' -i ' + inputDir + rmfile_str + outputDir_str + text_str
        print(cmd)
        os.system(cmd)

    else:
        print("Did not find the sensor automatically, unzip and run unpack routines manual")


if __name__ == '__main__':

    main()


