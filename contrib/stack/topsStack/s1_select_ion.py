#!/usr/bin/env python3

#Cunren Liang, 22-MAR-2018

#this command can run multiple times for a stack
#example command
#../../code/s1_select_ion.py -dir . -sn 34/38.5 -nr 5

import os
import sys
import glob
import shutil
import zipfile
import argparse
import datetime
import numpy as np
import xml.etree.ElementTree as ET


class sentinelSLC(object):
    """
        A Class representing the SLCs
    """
    def __init__(self, safe_file, orbit_file=None):
        self.safe_file = safe_file
        self.orbit_file = orbit_file

    def get_datetime(self):
        datefmt = "%Y%m%dT%H%M%S"
        safe = os.path.basename(self.safe_file)
        fields = safe.split('_')
        self.platform = fields[0]
        self.start_date_time = datetime.datetime.strptime(fields[5], datefmt)
        self.stop_date_time = datetime.datetime.strptime(fields[6], datefmt)
        self.date = (datetime.datetime.date(self.start_date_time)).isoformat().replace('-','')

    def get_param(self):
        #speed of light (m/s)
        c = 299792458.0

        #1. processing software version
        zf = zipfile.ZipFile(self.safe_file, 'r')
        manifest = [item for item in zf.namelist() if '.SAFE/manifest.safe' in item][0]
        xmlstr = zf.read(manifest)
        root = ET.fromstring(xmlstr)
        elem = root.find('.//metadataObject[@ID="processing"]')

        #setup namespace
        nsp = "{http://www.esa.int/safe/sentinel-1.0}"
        rdict = elem.find('.//xmlData/' + nsp + 'processing/' + nsp + 'facility').attrib
        self.proc_site = rdict['site'] +', '+ rdict['country']

        rdict = elem.find('.//xmlData/' + nsp + 'processing/' + nsp + 'facility/' + nsp + 'software').attrib
        #ver = rdict['name'] + ' ' + rdict['version']
        self.proc_version = rdict['version']


        #2. start ranges
        anna = sorted([item for item in zf.namelist() if '.SAFE/annotation/s1' in item])
        #dual polarization. for the same swath, the slant ranges of two polarizations should be the same.
        if len(anna) == 6:
            anna = anna[1:6:2]

        startingRange = []
        for k in range(3):
            xmlstr = zf.read(anna[k])
            root = ET.fromstring(xmlstr)
            startingRange.append(
                float(root.find('imageAnnotation/imageInformation/slantRangeTime').text)*c/2.0
                )

        self.startingRanges = startingRange


        #3. snwe
        map_overlay = [item for item in zf.namelist() if '.SAFE/preview/map-overlay.kml' in item][0]
        xmlstr = zf.read(map_overlay)
        xmlstr = xmlstr.decode('utf-8')
        start = '<coordinates>'
        end = '</coordinates>'
        pnts = xmlstr[xmlstr.find(start)+len(start):xmlstr.find(end)].split()

        lats=[]
        lons=[]
        for pnt in pnts:
           lons.append(float(pnt.split(',')[0]))
           lats.append(float(pnt.split(',')[1]))
        self.snwe=[min(lats),max(lats),min(lons),max(lons)]


def get_safe_from_group(group):

    safes = []
    ngroup = len(group)
    for i in range(ngroup):
        ngroupi = len(group[i])
        for j in range(ngroupi):
            safes.append(group[i][j].safe_file)

    return safes


def print_group(group):
    '''print group parameters
    '''
    print()
    print('slice                                                                    no   ver         IW1 (m)           IW2 (m)           IW3 (m)')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for i in range(len(group)):
        for j in range(len(group[i])):
            print_stuff = '%s %3d  %s  '%(os.path.basename(group[i][j].safe_file), i+1, group[i][j].proc_version)
            print_stuff += "{} {} {}".format(group[i][j].startingRanges[0], group[i][j].startingRanges[1], group[i][j].startingRanges[2])
            print(print_stuff)
        print()


def get_group(dir0):
    '''
    this routine group the slices, each group is an acquisition
    the returned result is a list (acquistions sorted by starting time) containing a number of lists (slices sorted by starting time)
    '''
    #sort by starting time
    zips = sorted(glob.glob(os.path.join(dir0, 'S1*_IW_SLC_*.zip')), key=lambda x: x.split('_')[-5], reverse=False)
    nzips = len(zips)

    group = []
    for i in range(nzips):
        safeObj=sentinelSLC(zips[i])
        safeObj.get_datetime()
        safeObj.get_param()

        datefmt = "%Y%m%dT%H%M%S"
        fields = zips[i].split('_')
        tbef = datetime.datetime.strptime(fields[-5], datefmt)
        taft = datetime.datetime.strptime(fields[-4], datefmt)
        
        if i == 0:
            #create new group
            tbef0 = tbef
            group0 = []

        #S-1A is capable of operating up to 25 min per orbit [21]
        #Yague-Martinez et al., "Interferometric Processing of Sentinel-1 TOPS Data,"
        #S1A/B revisit time is 6 days, here we use 1 day to check if from the same orbit
        if np.absolute((tbef - tbef0).total_seconds()) < 24 * 3600:
            group0.append(safeObj)
        else:
            group.append(group0)
            #create new group
            tbef0 = tbef
            group0 = []
            group0.append(safeObj)

        if i == nzips - 1:
            group.append(group0)

    return group


def check_redundancy(group, threshold=1):
    '''
    threshold: time difference threshold in seconds between two slices in second.
    this routine checks, for a slice, if there are multiple ones.
    '''

    print('\nchecking different copies of same slice')

    safe_removed_indices = []
    ngroup = len(group)
    for i in range(ngroup):
        ngroupi = len(group[i])
        if ngroupi == 1:
            continue
        else:
            for j in range(ngroupi-1):
                for k in range(j+1, ngroupi):
                    if np.absolute((group[i][j].start_date_time - group[i][k].start_date_time).total_seconds()) < threshold and \
                       np.absolute((group[i][j].stop_date_time - group[i][k].stop_date_time).total_seconds()) < threshold:

                        #determine which one to be removed
                        j_version = False
                        k_version = False
                        for l in range(ngroupi):
                            if l != j and l != k:
                                if group[i][j].proc_version == group[i][l].proc_version:
                                    j_version = True
                                if group[i][k].proc_version == group[i][l].proc_version:
                                    k_version = True
                        if j_version == k_version:
                            safe_removed_index = [i, j]
                        else:
                            if j_version == False and k_version == True:
                                safe_removed_index = [i, j]
                            else:
                                safe_removed_index = [i, k]
                        safe_removed_indices.append(safe_removed_index)

                        print('in acquistion {}, the following two slices are the same:'.format(i+1))
                        if safe_removed_index == [i, j]:
                            print(os.path.basename(group[i][j].safe_file) + ' (not to be used)')
                            print(os.path.basename(group[i][k].safe_file))
                        else:
                            print(os.path.basename(group[i][j].safe_file))
                            print(os.path.basename(group[i][k].safe_file) + ' (not to be used)')

    #remove redundant slices
    if safe_removed_indices != []:
        group_new = []
        for i in range(ngroup):
            ngroupi = len(group[i])
            group_new.append([group[i][j] for j in range(ngroupi) if [i,j] not in safe_removed_indices])

        print('slices removed:')
        for index in safe_removed_indices:
            print('%s %3d'%(os.path.basename(group[index[0]][index[1]].safe_file), index[0]+1))

    else:
        group_new = group
        print('no slices removed')

    return group_new


def check_version(group):
    print('\nchecking different slice versions of an acquisition')

    acquistion_removed_indices = [] 
    ngroup = len(group)
    for i in range(ngroup):
        ngroupi = len(group[i])
        for j in range(ngroupi):
            if group[i][0].proc_version != group[i][j].proc_version:
                print('different slice versions found in acquistion {}'.format(i+1))
                acquistion_removed_indices.append(i)
                break

    #remove acquistions
    if acquistion_removed_indices != []:
        group_new = [group[i] for i in range(ngroup) if i not in acquistion_removed_indices]
        print('acquistions removed:')
        for i in acquistion_removed_indices:
            for j in range(len(group[i])):
                print('%s %3d'%(os.path.basename(group[i][j].safe_file), i+1))
    else:
        group_new = group
        print('no acquistions removed')

    return group_new


def check_gap(group):
    print('\nchecking gaps in an acquistion')

    acquistion_removed_indices = []
    ngroup = len(group)
    for i in range(ngroup):
        ngroupi = len(group[i])
        if ngroupi == 1:
            continue
        else:
            for j in range(0, ngroupi-1):
                if (group[i][j+1].start_date_time - group[i][j].stop_date_time).total_seconds() > 0:
                    acquistion_removed_indices.append(i)
                    break

    #remove acquistions
    if acquistion_removed_indices != []:
        group_new = [group[i] for i in range(ngroup) if i not in acquistion_removed_indices]
        print('acquistions removed:')
        for i in acquistion_removed_indices:
            for j in range(len(group[i])):
                print('%s %3d'%(os.path.basename(group[i][j].safe_file), i+1))
    else:
        group_new = group
        print('no acquistions removed')

    return group_new


def acquistion_snwe(groupi):
    '''return snwe of an acquisition consisting a number of slices'''
    s = min([x.snwe[0] for x in groupi])
    n = max([x.snwe[1] for x in groupi])
    w = min([x.snwe[2] for x in groupi])
    e = max([x.snwe[3] for x in groupi])

    return [s, n, w, e]


def overlap(group):
    '''return snwe of the overlap of all acquistions'''

    s = max([(acquistion_snwe(x))[0] for x in group])
    n = min([(acquistion_snwe(x))[1] for x in group])
    w = max([(acquistion_snwe(x))[2] for x in group])
    e = min([(acquistion_snwe(x))[3] for x in group])

    if s >= n or w >= e:
       #raise Exception('no overlap among the acquistions')
       print('WARNING: there is no overlap among the acquistions, snwe: {}'.format([s, n, w, e]))

    return [s, n, w, e]


def check_aoi(group, s, n):
    '''
    check each group to see if it fully covers [s, n], if not remove the acquistion
    s: south bound
    n: north bound
    '''

    print('\nchecking if each acquistion fully covers user specifed south/north bound [{}, {}]'.format(s, n))

    acquistion_removed_indices = []
    ngroup = len(group)
    for i in range(ngroup):
        snwe = acquistion_snwe(group[i])
        if not (snwe[0] <= s and snwe[1] >= n):
            acquistion_removed_indices.append(i)

    #remove acquistions
    if acquistion_removed_indices != []:
        group_new = [group[i] for i in range(ngroup) if i not in acquistion_removed_indices]
        print('acquistions removed:')
        for i in acquistion_removed_indices:
            for j in range(len(group[i])):
                print('%s %3d'%(os.path.basename(group[i][j].safe_file), i+1))
    else:
        group_new = group
        print('no acquistions removed')

    return group_new


def check_different_starting_ranges(group):
    '''
    checking if there are different starting ranges in each acquistion.
    '''

    print('\nchecking if there are different starting ranges in each acquistion')

    acquistion_removed_indices = []
    ngroup = len(group)
    for i in range(ngroup):
        ngroupi = len(group[i])
        for j in range(1, ngroupi):
            if group[i][0].startingRanges != group[i][j].startingRanges:
                acquistion_removed_indices.append(i)
                #print('++++++++++++++{} {}'.format(group[i][0].safe_file, group[i][j].safe_file))
                break

    #remove acquistions
    if acquistion_removed_indices != []:
        group_new = [group[i] for i in range(ngroup) if i not in acquistion_removed_indices]
        print('acquistions removed:')
        for i in acquistion_removed_indices:
            for j in range(len(group[i])):
                print('%s %3d'%(os.path.basename(group[i][j].safe_file), i+1))
    else:
        group_new = group
        print('no acquistions removed')

    return group_new


def check_small_number_of_acquisitions_with_same_starting_ranges(group, threshold=1):
    '''
    for the same subswath starting ranges,
    if the number of acquistions < threshold, remove these acquistions 
    '''

    print('\nchecking small-number of acquistions with same starting ranges')

    ngroup = len(group)

    starting_ranges = [x[0].startingRanges for x in group]
    
    #get unique starting_ranges
    starting_ranges_unique = []
    for i in range(ngroup):
        if starting_ranges[i] not in starting_ranges_unique:
            starting_ranges_unique.append(starting_ranges[i])

    #get number of acquistions for each unique starting ranges
    ngroup_unique = len(starting_ranges_unique)
    starting_ranges_unique_number = [0 for i in range(ngroup_unique)]
    for k in range(ngroup_unique):
        for i in range(ngroup):
            if starting_ranges_unique[k] == starting_ranges[i]:
                starting_ranges_unique_number[k] += 1

    #get starting ranges to be removed (number of acquistions < threshold)
    starting_ranges_removed = []
    for k in range(ngroup_unique):
        if starting_ranges_unique_number[k] < threshold:
            starting_ranges_removed.append(starting_ranges_unique[k])

    #remove acquistions
    if starting_ranges_removed != []:
        group_new = [group[i] for i in range(ngroup) if group[i][0].startingRanges not in starting_ranges_removed]

        print('acquistions removed:')
        for i in range(ngroup):
            if group[i][0].startingRanges in starting_ranges_removed:
                for j in range(len(group[i])):
                    print('%s %3d'%(os.path.basename(group[i][j].safe_file), i+1))

    else:
        group_new = group
        print('no acquistions removed')

    return group_new



def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='select Sentinel-1A/B acquistions good for ionosphere correction. not used slices are moved to folder: not_used')
    parser.add_argument('-dir', dest='dir', type=str, required=True,
            help = 'directory containing the "S1*_IW_SLC_*.zip" files')
    parser.add_argument('-sn', dest='sn', type=str, required=True,
            help='south/north bound of area of interest, format: south/north')
    parser.add_argument('-nr', dest='nr', type=int, default=10,
            help = 'minimum number of acquisitions for same starting ranges. default: 10')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()
    s,n=[float(x) for x in inps.sn.split('/')]

    #group the slices
    group = get_group(inps.dir)
    safes_all = get_safe_from_group(group)

    #print overlap of group
    #print('overlap among acquisitions: {}'.format(overlap(group)))

    #print group before removing slices/acquistions
    print_group(group)

    #do checks and remove the slices/acquisitions
    group = check_redundancy(group, threshold=1)
    group = check_version(group)
    group = check_gap(group)
    group = check_aoi(group, s, n)
    group = check_different_starting_ranges(group)
    group = check_small_number_of_acquisitions_with_same_starting_ranges(group, threshold=inps.nr)

    #print group after removing slices/acquistions
    print_group(group)

    #move slices that are not used to 'not_used'
    safes_used = get_safe_from_group(group)
    not_used_dir = os.path.join(inps.dir, 'not_used')
    os.makedirs(not_used_dir, exist_ok=True)
    for safe in safes_all:
        if safe not in safes_used:
            shutil.move(safe, not_used_dir)





