#!/usr/bin/env python3

import os
import datetime
import argparse
import glob
import requests
from html.parser import HTMLParser

fmt = '%Y%m%d'
today = datetime.datetime.now().strftime(fmt)

server = 'https://scihub.copernicus.eu/gnss/'
queryfmt = '%Y-%m-%d'
datefmt = '%Y%m%dT%H%M%S'

#Generic credentials to query and download orbit files
credentials = ('gnssguest', 'gnssguest')

S1Astart = '20140901'
S1Astart_dt = datetime.datetime.strptime(S1Astart, '%Y%m%d')

S1Bstart = '20160501'
S1Bstart_dt = datetime.datetime.strptime(S1Bstart, '%Y%m%d')


def cmdLineParse():
    '''
    Automated download of orbits.
    '''
    parser = argparse.ArgumentParser('S1A and 1B AUX_POEORB precise orbit downloader')
    parser.add_argument('--start', '-b', dest='start', type=str, default=S1Astart, help='Start date')
    parser.add_argument('--end', '-e', dest='end', type=str, default=today, help='Stop date')
    parser.add_argument('--dir', '-d', dest='dirname', type=str, default='.', help='Directory with precise orbits')
    return parser.parse_args()


def fileToRange(fname):
    '''
    Derive datetime range from orbit file name.
    '''

    fields = os.path.basename(fname).split('_')
    start = datetime.datetime.strptime(fields[-2][1:16], datefmt)
    stop = datetime.datetime.strptime(fields[-1][:15], datefmt)
    mission = fields[0]

    return (start, stop, mission)


def gatherExistingOrbits(dirname):
    '''
    Gather existing orbits.
    '''

    fnames = glob.glob(os.path.join(dirname, 'S1?_OPER_AUX_POEORB*'))
    rangeList = []

    for name in fnames:
        rangeList.append(fileToRange(name))

    print(rangeList)

    return rangeList


def ifAlreadyExists(indate, mission, rangeList):
    '''
    Check if given time spanned by current list.
    '''
    found = False

    if mission == 'S1B':
        if not validS1BDate(indate):
            print('Valid: ', indate)
            return True

    for pair in rangeList:
        if (indate > pair[0]) and (indate < pair[1]) and (mission == pair[2]):
            found = True
            break

    return found


def validS1BDate(indate):
    if indate < S1Bstart_dt:
        return False
    else:
        return True


def download_file(url, outdir='.', session=None):
    '''
    Download file to specified directory.
    '''

    if session is None:
        session = requests.session()

    path = outdir
    print('Downloading URL: ', url)
    request = session.get(url, stream=True, verify=True, auth=credentials)

    try:
        request.raise_for_status()
        success = True
    except:
        success = False

    if success:
        with open(path, 'wb') as f:
            for chunk in request.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()

    return success


class MyHTMLParser(HTMLParser):

    def __init__(self,url):
        HTMLParser.__init__(self)
        self.fileList = []
        self._url = url
        
    def handle_starttag(self, tag, attrs):
        for name, val in attrs:
            if name == 'href':
                if val.startswith("https://scihub.copernicus.eu/gnss/odata") and val.endswith(")/"):
                    pass
                else:
                    downloadLink = val.strip()
                    downloadLink = downloadLink.split("/Products('Quicklook')")
                    downloadLink = downloadLink[0] + downloadLink[-1]
                    self._url = downloadLink
                
    def handle_data(self, data):
        if data.startswith("S1") and data.endswith(".EOF"):
            self.fileList.append((self._url, data.strip()))


if __name__ == '__main__':
    '''
    Main driver.
    '''

    # Parse command line
    inps = cmdLineParse()

    ###Compute interval
    tstart = datetime.datetime.strptime(inps.start, fmt)
    tend = datetime.datetime.strptime(inps.end, fmt)

    days = (tend - tstart).days
    print('Number of days to check: ', days)

    ranges = gatherExistingOrbits(inps.dirname)

    for dd in range(days):
        indate = tstart + datetime.timedelta(days=dd, hours=12)
        timebef = indate - datetime.timedelta(days=1)
        timeaft = indate + datetime.timedelta(days=1)
        timebef=str(timebef.strftime('%Y-%m-%d'))
        timeaft = str(timeaft.strftime('%Y-%m-%d'))
        url = server + 'search?q= ( beginPosition:[{0}T00:00:00.000Z TO {1}T23:59:59.999Z] AND endPosition:[{0}T00:00:00.000Z TO {1}T23:59:59.999Z] ) AND ( (platformname:Sentinel-1 AND producttype:AUX_POEORB))'.format(timebef, timeaft)
        session = requests.session()
        match = None
        success = False
        
        for selectMission in ['S1A', 'S1B']:
            if not ifAlreadyExists(indate, selectMission, ranges):
                try:
                    r = session.get(url, verify=True, auth=credentials)
                    r.raise_for_status()
                    parser = MyHTMLParser(url)
                    parser.feed(r.text)
                    
                    for resulturl, result in parser.fileList:
                        tbef, taft, mission = fileToRange(os.path.basename(result))
                        if selectMission==mission:
                            matchFileName = result
                            match = resulturl
                        

                    if match is not None:
                        success = True
                except:
                    pass

                if match is not None:
                    
                    output = os.path.join(inps.dirname, matchFileName)
                    print(output)
                    res = download_file(match, output, session)
                else:
                    print('Failed to find {1} orbits for tref {0}'.format(indate, selectMission))

            else:
                print('Already exists: ', selectMission, indate)

    print('Exit dloadOrbits Successfully')
