#!/usr/bin/env python3

import numpy as np
import re
import requests
import os
import argparse
import datetime
from html.parser import HTMLParser

server = 'http://aux.sentinel1.eo.esa.int/'

orbitMap = [('precise', 'POEORB/'),
            ('restituted', 'RESORB/')]

datefmt = "%Y%m%dT%H%M%S"
queryfmt = "%Y-%m-%d"
queryfmt2 = "%Y/%m/%d/"


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Fetch orbits corresponding to given SAFE package')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help='Path to SAFE package of interest')
    parser.add_argument('-o', '--output', dest='outdir', type=str, default='.',
                        help='Path to output directory')

    return parser.parse_args()


def FileToTimeStamp(safename):
    '''
    Return timestamp from SAFE name.
    '''
    safename = os.path.basename(safename)
    fields = safename.split('_')
    sstamp = []  # sstamp for getting SAFE file start time, not needed for orbit file timestamps

    try:
        tstamp = datetime.datetime.strptime(fields[-4], datefmt)
        sstamp = datetime.datetime.strptime(fields[-5], datefmt)
    except:
        p = re.compile(r'(?<=_)\d{8}')
        dt2 = p.search(safename).group()
        tstamp = datetime.datetime.strptime(dt2, '%Y%m%d')

    satName = fields[0]

    return tstamp, satName, sstamp


class MyHTMLParser(HTMLParser):

    def __init__(self, satName, url):
        HTMLParser.__init__(self)
        self.fileList = []
        self.in_td = False
        self.in_a = False
        self.in_table = False
        self._url = url
        self.satName = satName

    def handle_starttag(self, tag, attrs):
        if tag == 'td':
            self.in_td = True
        elif tag == 'a':
            self.in_a = True
            for name, val in attrs:
                if name == "href":
                    if val.startswith("http"):
                        self._url = val.strip()

    def handle_data(self, data):
        if self.in_td and self.in_a:
            if self.satName in data:
                self.fileList.append((self._url, data.strip()))

    def handle_tag(self, tag):
        if tag == 'td':
            self.in_td = False
            self.in_a = False
        elif tag == 'a':
            self.in_a = False
            self._url = None


def download_file(url, outdir='.', session=None):
    '''
    Download file to specified directory.
    '''

    if session is None:
        session = requests.session()

    path = os.path.join(outdir, os.path.basename(url))
    print('Downloading URL: ', url)
    request = session.get(url, stream=True, verify=False)

    try:
        val = request.raise_for_status()
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


def fileToRange(fname):
    '''
    Derive datetime range from orbit file name.
    '''

    fields = os.path.basename(fname).split('_')
    start = datetime.datetime.strptime(fields[-2][1:16], datefmt)
    stop = datetime.datetime.strptime(fields[-1][:15], datefmt)
    mission = fields[0]

    return (start, stop, mission)


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()

    fileTS, satName, fileTSStart = FileToTimeStamp(inps.input)
    print('Reference time: ', fileTS)
    print('Satellite name: ', satName)

    match = None
    session = requests.Session()

    for spec in orbitMap:
        oType = spec[0]

        if oType == 'precise':
            end_date = fileTS + datetime.timedelta(days=19)
        elif oType == 'restituted':
            end_date = fileTS
        else:
            raise ValueError("Unexpected orbit type: '" + oType + "'")
        end_date2 = end_date + datetime.timedelta(days=1)
        urls = (server + spec[1] + end_date.strftime("%Y/%m/%d/") for end_date in (end_date, end_date2))

        success = False
        match = None

        try:
            for url in urls:
                r = session.get(url, verify=False)
                r.raise_for_status()
                parser = MyHTMLParser(satName, url)
                parser.feed(r.text)
                
                for resulturl, result in parser.fileList:
                    if oType == 'precise':
                        match = os.path.join(resulturl, result)
                    elif oType == 'restituted':
                        tbef, taft, mission = fileToRange(os.path.basename(result))
                        if (tbef <= fileTSStart) and (taft >= fileTS):
                            match = os.path.join(resulturl, result)

            if match is not None:
                success = True
        except:
            pass

        if success:
            break

    if match is not None:

        res = download_file(match, inps.outdir, session)
        if res is False:
            print('Failed to download URL: ', match)
    else:
        print('Failed to find {1} orbits for tref {0}'.format(fileTS, satName))
