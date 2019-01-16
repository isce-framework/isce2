#!/usr/bin/env python3

import numpy as np
import re
import requests
import os
import argparse
import datetime
from html.parser import HTMLParser

server = 'https://qc.sentinel1.eo.esa.int/'
server2 = 'http://aux.sentinel1.eo.esa.int/'

orbitMap = [('precise','aux_poeorb'),
            ('restituted','aux_resorb')]

datefmt = "%Y%m%dT%H%M%S"
queryfmt = "%Y-%m-%d"
queryfmt2= "%Y/%m/%d/"

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

    try:
        tstamp = datetime.datetime.strptime(fields[-4], datefmt)
    except: 
        p = re.compile(r'(?<=_)\d{8}')  
        dt2 = p.search(safename).group() 
        tstamp = datetime.datetime.strptime(dt2, '%Y%m%d')

    satName = fields[0]

    return tstamp, satName


class MyHTMLParser(HTMLParser):

    def __init__(self, satName):
        HTMLParser.__init__(self)
        self.fileList = []
        self.pages = 0
        self.in_td = False
        self.in_a = False
        self.in_ul = False
        self.satName = satName
    def handle_starttag(self, tag, attrs):
        if tag == 'td':
            self.in_td = True
        elif tag == 'a' and self.in_td:
            self.in_a = True
        elif tag == 'ul':
            for k,v in attrs:
                if k == 'class' and v.startswith('pagination'):
                    self.in_ul = True
        elif tag == 'li' and self.in_ul:
            self.pages += 1

    def handle_data(self,data):
        if self.in_td and self.in_a:
            #if 'S1A_OPER' in data:
            #if 'S1B_OPER' in data:
            if satName in data:
                self.fileList.append(data.strip())

    def handle_endtag(self, tag):
        if tag == 'td':
            self.in_td = False
            self.in_a = False
        elif tag == 'a' and self.in_td:
            self.in_a = False
        elif tag == 'ul' and self.in_ul:
            self.in_ul = False
        elif tag == 'html':
            if self.pages == 0:
                self.pages = 1
            else:
                # decrement page back and page forward list items
                self.pages -= 2


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
        with open(path,'wb') as f:
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

    fileTS, satName = FileToTimeStamp(inps.input)
    print('Reference time: ', fileTS)
    print('Satellite name: ', satName)
    
    match = None
    session = requests.Session()
    
    for spec in orbitMap:
        oType = spec[0]

        if oType == 'precise':
            delta =datetime.timedelta(days=2) 
        elif oType == 'restituted':
            delta = datetime.timedelta(days=1)

        timebef = (fileTS - delta).strftime(queryfmt)
        timeaft = (fileTS + delta).strftime(queryfmt)

        url = server + spec[1]
        
        query = (url + 
                '/?validity_start={0}..{1}&sentinel1__mission={2}'
                .format(timebef, timeaft,satName))

        print(query)
        success = False
        match = None
        try:
            print('Querying for {0} orbits'.format(oType))
            r = session.get(query, verify=False)
            r.raise_for_status()
            parser = MyHTMLParser(satName)
            parser.feed(r.text)
            print("Found {} pages".format(parser.pages))

            # get results from first page
            results = parser.fileList

            # page through and get more results
            for page in range(2, parser.pages + 1):
                page_parser = MyHTMLParser(satName)
                page_query = "{}&page={}".format(query, page)
                print(page_query)
                r = session.get(page_query, verify=False)
                r.raise_for_status()

                page_parser.feed(r.text)
                results.extend(page_parser.fileList)

            # run through all results and pull the orbit files
            if results:
                for result in results:
                    tbef, taft, mission = fileToRange(os.path.basename(result))

                    if (tbef <= fileTS) and (taft >= fileTS):
                        datestr2 = FileToTimeStamp(result)[0].strftime(queryfmt2) 
                        match = (server2 + spec[1].replace('aux_', '').upper() +
                                 '/' +datestr2+ result + '.EOF')
                        break

            if match is not None:
                success = True
        except Exception as e:
            print('Exception - something went wrong with the web scraper:')
            print('Exception: {}'.format(e))
            print('Continuing process')
            pass

        if match is None:
            print('Failed to find {0} orbits for Time {1}'.format(oType, fileTS))

        if success:
            break

    if match:
        res = download_file(match, inps.outdir, session=session)

        if res is False:
            print('Failed to download URL: ', match)

    session.close()

