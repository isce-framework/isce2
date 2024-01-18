#!/usr/bin/env python3

from types import SimpleNamespace
import json
import requests
import re
import os
import argparse
import datetime

server = 'https://scihub.copernicus.eu/gnss/'

orbitMap = [('precise', 'AUX_POEORB'),
            ('restituted', 'AUX_RESORB')]

datefmt = "%Y%m%dT%H%M%S"
queryfmt = "%Y-%m-%d"

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Fetch orbits corresponding to given SAFE package')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help='Path to SAFE package of interest')
    parser.add_argument('-o', '--output', dest='outdir', type=str, default='.',
                        help='Path to output directory')

    parser.add_argument('-u', '--username', dest='username', type=str,
                        help='Copernicus Data Space Ecosystem username')
    parser.add_argument('-p', '--password', dest='password', type=str,
                        help='Copernicus Data Space Ecosystem password')

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


def download_file(file_id, outdir='.', session=None, token=None):
    '''
    Download file to specified directory.
    '''

    if session is None:
        session = requests.session()

    url = "https://zipper.dataspace.copernicus.eu/odata/v1/"
    url += f"Products({file_id})/$value"

    path = outdir
    print('Downloading URL: ', url)
    request = session.get(url, stream=True, verify=True,
                          headers = {"Authorization": f"Bearer {token}"})

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


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    username = inps.username
    password = inps.password

    fileTS, satName, fileTSStart = FileToTimeStamp(inps.input)
    print('Reference time: ', fileTS)
    print('Satellite name: ', satName)
    match = None
    session = requests.Session()

    for spec in orbitMap:
        oType = spec[0]
        delta = datetime.timedelta(days=1)
        timebef = (fileTS - delta).strftime(queryfmt)
        timeaft = (fileTS + delta).strftime(queryfmt)
        url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

        start_time = timebef + "T00:00:00.000Z"
        stop_time = timeaft + "T23:59:59.999Z"

        query = " and ".join([
            f"ContentDate/Start gt '{start_time}'",
            f"ContentDate/Start lt '{stop_time}'",
            f"ContentDate/End gt '{start_time}'",
            f"ContentDate/End lt '{stop_time}'",
            f"startswith(Name,'{satName}')",
            f"contains(Name,'{spec[1]}')",
        ])

        success = False
        match = None

        try:
            r = session.get(url, verify=True, params={"$filter": query})
            r.raise_for_status()

            entries = json.loads(r.text, object_hook=lambda x: SimpleNamespace(**x)).value

            for entry in entries:
                entry_datefmt = "%Y-%m-%dT%H:%M:%S.000Z"
                tbef = datetime.datetime.strptime(entry.ContentDate.Start, entry_datefmt)
                taft = datetime.datetime.strptime(entry.ContentDate.End, entry_datefmt)
                if (tbef <= fileTSStart) and (taft >= fileTS):
                    matchFileName = entry.Name
                    match = entry.Id

            if match is not None:
                success = True
        except:
            raise

        if success:
            break

    if match is not None:

        if username is None:
            username = input("Username: ")
        if password is None:
            import getpass
            password = getpass.getpass("Password (will not be displayed): ")

        data = {
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        }

        url = "https://identity.dataspace.copernicus.eu/"
        url += "auth/realms/CDSE/protocol/openid-connect/token"
        r = session.post(url, data=data)
        r.raise_for_status()
        token = r.json()["access_token"]

        output = os.path.join(inps.outdir, matchFileName)
        res = download_file(match, output, session, token)
        if res is False:
            print('Failed to download orbit ID:', match)
    else:
        print('Failed to find {1} orbits for tref {0}'.format(fileTS, satName))
