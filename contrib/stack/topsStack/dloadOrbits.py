#!/usr/bin/env python3

import os
import datetime
import argparse
import glob
import requests
from types import SimpleNamespace
import json
import time

fmt = '%Y%m%d'
today = datetime.datetime.now().strftime(fmt)

queryfmt = '%Y-%m-%d'
datefmt = '%Y%m%dT%H%M%S'

S1Astart = '20140901'
S1Astart_dt = datetime.datetime.strptime(S1Astart, '%Y%m%d')

S1Bstart = '20160501'
S1Bstart_dt = datetime.datetime.strptime(S1Bstart, '%Y%m%d')

satName = ['S1A', 'S1B']


def cmdLineParse():
    '''
    Automated download of orbits.
    '''
    parser = argparse.ArgumentParser('S1A and 1B AUX_POEORB precise orbit downloader')
    parser.add_argument('--start', '-b', dest='start', type=str, default=S1Astart, 
                        help='Start date')
    parser.add_argument('--end', '-e', dest='end', type=str, default=today, 
                        help='Stop date')
    parser.add_argument('--dir', '-d', dest='dirname', type=str, default='.', 
                        help='Directory with precise orbits')
    parser.add_argument('-t', '--token-file', dest='token_file', type=str, default='.copernicus_dataspace_token',
                        help='Filename to save auth token file')
    parser.add_argument('-u', '--username', dest='username', type=str, default=None,
                        help='Copernicus Data Space Ecosystem username')
    parser.add_argument('-p', '--password', dest='password', type=str, default=None,
                        help='Copernicus Data Space Ecosystem password')
    
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


    return rangeList


    
def get_saved_token_data(token_file):
    try:
        with open(token_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return None

def save_token_data(access_token, expires_in, token_file):
    token_data = {
        "access_token": access_token,
        "expires_at": time.time() + expires_in
    }
    with open(token_file, 'w') as file:
        json.dump(token_data, file)


def is_token_valid(token_data):
    if token_data and "expires_at" in token_data:
        return time.time() < token_data["expires_at"]
    return False


def get_new_token(username, password, session):
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    response = session.post(url, data=data)
    response.raise_for_status()
    token_info = response.json()
    return token_info["access_token"], token_info["expires_in"]

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

    # Parse command line
    inps = cmdLineParse()
    username = inps.username
    password = inps.password
    token_file = os.path.expanduser(inps.token_file)
    match = None

    ###Compute interval
    tstart = datetime.datetime.strptime(inps.start, fmt)
    tend = datetime.datetime.strptime(inps.end, fmt)

    days = (tend - tstart).days
    print('Number of days to check: ', days)

    ranges = gatherExistingOrbits(inps.dirname)

    oType = 'precise'
    timebef = tstart - datetime.timedelta(days=1)
    timeaft = tend + datetime.timedelta(days=1)
    timebef = str(timebef.strftime('%Y-%m-%d'))
    timeaft = str(timeaft.strftime('%Y-%m-%d'))


    session = requests.Session()

    start_time = timebef + "T00:00:00.000Z"
    stop_time = timeaft + "T23:59:59.999Z"
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    for mission in satName:
        query = " and ".join([
            f"ContentDate/Start gt '{start_time}'",
            f"ContentDate/Start lt '{stop_time}'",
            f"ContentDate/End gt '{start_time}'",
            f"ContentDate/End lt '{stop_time}'",
            f"startswith(Name,'{mission}')",
            f"contains(Name,'AUX_POEORB')",
            ])
    

        match = None
        success = False
        
        try:
            r = session.get(url, verify=True, params={"$filter": query})
            r.raise_for_status()

            entries = json.loads(r.text, object_hook=lambda x: SimpleNamespace(**x)).value
            for entry in entries:
                entry_datefmt = "%Y-%m-%dT%H:%M:%S.000000Z"
                tbef = datetime.datetime.strptime(entry.ContentDate.Start, entry_datefmt)
                taft = datetime.datetime.strptime(entry.ContentDate.End, entry_datefmt)
                matchFileName = entry.Name
                match = entry.Id

                if match is not None:
                    token_data = get_saved_token_data(token_file)
                    if token_data and is_token_valid(token_data):
                        print("using saved access token")
                        token = token_data["access_token"]
                    else:
                        print("generating a new access token")
                        if username is None or password is None:
                            try:
                                import netrc
                                host = "dataspace.copernicus.eu"
                                creds = netrc.netrc().hosts[host]
                            except:
                                if username is None:
                                    username = input("Username: ")
                                if password is None:
                                    from getpass import getpass
                                    password = getpass("Password (will not be displayed): ")
                            else:
                                if username is None:
                                    username, _, _ = creds
                                if password is None:
                                    _, _, password = creds
                        token, expires_in = get_new_token(username, password, session)
                        save_token_data(token, expires_in, token_file)

                    output = os.path.join(inps.dirname, matchFileName)
                    res = download_file(match, output, session, token)
                    print("Orbit is downloaded successfully")
                    print(" ")

                else:
                    print("Orbit is not downloaded successfully")
                    print(" ")

        except:
            raise
