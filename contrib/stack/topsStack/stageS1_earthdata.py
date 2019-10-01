#!/usr/bin/env python3


def fetchCookies(infile):
    '''
    Derived from ASF's bulk downloader python script.
    '''
    import os
    from urllib.request import build_opener, install_opener, Request, urlopen
    from urllib.request import HTTPHandler, HTTPSHandler, HTTPCookieProcessor
    from urllib.error import HTTPError, URLError
    from http.cookiejar import MozillaCookieJar
    import netrc
    import base64


    def check_cookie_is_logged_in(jar):
        '''
        Return true if logged in already.
        '''
        for cookie in jar:
            if cookie.name == 'urs_user_already_logged':
                return True

        return False

    def check_cookie(cookie_jar):
        '''
        Check if cookie in the jar is any good.
        '''
        #Make sure cookie is still valid against a known file
        file_check = 'https://urs.earthdata.nasa.gov/profile'

        # Apply custom Redirect Hanlder
        ctx = {}
        opener = build_opener(HTTPCookieProcessor(cookie_jar), HTTPHandler(), HTTPSHandler(**ctx))
        install_opener(opener)

        # Attempt a HEAD request
        request = Request(file_check)
        request.get_method = lambda : 'HEAD'
        try:
            print (" > attempting to download {0}".format(file_check))
            response = urlopen(request, timeout=30)
            resp_code = response.getcode()
            # Make sure we're logged in
            if not check_cookie_is_logged_in(cookie_jar):
                 cookie_status = False

        except HTTPError:
            # If we ge this error, again, it likely means the user has not agreed to current EULA
            print ("\nIMPORTANT: ")
            print ("Your user appears to lack permissions to download data from the ASF Datapool.")
            print ("\n\nNew users: you must first log into Vertex and accept the EULA. In addition, your Study Area must be set at Earthdata https://urs.earthdata.nasa.gov")
            raise HTTPError('No permission to work with ASF datapool. \n')

        # This return codes indicate the USER has not been approved to download the data
        if resp_code in (300, 301, 302, 303):
            try:
                redir_url = response.info().getheader('Location')
            except AttributeError:
                redir_url = response.getheader('Location')

            print ("Redirect ({0}) occured, invalid cookie value!".format(resp_code))
            cookie_status = False

        # These are successes!
        if resp_code in (200, 307):
            cookie_status = True

        return cookie_status


    def get_new_cookie(infile):
        '''
        Get new cookie from earthdata.
        '''
        #URS
        urs = 'urs.earthdata.nasa.gov'
        username, dummy, password = netrc.netrc().authenticators(urs)
        
        #ASF credentials
        asf_urs4 = { 'url': 'https://urs.earthdata.nasa.gov/oauth/authorize',
                     'client': 'BO_n7nTIlMljdvU6kRRB3g',
                     'redir': 'https://vertex-retired.daac.asf.alaska.edu/services/urs4_token_request'}



        # Build URS4 Cookie request
        auth_cookie_url = asf_urs4['url'] + '?client_id=' + asf_urs4['client'] + '&redirect_uri=' + asf_urs4['redir'] + '&response_type=code&state='

        #python3
        user_pass = base64.b64encode (bytes(username+":"+password, "utf-8"))
        user_pass = user_pass.decode("utf-8")

        # Authenticate against URS, grab all the cookies
        ctx = {}
        cookie_jar = MozillaCookieJar()
        opener = build_opener(HTTPCookieProcessor(cookie_jar), HTTPHandler(), HTTPSHandler(**ctx))
        request = Request(auth_cookie_url, headers={"Authorization": "Basic {0}".format(user_pass)})

        # Watch out cookie rejection!
        try:
            response = opener.open(request)
        except HTTPError as e:
            if e.code == 401:
                print (" > Username and Password combo was not successful. Please try again.")
                return False
            else:
                # If an error happens here, the user most likely has not confirmed EULA.
                print ("\nIMPORTANT: There was an error obtaining a download cookie!")
                print ("Your user appears to lack permission to download data from the ASF Datapool.")
                print ("\n\nNew users: you must first log into Vertex and accept the EULA. In addition, your Study Area must be set at Earthdata https://urs.earthdata.nasa.gov")
                raise Exception('No permission to work with ASF datapool.')
        except URLError as e:
            print ("\nIMPORTANT: There was a problem communicating with URS, unable to obtain cookie. ")
            print ("Try cookie generation later.")
            raise Exception('Could not obtain cookie')

        # Did we get a cookie?
        if check_cookie_is_logged_in(cookie_jar):
            #COOKIE SUCCESS!
            cookie_jar.save(infile)
            return True

        # if we aren't successful generating the cookie, nothing will work. Stop here!
        print ("WARNING: Could not generate new cookie! Cannot proceed. Please try Username and Password again.")
        print ("Response was {0}.".format(response.getcode()))
        print ("\n\nNew users: you must first log into Vertex and accept the EULA. In addition, your Study Area must be set at Earthdata https://urs.earthdata.nasa.gov")
        raise Exception('Could not generate cookie!!! Check your login credentials....')
       

    #If cookie file exists load it
    if os.path.exists(infile):
        cookie_jar = MozillaCookieJar()
        cookie_jar.load(infile)


        # Get cookie is good, return
        if check_cookie(cookie_jar):
            # Save cookiejar
            cookie_jar.save(infile)
            return

    #Setup fresh cookies
    status = False
    while not status:
        status = get_new_cookie(infile)


class Sentinel1:
    '''
    Class for virtual download of S1 products.
    '''
    def __init__(self, url, dest):
        '''
        Constructor with URL.
        '''
        import os

        #URL
        self.url = url
        
        #Destination folder
        self.dest = os.path.join(dest, os.path.basename(url))

        #Product Type
        if "IW_GRD" in self.url:
            self.productType = "GRD"
        elif "IW_SLC" in self.url:
            self.productType = "SLC"
        else:
            raise Exception("Product type could not be determined for: {0}".format(self.url))

        #Write dummy zip file to test output can be written
        if os.path.exists(self.dest):
            print('Destination zip file already exists. Will be overwritten ....')
            os.remove(self.dest)
        self.createZip()

        ##Fetch manifest
        self.IPF = None   ##TODO: Get calibration XML for IPF 2.36 - low priority
        self.fetchManifest()

        ##Fetch annotation
        self.fetchAnnotation()

        ##Fetch images - TODO: GRD support
        if self.productType == "SLC":
            self.fetchSLCImagery()

    def createZip(self):
        '''
        Create local zip file to populate.
        '''
        import zipfile
        try:
            with zipfile.ZipFile(self.dest, mode='w') as myzip:
                with myzip.open('download.log', 'w') as myfile:
                    myfile.write('Downloaded with ISCE2\n'.encode('utf-8'))
        except:
            raise Exception('Could not create zipfile: {0}'.format(self.dest))

    def fetchManifest(self):
        '''
        Fetch manifest.safe
        '''
        import os
        import logging
        import zipfile
        from osgeo import gdal
        try:
            res = gdal.ReadDir(self.srcsafe)
            if 'manifest.safe' not in res:
                raise Exception('Manifest file not found in {0}'.format(self.srcsafe))
        except:
            raise Exception('Could not fetch manifest from {0}'.format(self.srcsafe))

        try:
            with zipfile.ZipFile(self.dest, mode='a') as myzip:
                with myzip.open('manifest.safe', 'w') as myfile:
                    logging.info('Fetching manifest.safe')
                    self.downloadFile( os.path.join(self.srcsafe, 'manifest.safe'), myfile)
        except:
            raise Exception('Could not download manifest.safe from {0} to {1}'.format(self.url, self.dest))

    def fetchAnnotation(self):
        '''
        Fetch annotation files.
        '''
        import os
        import logging
        from osgeo import gdal
        import zipfile

        dirname = os.path.join(self.srcsafe, 'annotation')
        res = gdal.ReadDir(dirname )

        try:
            with zipfile.ZipFile(self.dest, mode='a') as myzip:
                for ii in res:
                    if ii.endswith('.xml'):
                        srcname = os.path.join(dirname, ii)
                        destname = os.path.join( self.zip2safe, 'annotation', ii) 
                        logging.info('Fetching {0}'.format(srcname))
                        with myzip.open(destname, 'w') as myfile:
                            self.downloadFile( srcname, myfile)
        except:
            raise Exception('Could not download {0} from {1} to {2}'.format(ii, self.url, self.dest))

    def fetchSLCImagery(self):
        '''
        Create VRTs for TIFF files.
        '''
        import os
        import logging
        from osgeo import gdal
        import zipfile
        import isce
        from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1
    
        dirname = os.path.join(self.srcsafe, 'measurement')
        res = gdal.ReadDir(dirname)

        #If more were known about the tiff, this can be improved 
        vrt_template = """<VRTDataset rasterXSize="{samples}" rasterYSize="{lines}">
    <VRTRasterBand dataType="CInt16" band="1">
        <NoDataValue>0.0</NoDataValue>
        <SimpleSource>
            <SourceFilename relativeToVRT="0">{tiffname}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{samples}" RasterYSize="{lines}" DataType="CInt16" BlockXSize="{samples}" BlockYSize="1"/>
            <SrcRect xOff="0" yOff="0" xSize="{samples}" ySize="{lines}"/>
            <DstRect xOff="0" yOff="0" xSize="{samples}" ySize="{lines}"/>
        </SimpleSource>
    </VRTRasterBand>
</VRTDataset>"""

        
        #Parse annotation files to have it ready with information
        for ii in res:
            parts = ii.split('-')
            swath = int(parts[1][-1])
            pol = parts[3]
           
            ##Read and parse metadata for swath
            xmlname = ii.replace('.tiff', '.xml')

            try:
                reader = Sentinel1()
                reader.configure()
                reader.xml = [ os.path.join("/vsizip", self.dest, self.zip2safe, 'annotation', xmlname) ]
                reader.manifest = [ os.path.join("/vsizip", self.dest, self.zip2safe, 'manifest.safe') ] 
                reader.swathNumber = swath
                reader.polarization = pol
                reader.parse()

                vrtstr = vrt_template.format( samples = reader.product.bursts[0].numberOfSamples,
                                              lines = reader.product.bursts[0].numberOfLines * len(reader.product.bursts),
                                              tiffname = os.path.join(self.srcsafe, 'measurement', ii))

                #Write the VRT to zip file
                with zipfile.ZipFile(self.dest, mode='a') as myzip:
                    destname = os.path.join( self.zip2safe, 'measurement', ii + '.vrt')
                    with myzip.open(destname, 'w') as myfile:
                        myfile.write(vrtstr.encode('utf-8'))
            except:
                raise Exception('Could not create vrt for {0} at {1} in {2}'.format(ii, self.url, self.dest))

    @property
    def vsi(self):
        import os
        return os.path.join('/vsizip/vsicurl', self.url)

    @property
    def srcsafe(self):
        import os
        return os.path.join( self.vsi, self.zip2safe)

    @property
    def zip2safe(self):
        '''
        Get safe directory path from zip name.
        '''
        import os
        return os.path.basename(self.url).replace('.zip', '.SAFE')


    @staticmethod
    def downloadFile(inname, destid):
        from osgeo import gdal

        #Get file size
        stats = gdal.VSIStatL(inname)
        if stats is None:
                raise Exception('Could not get stats for {0}'.format(inname))
            
        #Copy file to local folder
        success = False
        while not success:
            try:
                vfid = gdal.VSIFOpenL(inname, 'rb')
                data = gdal.VSIFReadL(1, stats.size, vfid)
                gdal.VSIFCloseL(vfid)
                success = True
            except AttributeError as errmsg:
                if errmsg.endswith('307'):
                    print('Redirected on {0}. Retrying ... '.format(inname))
            except Excepton as err:
                print(err)
                raise Exception('Could not download file: {0}'.format(inname))

        #Write to destination id
        destid.write(data)




def cmdLineParse():
    '''
    Command line parser.
    '''
    import argparse

    parser = argparse.ArgumentParser(
             description='Download S1 annotation files with VRT pointing to tiff files')
    parser.add_argument('-i', '--input', dest='inlist', type=str,
                        required=True, help='Text file with URLs to fetch')
    parser.add_argument('-o', '--output', dest='outdir', type=str,
                        default='.', help='Output folder to store the data in')
    parser.add_argument('-c', '--cookies', dest='cookies', type=str,
                        default='asfcookies.txt', help='Path to cookies file')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        default=False, help='Set to CPL_DEBUG to ON')

    return parser.parse_args()

def setupCookies(cookiefile, inurl):
    '''
    Get information from netrc and setup cookies.
    '''



def main(inps=None):
    '''
    Main driver.
    '''

    #Check / Fetch Earthdata cookies
    #This is derived from ASF's bulk download script
    #Needed since vsicurl recognizes 200 as the only right response codde
    #From ASF script, 307 can also be a valid response.
    #If vsicurl interface is updated, might no longer be needed.
    fetchCookies(inps.cookies)

    ###Setup GDAL with cookies
    import os
    import logging
    from osgeo import gdal
    gdal.UseExceptions()

    gdal.SetConfigOption('GDAL_HTTP_COOKIEFILE',inps.cookies)
    gdal.SetConfigOption('GDAL_HTTP_COOKIEJAR', inps.cookies)
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
    if inps.debug:
        gdal.SetConfigOption('CPL_DEBUG', 'ON')
        gdal.SetConfigOption('CPL_CURL_VERBOSE', 'YES')
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
        

    ##Read in URLs into a list
    urlList = []
    try:
        with open(inps.inlist, 'r') as fid:
            for cnt, line in enumerate(fid):
                urlList.append(line.strip())

    except:
        raise Exception('Could not parse input file "{0}" as a list of line separated URLs'.format(inps.inlist))



    for url in urlList:
        logging.info('Downloading: {0}'.format(url))
        downloader = Sentinel1(url, inps.outdir)




if __name__ == '__main__':
    '''
    Script driver.
    '''
    #Parse command line
    inps = cmdLineParse()

    #Process
    main(inps)
