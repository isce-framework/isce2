#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from __future__ import print_function
import os
import sys
import isce
from iscesys.Parsers.FileParserFactory import createFileParser
from iscesys.ImageApi import DataAccessor as DA
from isceobj.Util import key_of_same_content
import math
class Display(object):

    def getRscExt(self,ext):
        ret = ''
        for k in self._mapDataType['rsc'].keys():
            if self._ext[k].count(ext):
                ret = k
                break
        return ret

    def setIfNotPresent(self,opList,option,default):
        # check if option is present in the opList
        # and if not add it with default value
        if not option in opList:
            opList.append(option)
            opList.append(default)
    def getIfPresent(self,opList,option):
        # check if option is present in the opList
        # and return the value. Return None if not present
        ret = None
        try:
            indx = opList.index(option)
            ret = opList[indx+1]
            #remove the option
            opList.pop(indx)
            #remove the value. same indx since just popped one
            opList.pop(indx)
        except:
            # the option is not present
            pass
        return ret
    def createCommand(self,options):
        ext = options['ext']
        dataType = options['dataType']
        image = options['image']
        width = options['width']
        # numBands and length only used for isce products, not roi_pac
        numBands = options['numBands'] if 'numBands' in options else 1
        length = options['length'] if 'length' in options else 0
        argv = options['other']


        command = ''
        if ext in self._ext['cpx'] or ext in self._ext['scor'] or ext in self._ext['byt']:
            command = image + ' ' + dataType + ' -s ' + str(width) + ' ' + ' '.join(argv)
        elif ext in self._ext['rmg']:
            command = image + ' -s ' + str(width)  + ' ' + ' -rmg -RMG-Mag -CW -RMG-Hgt ' + ' '.join(argv)
        elif ext in self._ext['unw']:
            tpi=str(2.*math.pi)
            self.setIfNotPresent(argv,'-wrap',tpi)
            command = image + ' -s ' + str(width) +  '  -amp ' + dataType + ' -rtlr ' + str(width*int(dataType[2:])) + ' -CW -unw '  + dataType + ' -rhdr ' + str(width*int(dataType[2:])) + ' -cmap cmy ' + ' '.join(argv)
        elif ext in self._ext['cor']:
            self.setIfNotPresent(argv,'-wrap','1.2')
            if numBands == 2:
                command = image + ' -s ' + str(width) +  '  -rmg -RMG-Mag -CW -RMG-Hgt '  + ' '.join(argv)
                command = image + ' -s ' + str(width) +  '  -rmg -RMG-Mag -CW -RMG-Hgt '  + ' '.join(argv)
            elif numBands == 1:
                command = image + ' -s ' + str(width) + ' -cmap cmy ' + ' '.join(argv)
        elif ext in self._ext['dem']:
            self.setIfNotPresent(argv,'-wrap','100')
            self.setIfNotPresent(argv,'-cmap','cmy')
            command =  image + ' -slope '  + dataType + ' -s ' + str(width) + ' ' + image + ' ' + dataType +' -s ' + str(width) + ' ' + ' '.join(argv)
        elif ext in self._ext['amp']:
            #get the numeric part of the data type which corresponds to the size
            chdr = dataType[2:]
            ctlr = dataType[2:]
            newChdr = self.getIfPresent(argv,'-chdr')
            if not newChdr is None:
                chdr = newChdr
            newCtlr = self.getIfPresent(argv,'-ctlr')
            if not newCtlr is None:
                ctlr = newCtlr

            command =  image + ' -s ' + str(width) + ' -CW ' +' -amp1 ' + dataType + ' -ctlr ' + ctlr + ' -amp2 ' + dataType + '  -chdr ' + chdr +  ' ' + ' '.join(argv)

        elif ext in self._ext['bil']:
            sizeof = self.getDataSize(dataType)
            command = image + ' -s ' + str(width)
            for i in range(1,numBands + 1):#do it one based
                rhdr = (i - 1)*width*sizeof
                rtlr = (numBands - i)*width*sizeof
                command += ' -ch' + str(i) + ' ' + dataType
                command += ((' -rhdr ' + str(rhdr)) if rhdr else '') + ' '
                command += ((' -rtlr ' + str(rtlr)) if rtlr else '') + ' '
                command += ' '.join(argv)
        elif ext in self._ext['bip']:
            sizeof = self.getDataSize(dataType)
            command = image + ' -s ' + str(width)
            for i in range(1,numBands + 1):#do it one based
                chdr = (i - 1)*sizeof
                ctlr = (numBands - i)*sizeof
                command += ' -ch' + str(i) + ' ' + dataType
                command += ((' -chdr ' + str(chdr)) if chdr else '') + ' '
                command += ((' -ctlr ' + str(ctlr)) if ctlr else '') + ' '
                command += ' '.join(argv)
        elif ext in self._ext['bsq']:
            sizeof = self.getDataSize(dataType)
            command = image + ' -s ' + str(width)
            for i in range(1,numBands + 1):#do it one based
                shdr = (i - 1)*width*length*sizeof
                stlr = (numBands - i)*width*length*sizeof
                command += ' -ch' + str(i) + ' ' + dataType
                command += ((' -shdr ' + str(shdr)) if shdr else '') + ' '
                command += ((' -stlr ' + str(stlr)) if stlr else '') + ' '
                command += ' '.join(argv)


        return command

    def parse(self,argv):
        ret = {}
        upTo = 0

        #get global options '-z val' and '-kml'
        #the first option could possibly be the -z applied globally
        try:
            indx = argv.index('-z')
            ret['-z'] = argv[indx+1]
            # remove the -z val from list.
            argv.pop(indx)
            # same indx since popped the previous
            argv.pop(indx)
        except:
            #not present
            pass
        try:
            indx = argv.index('-P')
            ret['-P'] = '-P'
            # remove the -P  from list.
            argv.pop(indx)
        except:
            #not present
            pass
        try:
            indx = argv.index('-kml')
            ret['-kml'] = argv[indx+1]
            # remove the -kml and val  from list.
            argv.pop(indx)
            argv.pop(indx)
        except:
            #not present
            pass

        # the reamining part of the command has to be
        # file -op val -op val file -op val -op val ....
        # so the different file options are recognized with two argv with
        # no dash are present (first is a val and second an image)
        imgOpt = []
        parOpt = []
        pos = 0
        while(True):
            if(pos >= len(argv)):
                imgOpt.append(parOpt)
                break
            # is an option
            if argv[pos].startswith('-'):
                parOpt.append(argv[pos])
                pos += 1
                parOpt.append(argv[pos])
            # is a metadata file
            else:
                # is the first time, just add the image
                if not parOpt:
                    parOpt.append(argv[pos])
                # else save what was there before, clear and add the new image
                else:
                    imgOpt.append(parOpt)
                    parOpt = [argv[pos]]
            pos += 1


        ret['imageArgs'] = imgOpt

        return ret

    def getMetaFile(self,image):

        metafile = None
        for ext in self._metaExtensions:
            if os.path.exists(image + ext):
                metafile = image + ext
                break
        if metafile is None:
            print("Error. Cannot find any metadata file associated with the image",image)
            raise ValueError

        return metafile


    def getInfo(self,image):
        metafile = self.getMetaFile(image)
        ret = None
        if(metafile.endswith('xml')):
            ret = self.getInfoFromXml(metafile,image)
        elif(metafile.endswith('rsc')):
            ret = self.getInfoFromRsc(metafile,image)

        else:
            print("Error. Metadata file must have extension 'rsc' or 'xml'")
            raise ValueError
        return ret

    def getInfoFromXml(self,imagexml,image):
        """ Determines  image name, width, image type and data type from input xml"""
        # first is alway the xml file
        ext = None
        dataType = None
        width = None
        length = None
        PA = createFileParser('xml')
        dictNow, dictFact, dictMisc = PA.parse(imagexml) #get only the property dictionary
        numBands = 0

        numBands = key_of_same_content('number_bands', dictNow)[1]
        dataTypeImage = key_of_same_content('data_type', dictNow)[1]
        dataType = self._mapDataType['xml'][dataTypeImage]
        try:#new format of image
            coordinate1  =  key_of_same_content('coordinate1',dictNow)[1]
            width = key_of_same_content('size',coordinate1)[1]
            coordinate2  =  key_of_same_content('coordinate2',dictNow)[1]
            length = key_of_same_content('size',coordinate2)[1]
            try:#only for geo image to create kml
                self._width.append(float(width))
                self._startLon.append(float(key_of_same_content('startingValue',coordinate1)[1]))
                self._deltaLon.append(float(key_of_same_content('delta',coordinate1)[1]))

                coordinate2  =  key_of_same_content('coordinate2',dictNow)[1]
                self._length.append(float(key_of_same_content('size',coordinate2)[1]))
                self._startLat.append(float(key_of_same_content('startingValue',coordinate2)[1]))
                self._deltaLat.append(float(key_of_same_content('delta',coordinate2)[1]))
                self._names.append(imagexml.replace('.xml',''))
            except Exception as e:
                pass # not a geo image
        except:# use old format
            try:
                width = key_of_same_content('width',dictNow)[1]
            except:
                print("Error. Cannot figure out width from input file.")
                raise Exception

        ext = self.getIsceExt(dictNow,image)

        if ext is None or dataType is None or width is None:#nothing worked. Through exception caught next
            print("Error. Cannot figure out extension from input file.")
            raise Exception
        return {'image':image,'ext':ext,'width':width,'length':length,'dataType':dataType,'numBands':numBands}

    def isExt(self,ext):
        found = False
        for k,v in self._ext.items():
            if ext in v:
                found  = True
                break
        return found

    #try few things to get the right extension
    def getIsceExt(self,info,imagename):
        ext = None
        # try to see if the image has the property imageType
        try:
            ext  =  key_of_same_content('image_type',info)[1]
            #if it is not a valid extension try something else
            if(not self.isExt(ext)):
                raise Exception
        except:
            # if not try to get the ext from the filename
            try:
                nameSplit = imagename.split('.')
                if len(nameSplit) > 1:#there was atleast one dot in the name
                    ext = nameSplit[-1]
                if(not self.isExt(ext)):
                    raise Exception
            except:
                #try to use the scheme
                try:
                    scheme = key_of_same_content('scheme',info)[1]
                    ext = scheme.lower()
                    if(not self.isExt(ext)):
                        raise Exception
                except:
                    ext = None
        return ext

    def getInfoFromRsc(self,imagersc,image):
        """ Determines image name, width, image type and data type from input rsc"""
        try:
            PA = createFileParser('rsc')
            dictOut = PA.parse(imagersc)
            #dictOut has a top node that is just a name
            dictNow = dictOut[list(dictOut.keys())[0]]
            if 'WIDTH' in dictNow:
                width = int(dictNow['WIDTH'])
            try:
                if 'LAT_REF1' in dictNow:
                    #extract the geo info
                    self._width.append(float(width))
                    self._startLon.append(float(dictNow['X_FIRST']))
                    self._deltaLon.append(float(dictNow['X_STEP']))
                    self._length.append(float(dictNow['FILE_LENGTH']))
                    self._startLat.append(float(dictNow['Y_FIRST']))
                    self._deltaLat.append(float(dictNow['Y_STEP']))
                    self._names.append(image)
            except:
                pass#not a geo file
        except:
            print("Error. Cannot figure out width from input file.")
            raise Exception
        # assume imagersc = 'name.ext.rsc'
        try:
            ext = imagersc.split('.')[-2]
        except:
            print("Error. Cannot figure out extension from input file.")
            raise Exception
        found = False

        for k,v in self._ext.items():
            if ext in v:
                found  = True
                break
        if not found:
            print("Error. Invalid image extension",ext,".")
            self.printExtensions()
            raise Exception

        extNow = self.getRscExt(ext)

        dataType = self._mapDataType['rsc'][extNow]

        return {'image':image,'ext':ext,'width':width,'dataType':dataType}

    def getCommand(self,options):
        #if creating kml then commands is a list of singles mdx commands, one per input image with -P option
        #otherwise is a string made of a unique command  for all the images at once
        commands = ''
        command = 'mdx'
        if '-z' in options:
            command += ' -z ' + options['-z']
        if ('-kml' in options or '-P' in options):
            command += ' -P '
            commands = []
        #build command for each single image
        for listOp in options['imageArgs']:
            #first arg is the metadata file. opDict contains image,ext,width,dataType
            opDict = self.getInfo(listOp[0])
            if not (opDict is None):
                try:
                    # if any extra command put it into other
                    opDict['other'] = listOp[1:]
                except:
                    # only had the metadata in listOp
                    pass

            if not '-kml' in options:
                command += ' ' + self.createCommand(opDict)
            else:
                commands.append(command + ' ' + self.createCommand(opDict))


        if not '-kml' in options:
            commands = command

        return commands





    def printExtensions(self):

        #perhaps turn it into a dictionary with key = extension and value = description
        print("Supported extensions:")
        for k,v in self._ext.items():
            for ext in v:
                print(ext)
    def printUsage(self):
        print("  Usage:\n")
        print("  mdx.py   filename [-wrap wrap] ... [-z zoom -kml output.kml]\n")
        print("  where\n")
        for mess in self._docIn:
            print(mess)
        print('\n  or\n')
        print("  mdx.py -ext\n")
        print("  to see the supported image extensions.\n\n")

    def mdx(self, argv=None):
        if argv is None:
            self.printUsage()
        else:
            if len(argv) == 1 and argv[0] == '-ext':
                self.printExtensions()
            elif len(argv) == 0:
                self.printUsage()
            else:
                #argv is  modified in parse and -kml is removed so check before parsing
                doKml = self.isKml(argv)
                options  = self.parse(argv)
                command = self.getCommand(options)
                if not doKml:
                    print("Running:",command)
                    os.system(command)
                else:
                    #options['-kml'] is the output filename passed as input arg
                    self.createKml(options['-kml'],command)

    def createKml(self,name,commands):

        fp = open(name,'w')
        fp.write('<?xml version="1.0" encoding="UTF-8"?>\n\
        <kml xmlns="http://www.opengis.net/kml/2.2">\n<Folder>\n')

        #mdx creates a out.ppm file in the cwd
        ppm = 'out.ppm'
        cwd = os.getcwd()
        for i in range(len(self._startLat)):
            os.system(commands[i])
            lat1 = self._startLat[i]
            lat2 = self._startLat[i] + self._deltaLat[i]*(self._length[i] - 1)
            lon1 = self._startLon[i]
            lon2 = self._startLon[i] + self._deltaLon[i]*(self._width[i] - 1)
            maxLon = max(lon1,lon2)
            minLon = min(lon1,lon2)
            maxLat = max(lat1,lat2)
            minLat = min(lat1,lat2)
            icon = os.path.join(cwd,os.path.basename(self._names[i])) + '.png'
            command = 'convert ' + ppm + ' -resize 80% -transparent black' + ' ' + icon
            os.system(command)
            os.remove(ppm)
            self.appendToKmlFile(fp,os.path.basename(self._names[i]),icon,[maxLat,minLat,maxLon,minLon])
        fp.write('</Folder>\n</kml>\n')
        fp.close()

    def appendToKmlFile(self,fp,name,icon,bbox):
        fp.write('\t<GroundOverlay>\n')
        fp.write('\t\t<name>%s</name>\n'%name)
        fp.write('\t\t<color>afffffff</color>\n')
        fp.write('\t\t<drawOrder>1</drawOrder>\n')
        fp.write('\t\t<Icon>\n')
        fp.write('\t\t\t<href>%s</href>\n'%icon)
        fp.write('\t\t</Icon>\n')
        fp.write('\t\t<LatLonBox>\n')
        fp.write('\t\t\t<north>%f</north>\n'%bbox[0])
        fp.write('\t\t\t<south>%f</south>\n'%bbox[1])
        fp.write('\t\t\t<east>%f</east>\n'%bbox[2])
        fp.write('\t\t\t<west>%f</west>\n'%bbox[3])
        fp.write('\t\t</LatLonBox>\n')
        fp.write('\t</GroundOverlay>\n')


    def isKml(self,argv):
        try:
            argv.index('-kml')
            ret = True
        except:
            ret = False

        return ret

    def getDataSize(self,dataType):
        try:
            size = int(dataType[2:])
        except:
            size = 0
        return size

    def __init__(self):



        size = DA.getTypeSize('LONG') #the size depends on the platform. the ImageAPI does e sizeof(long int) and returns the size
        #NOTE the unw  doent't need a datatype so put ''
        self._mapDataType = {'xml':{'BYTE':'-i1','SHORT':'-i2','CFLOAT':'-c8','FLOAT':'-r4','INT':'-i4','LONG':'-i'+ str(size),'DOUBLE':'-r8'},'rsc':{'cpx':'-c8','rmg':'-r4','scor':'-r4','dem':'-i2','byt':'-i1','amp':'-r4','unw':'-r4','cor':''}}

        self._docIn = [
                       '  mdx.py  : displays one or more data files simultaneously by ',
                       '            specifying their names as input. The maximun number,'
                       '            of images that can be displayed depends on the machine',
                       '            architecture and  mdx limits. If displayed (no -kml flag)',
                       '            the images don\'t need to have the same extension, but need',
                       '            to have same width.',
                       ' ',

                       '  filename: input file containing the image metadata.',
                       '            Metadata files must be of format filename.{xml,rsc}',
                       '            and must be  present in the same directory as filename.',
                       '            Different formats (xml,rsc) can be mixed.',
                       ' ',
                       '  -wrap   : sets display scaling to wrap mode with a modules of Pi.',
                       '            It must follow the filename to which the wrap is applied.',
                       ' ',
                       '  ...     : the command can be repeated for different images.',
                       ' ',
                       '  -z      : zoom factor (+ or -) to apply to all layers. It\'s optional',
                       '            and can appear anywhere in the command sequence and must',
                       '            appear only once.',
                       ' ',
                       '  -kml    : only for geocoded images it creates a klm file with all the',
                       '            input images overlaid. Each layer can be turn on or off in ',
                       '            Goolge Earth. It\'s optional and can appear anywhere in the ',
                       '            command sequence and must appear only once. The images don\'t ',
                       '             need to be co-registed.',
                       ' ',
                       '  Examples:',
                       '  mdx.py 01_02.int                                  # Standard way to run mdx.py',
                       '  mdx.py -P 01_02.int                               # Create a ppm image named out.ppm',
                       '  mdx.py 03_04.int 05_06.int -z -8                  # Display two images; zoom out by 8',
                       '  mdx.py 03_04.geo -z 8 05_06.geo -kml fileout.klm  # Create a kml file named fileout.kml with two',
                       '                                                    # layers, one per image. Both images are zoomed in',
                       '                                                    # by a factor of 8     ' ,
                       '  mdx.py 03_04.int 05_06.int -wrap 6.28             # Display two images. Wrap the second modulo 2Pi',
                       ' ']
        # the input file is the image itself. Search for the same filename
        # and one of the extensions below to figure out the metadata type
        self._metaExtensions = ['.xml','.rsc']
        self._ext = {}
        self._ext['cpx'] = ['slc','int','flat','mph','cpx']
        self._ext['rmg'] = ['hgt','hgt_holes','rect','rmg']
        self._ext['scor'] = ['scor']
        self._ext['dem'] = ['dem','dte','dtm']
        self._ext['unw'] = ['unw']
        self._ext['cor'] = ['cor']
        self._ext['byt'] = ['byt','flg']
        self._ext['amp'] = ['amp']
        self._ext['bil'] = ['bil']
        self._ext['bip'] = ['bip']
        self._ext['bsq'] = ['bsq']
        # save this quantities in the case we are dealing with a geo image
        self._startLat = []
        self._deltaLat = []
        self._startLon = []
        self._deltaLon = []
        self._length = []
        self._width = []
        self._names = []

def main():
    # just for testing purposes
    ds = Display()
    """ #test parser
    print(ds.parse(sys.argv[1:]))
    """
    """ test info extractor from xml
    print(ds.getInfoFromXml(sys.argv[1]))
    """
    """ test info extractor from rsc
    print(ds.getInfoFromRsc(sys.argv[1]))
    """
    ds.mdx(sys.argv[1:])
if __name__ == '__main__':
    sys.exit(main())
