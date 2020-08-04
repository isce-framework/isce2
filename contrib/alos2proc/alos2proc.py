# Cunren Liang
# Copyright 2018, Caltech

import os
import copy
import ctypes
import logging
import isceobj
from xml.etree.ElementTree import ElementTree

def mbf(inputfile, outputfile, prf, prf_frac, nb, nbg, nboff, bsl, kacoeff, dopcoeff1, dopcoeff2):
    #############################
    # inputfile:       input file
    # outputfile:      output file
    # prf:             PRF
    # prf_frac:        fraction of PRF processed
    #                     (represents azimuth bandwidth)
    # nb:              number of lines in a burst
    #                     (float, in terms of 1/PRF)
    # nbg:             number of lines in a burst gap
    #                     (float, in terms of 1/PRF)
    # nboff:           number of unsynchronized lines in a burst
    #                     (float, in terms of 1/PRF, with sign, see burst_sync.py for rules of sign)
    #                     (the image to be processed is always considered to be reference)
    # bsl:             start line number of a burst
    #                     (float, the line number of the first line of the full-aperture SLC is zero)
    #                     (no need to be first burst, any one is OK)

    # kacoeff[0-3]:    FM rate coefficients
    #                     (four coefficients of a third order polynomial with regard to)
    #                     (range sample number. range sample number starts with zero)

    # dopcoeff1[0-3]:  Doppler centroid frequency coefficients of this image
    #                     (four coefficients of a third order polynomial with regard to)
    #                     (range sample number. range sample number starts with zero)

    # dopcoeff2[0-3]:  Doppler centroid frequency coefficients of the other image
    #                     (four coefficients of a third order polynomial with regard to)
    #                     (range sample number. range sample number starts with zero)
    #############################

    #examples:
    # kacoeff = [-625.771055784221, 0.007887946763383646, -9.10142814131697e-08, 0.0]
    # dopcoeff1 = [-0.013424025141940908, -6.820475445542178e-08, 0.0, 0.0]
    # dopcoeff2 = [-0.013408164465406417, -7.216577938502655e-08, 3.187158113584236e-24, -9.081842749918244e-28]

    img = isceobj.createSlcImage()
    img.load(inputfile + '.xml')

    width = img.getWidth()
    length = img.getLength()

    inputimage = find_vrt_file(inputfile+'.vrt', 'SourceFilename')
    byteorder = find_vrt_keyword(inputfile+'.vrt', 'ByteOrder')
    if byteorder == 'LSB':
        byteorder = 0
    else:
        byteorder = 1
    imageoffset = find_vrt_keyword(inputfile+'.vrt', 'ImageOffset')
    imageoffset = int(imageoffset)
    lineoffset = find_vrt_keyword(inputfile+'.vrt', 'LineOffset')
    lineoffset = int(lineoffset)

    #lineoffset = lineoffset - width * 8
    #imageoffset = imageoffset - lineoffset

    if type(kacoeff) != list:
        raise Exception('kacoeff must be a python list.\n')
        if len(kacoeff) != 4:
            raise Exception('kacoeff must have four elements.\n')
    if type(dopcoeff1) != list:
        raise Exception('dopcoeff1 must be a python list.\n')
        if len(dopcoeff1) != 4:
            raise Exception('dopcoeff1 must have four elements.\n')
    if type(dopcoeff2) != list:
        raise Exception('dopcoeff2 must be a python list.\n')
        if len(dopcoeff2) != 4:
            raise Exception('dopcoeff2 must have four elements.\n')

    filters = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libalos2proc.so'))
    filters.mbf(
        ctypes.c_char_p(bytes(inputimage,'utf-8')),
        ctypes.c_char_p(bytes(outputfile,'utf-8')),
        ctypes.c_int(width),
        ctypes.c_int(length),
        ctypes.c_float(prf),
        ctypes.c_float(prf_frac),
        ctypes.c_float(nb),
        ctypes.c_float(nbg),
        ctypes.c_float(nboff),
        ctypes.c_float(bsl),
        (ctypes.c_float * len(kacoeff))(*kacoeff),
        (ctypes.c_float * len(dopcoeff1))(*dopcoeff1),
        (ctypes.c_float * len(dopcoeff2))(*dopcoeff2),
        ctypes.c_int(byteorder),
        ctypes.c_long(imageoffset),
        ctypes.c_long(lineoffset)
        )

    #img = isceobj.createSlcImage()
    #img.load(inputfile + '.xml')
    img.setFilename(outputfile)
    img.extraFilename = outputfile + '.vrt'
    img.setAccessMode('READ')
    img.renderHdr()


def rg_filter(inputfile, nout, outputfile, bw, bc, nfilter, nfft, beta, zero_cf, offset):
    #############################
    # inputfile:  input file
    # nout:       number of output files
    # outputfile: [value_of_out_1, value_of_out_2, value_of_out_3...] output files
    # bw:         [value_of_out_1, value_of_out_2, value_of_out_3...] filter bandwidth divided by sampling frequency [0, 1]
    # bc:         [value_of_out_1, value_of_out_2, value_of_out_3...] filter center frequency divided by sampling frequency

    # nfilter:    number samples of the filter (odd). Reference Value: 65
    # nfft:       number of samples of the FFT. Reference Value: 1024
    # beta:       kaiser window beta. Reference Value: 1.0
    # zero_cf:    if bc != 0.0, move center frequency to zero? 0: Yes (Reference Value). 1: No.
    # offset:     offset (in samples) of linear phase for moving center frequency. Reference Value: 0.0
    #############################

    #examples
    #outputfile = ['result/crop_filt_1.slc', 'result/crop_filt_2.slc']
    #bw = [0.3, 0.3]
    #bc = [0.1, -0.1]

    img = isceobj.createSlcImage()
    img.load(inputfile + '.xml')

    width = img.getWidth()
    length = img.getLength()

    inputimage = find_vrt_file(inputfile+'.vrt', 'SourceFilename')
    byteorder = find_vrt_keyword(inputfile+'.vrt', 'ByteOrder')
    if byteorder == 'LSB':
        byteorder = 0
    else:
        byteorder = 1
    imageoffset = find_vrt_keyword(inputfile+'.vrt', 'ImageOffset')
    imageoffset = int(imageoffset)
    lineoffset = find_vrt_keyword(inputfile+'.vrt', 'LineOffset')
    lineoffset = int(lineoffset)

    #lineoffset = lineoffset - width * 8
    #imageoffset = imageoffset - lineoffset

    outputfile2 = copy.deepcopy(outputfile)
    if type(outputfile) != list:
        raise Exception('outputfile must be a python list.\n')
        if len(outputfile) != nout:
            raise Exception('number of output files is not equal to outputfile list length.\n')
    else:
        tmp = []
        for x in outputfile:
            tmp.append(bytes(x,'utf-8'))
        outputfile = tmp

    if type(bw) != list:
        raise Exception('bw must be a python list.\n')
        if len(bw) != nout:
            raise Exception('number of output files is not equal to bw list length.\n')

    if type(bc) != list:
        raise Exception('bc must be a python list.\n')
        if len(bc) != nout:
            raise Exception('number of output files is not equal to bc list length.\n')

    filters = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libalos2proc.so'))
    filters.rg_filter(
        ctypes.c_char_p(bytes(inputimage,'utf-8')),
        ctypes.c_int(width),
        ctypes.c_int(length),
        ctypes.c_int(nout),
        (ctypes.c_char_p * len(outputfile))(*outputfile),
        (ctypes.c_float * len(bw))(*bw),
        (ctypes.c_float * len(bc))(*bc),
        ctypes.c_int(nfilter),
        ctypes.c_int(nfft),
        ctypes.c_float(beta),
        ctypes.c_int(zero_cf),
        ctypes.c_float(offset),
        ctypes.c_int(byteorder),
        ctypes.c_long(imageoffset),
        ctypes.c_long(lineoffset)
        )

    #img = isceobj.createSlcImage()
    #img.load(inputfile + '.xml')
    for x in outputfile2:
        img.setFilename(x)
        img.extraFilename = x + '.vrt'
        img.setAccessMode('READ')
        img.renderHdr()


def resamp(slc2, rslc2, rgoff_file, azoff_file, nrg1, naz1, prf, dopcoeff, rgcoef=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], azcoef=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], azpos_off=0.0, verbose=0):
    #############################
    # mandatory:
    # slc2:             secondary image
    # rslc2:            resampled secondary image
    # rgoff_file:       range offset file. if no range offset file, specify 'fake'
    # azoff_file:       azimuth offset file. if no azimuth offset file, specify 'fake'
    # nrg1:             number of columns in reference image
    # naz1:             number of lines in reference image
    # prf:              PRF of secondary image
    # dopcoeff[0]-[3]:  Doppler centroid frequency coefficents
    # optional:
    # rgcoef[0]-[9]:    range offset polynomial coefficents. First of two fit results of resamp_roi
    # azcoef[0]-[9]:    azimuth offset polynomial coefficents. First of two fit results of resamp_roi
    # azpos_off:        azimuth position offset. Azimuth line number (column 3) of first offset in culled offset file
    # verbose:          if not zero, print resampling info
    #############################

    #examples:
    # dopcoeff = [-0.013424025141940908, -6.820475445542178e-08, 0.0, 0.0]

    img = isceobj.createSlcImage()
    img.load(slc2 + '.xml')

    width = img.getWidth()
    length = img.getLength()

    inputimage = find_vrt_file(slc2+'.vrt', 'SourceFilename')
    byteorder = find_vrt_keyword(slc2+'.vrt', 'ByteOrder')
    if byteorder == 'LSB':
        byteorder = 0
    else:
        byteorder = 1
    imageoffset = find_vrt_keyword(slc2+'.vrt', 'ImageOffset')
    imageoffset = int(imageoffset)
    lineoffset = find_vrt_keyword(slc2+'.vrt', 'LineOffset')
    lineoffset = int(lineoffset)

    #lineoffset = lineoffset - width * 8
    #imageoffset = imageoffset - lineoffset

    if type(dopcoeff) != list:
        raise Exception('dopcoeff must be a python list.\n')
        if len(dopcoeff) != 4:
            raise Exception('dopcoeff must have four elements.\n')
    if type(rgcoef) != list:
        raise Exception('rgcoef must be a python list.\n')
        if len(rgcoef) != 10:
            raise Exception('rgcoef must have 10 elements.\n')
    if type(azcoef) != list:
        raise Exception('azcoef must be a python list.\n')
        if len(azcoef) != 10:
            raise Exception('azcoef must have 10 elements.\n')

    filters = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libalos2proc.so'))
    filters.resamp(
        ctypes.c_char_p(bytes(inputimage,'utf-8')),
        ctypes.c_char_p(bytes(rslc2,'utf-8')),
        ctypes.c_char_p(bytes(rgoff_file,'utf-8')),
        ctypes.c_char_p(bytes(azoff_file,'utf-8')),
        ctypes.c_int(nrg1),
        ctypes.c_int(naz1),
        ctypes.c_int(width),
        ctypes.c_int(length),
        ctypes.c_float(prf),
        (ctypes.c_float * len(dopcoeff))(*dopcoeff),
        (ctypes.c_float * len(rgcoef))(*rgcoef),
        (ctypes.c_float * len(azcoef))(*azcoef),
        ctypes.c_float(azpos_off),
        ctypes.c_int(byteorder),
        ctypes.c_long(imageoffset),
        ctypes.c_long(lineoffset),
        ctypes.c_int(verbose)
        )

    #img = isceobj.createSlcImage()
    #img.load(inputfile + '.xml')
    img.setFilename(rslc2)
    img.extraFilename = rslc2 + '.vrt'
    img.setWidth(nrg1)
    img.setLength(naz1)
    img.setAccessMode('READ')
    img.renderHdr()


def mosaicsubswath(outputfile, nrgout, nazout, delta, diffflag, n, inputfile, nrgin, nrgoff, nazoff, phc, oflag):
    '''
    outputfile: (char) output file
    nrgout:     (int)  number of output samples
    nazout:     (int)  number of output lines
    delta:      (int)  edge to be removed of the overlap area (number of samples)
    diffflag:   (int)  whether output the overlap area as two-band BIL image. 0: yes, otherwise: no
    n:          (int)  number of input file
    inputfile:  (char list) [value_of_out_1, value_of_out_2, value_of_out_3...] input files to mosaic
    nrgin:      (int list)  [value_of_out_1, value_of_out_2, value_of_out_3...] input file widths
    nrgoff:     (int list)  [value_of_out_1, value_of_out_2, value_of_out_3...] input file range offsets
    nazoff:     (int list)  [value_of_out_1, value_of_out_2, value_of_out_3...] input file azimuth offsets
    phc:        (float list) [value_of_out_1, value_of_out_2, value_of_out_3...] input file compensation phase
    oflag:      (int list)  [value_of_out_1, value_of_out_2, value_of_out_3...] overlap area mosaicking flag 

    for each frame
    range offset is relative to the first sample of last subswath
    azimuth offset is relative to the uppermost line
    '''

    if type(inputfile) != list:
        raise Exception('inputfile must be a python list.\n')
        if len(inputfile) != n:
            raise Exception('number of input files is not equal to inputfile list length.\n')
    else:
        inputfile = [bytes(x,'utf-8') for x in inputfile]

    filters = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libalos2proc.so'))
    filters.mosaicsubswath(
        ctypes.c_char_p(bytes(outputfile,'utf-8')),
        ctypes.c_int(nrgout),
        ctypes.c_int(nazout),
        ctypes.c_int(delta),
        ctypes.c_int(diffflag),
        ctypes.c_int(n),
        (ctypes.c_char_p * len(inputfile))(*inputfile),
        (ctypes.c_int * len(nrgin))(*nrgin),
        (ctypes.c_int * len(nrgoff))(*nrgoff),
        (ctypes.c_int * len(nazoff))(*nazoff),
        (ctypes.c_float * len(phc))(*phc),
        (ctypes.c_int * len(oflag))(*oflag)
        )


def look(inputfile, outputfile, nrg, nrlks, nalks, ft=0, sum=0, avg=0):
    '''
    inputfile:  input file
    outputfile: output file
    nrg:     file width
    nrlks:   number of looks in range (default: 4)
    nalks:   number of looks in azimuth (default: 4)
    ft:      file type (default: 0)
               0: signed char
               1: int
               2: float
               3: double
               4: complex (real and imagery: float)
               5: complex (real and imagery: double)
    sum:     sum method (default: 0)
               0: simple sum
               1: power sum (if complex, do this for each channel seperately)
    avg:     take average (default: 0)
               0: no
               1: yes
    '''

    filters = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libalos2proc.so'))
    filters.look(
        ctypes.c_char_p(bytes(inputfile,'utf-8')),
        ctypes.c_char_p(bytes(outputfile,'utf-8')),
        ctypes.c_long(nrg),
        ctypes.c_int(nrlks),
        ctypes.c_int(nalks),
        ctypes.c_int(ft),
        ctypes.c_int(sum),
        ctypes.c_int(avg)
        )


def extract_burst(inputf, outputf, prf, prf_frac, nb, nbg, bsl, kacoeff, dopcoeff, az_ratio, min_line_offset):
    '''
    see extract_burst.c for usage
    '''

    img = isceobj.createSlcImage()
    img.load(inputf + '.xml')

    width = img.getWidth()
    length = img.getLength()

    inputimage = find_vrt_file(inputf+'.vrt', 'SourceFilename')
    byteorder = find_vrt_keyword(inputf+'.vrt', 'ByteOrder')
    if byteorder == 'LSB':
        byteorder = 0
    else:
        byteorder = 1
    imageoffset = find_vrt_keyword(inputf+'.vrt', 'ImageOffset')
    imageoffset = int(imageoffset)
    lineoffset = find_vrt_keyword(inputf+'.vrt', 'LineOffset')
    lineoffset = int(lineoffset)

    if type(kacoeff) != list:
        raise Exception('kacoeff must be a python list.\n')
        if len(kacoeff) != 4:
            raise Exception('kacoeff must have four elements.\n')
    if type(dopcoeff) != list:
        raise Exception('dopcoeff must be a python list.\n')
        if len(dopcoeff) != 4:
            raise Exception('dopcoeff must have four elements.\n')

    filters = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libalos2proc.so'))
    filters.extract_burst(
        ctypes.c_char_p(bytes(inputimage,'utf-8')),
        ctypes.c_char_p(bytes(outputf,'utf-8')),
        ctypes.c_int(width),
        ctypes.c_int(length),
        ctypes.c_float(prf),
        ctypes.c_float(prf_frac),
        ctypes.c_float(nb),
        ctypes.c_float(nbg),
        ctypes.c_float(bsl),
        (ctypes.c_float * len(kacoeff))(*kacoeff),
        (ctypes.c_float * len(dopcoeff))(*dopcoeff),
        ctypes.c_float(az_ratio),
        ctypes.c_float(min_line_offset),
        ctypes.c_int(byteorder),
        ctypes.c_long(imageoffset),
        ctypes.c_long(lineoffset)
        )

    #img = isceobj.createSlcImage()
    #img.load(inputfile + '.xml')
    #img.setFilename(outputfile)
    #img.extraFilename = outputfile + '.vrt'
    #img.setAccessMode('READ')
    #img.renderHdr()


def find_vrt_keyword(xmlfile, keyword):
    from xml.etree.ElementTree import ElementTree

    value = None
    xmlx = ElementTree(file=open(xmlfile,'r')).getroot()
    #try 10 times
    for i in range(10):
        path=''
        for j in range(i):
            path += '*/'
        value0 = xmlx.find(path+keyword)
        if value0 != None:
            value = value0.text
            break

    return value



def find_vrt_file(xmlfile, keyword, relative_path=True):
    '''
    find file in vrt in another directory
    xmlfile: vrt file
    relative_path: True: return relative (to current directory) path of the file
                   False: return absolute path of the file
    '''
    import os
    #get absolute directory of xmlfile
    xmlfile_dir = os.path.dirname(os.path.abspath(xmlfile))
    #find source file path
    file = find_vrt_keyword(xmlfile, keyword)
    #get absolute path of source file
    file = os.path.abspath(os.path.join(xmlfile_dir, file))
    #get relative path of source file
    if relative_path:
        file = os.path.relpath(file, './')
    return file








