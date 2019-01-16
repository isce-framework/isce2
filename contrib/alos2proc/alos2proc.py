# Cunren Liang
# Copyright 2018, Caltech

import os
import copy
import ctypes
import logging
import isceobj


def mbf(inputfile, outputfile, nrg, prf, prf_frac, nb, nbg, nboff, bsl, kacoeff, dopcoeff1, dopcoeff2):
    #############################
    # inputfile:       input file
    # outputfile:      output file
    # nrg:             file width
    # prf:             PRF
    # prf_frac:        fraction of PRF processed
    #                     (represents azimuth bandwidth)
    # nb:              number of lines in a burst
    #                     (float, in terms of 1/PRF)
    # nbg:             number of lines in a burst gap
    #                     (float, in terms of 1/PRF)
    # nboff:           number of unsynchronized lines in a burst
    #                     (float, in terms of 1/PRF, with sign, see burst_sync.py for rules of sign)
    #                     (the image to be processed is always considered to be master)
    # bsl:             start line number of a burst
    #                     (float, the line number of the first line of the full-aperture SLC is zero)
    #                     (no need to be first burst, any one is OK)

    # kacoeff[0-2]:    FM rate coefficients
    #                     (three coefficients of a quadratic polynomial with regard to)
    #                     (range sample number. range sample number starts with zero)

    # dopcoeff1[0-3]:  Doppler centroid frequency coefficients of this image
    #                     (four coefficients of a third order polynomial with regard to)
    #                     (range sample number. range sample number starts with zero)

    # dopcoeff2[0-3]:  Doppler centroid frequency coefficients of the other image
    #                     (four coefficients of a third order polynomial with regard to)
    #                     (range sample number. range sample number starts with zero)
    #############################

    #examples:
    # kacoeff = [-625.771055784221, 0.007887946763383646, -9.10142814131697e-08]
    # dopcoeff1 = [-0.013424025141940908, -6.820475445542178e-08, 0.0, 0.0]
    # dopcoeff2 = [-0.013408164465406417, -7.216577938502655e-08, 3.187158113584236e-24, -9.081842749918244e-28]

    inputfile2 = copy.deepcopy(inputfile)
    outputfile2 = copy.deepcopy(outputfile)
    inputfile = bytes(inputfile,'utf-8')
    outputfile = bytes(outputfile,'utf-8')
    if type(kacoeff) != list:
        raise Exception('kacoeff must be a python list.\n')
        if len(kacoeff) != 3:
            raise Exception('kacoeff must have three elements.\n')
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
        ctypes.c_char_p(inputfile),
        ctypes.c_char_p(outputfile),
        ctypes.c_int(nrg),
        ctypes.c_float(prf),
        ctypes.c_float(prf_frac),
        ctypes.c_float(nb),
        ctypes.c_float(nbg),
        ctypes.c_float(nboff),
        ctypes.c_float(bsl),
        (ctypes.c_float * len(kacoeff))(*kacoeff),
        (ctypes.c_float * len(dopcoeff1))(*dopcoeff1),
        (ctypes.c_float * len(dopcoeff2))(*dopcoeff2),
        )

    img = isceobj.createSlcImage()
    img.load(inputfile2 + '.xml')
    img.setFilename(outputfile2)
    img.extraFilename = outputfile2 + '.vrt'
    img.setAccessMode('READ')
    img.renderHdr()


def rg_filter(inputfile, nrg, nout, outputfile, bw, bc, nfilter, nfft, beta, zero_cf, offset):
    #############################
    # inputfile:  input file
    # nrg         file width
    # nout:       number of output files
    # outputfile: (value_of_out_1, value_of_out_2, value_of_out_3...) output files
    # bw:         (value_of_out_1, value_of_out_2, value_of_out_3...) filter bandwidth divided by sampling frequency [0, 1]
    # bc:         (value_of_out_1, value_of_out_2, value_of_out_3...) filter center frequency divided by sampling frequency

    # nfilter:    number samples of the filter (odd). Reference Value: 65
    # nfft:       number of samples of the FFT. Reference Value: 1024
    # beta:       kaiser window beta. Reference Value: 1.0
    # zero_cf:    if bc != 0.0, move center frequency to zero? 0: Yes (Reference Value). 1: No.
    # offset:     offset (in samples) of linear phase for moving center frequency. Reference Value: 0.0
    #############################

    #examples
    #outputfile = [bytes('result/crop_filt_1.slc','utf-8'), bytes('result/crop_filt_2.slc','utf-8')]
    #bw = [0.3, 0.3]
    #bc = [0.1, -0.1]

    inputfile2 = copy.deepcopy(inputfile)
    outputfile2 = copy.deepcopy(outputfile)

    inputfile = bytes(inputfile,'utf-8')

    if type(outputfile) != list:
        raise Exception('outputfile must be a python list.\n')
        if len(outputfile) != nout:
            raise Exception('number of output files is not equal to list length.\n')
    else:
        tmp = []
        for x in outputfile:
            tmp.append(bytes(x,'utf-8'))
        outputfile = tmp

    if type(bw) != list:
        raise Exception('bw must be a python list.\n')
        if len(bw) != nout:
            raise Exception('number of output files is not equal to list length.\n')

    if type(bc) != list:
        raise Exception('bc must be a python list.\n')
        if len(bc) != nout:
            raise Exception('number of output files is not equal to list length.\n')

    filters = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libalos2proc.so'))
    filters.rg_filter(
        ctypes.c_char_p(inputfile),
        ctypes.c_int(nrg),
        ctypes.c_int(nout),
        (ctypes.c_char_p * len(outputfile))(*outputfile),
        (ctypes.c_float * len(bw))(*bw),
        (ctypes.c_float * len(bc))(*bc),
        ctypes.c_int(nfilter),
        ctypes.c_int(nfft),
        ctypes.c_float(beta),
        ctypes.c_int(zero_cf),
        ctypes.c_float(offset)
        )

    img = isceobj.createSlcImage()
    img.load(inputfile2 + '.xml')
    for x in outputfile2:
        img.setFilename(x)
        img.extraFilename = x + '.vrt'
        img.setAccessMode('READ')
        img.renderHdr()
