#
# Cunren Liang, 03-MAY-2018
# California Institute of Technology
#
# optimal burst merging program, with some functions adapted from Piyush's original merging program
# 1. adjust the position of the first valid line, last valid line, first valid sample and last valid sample,
#    so that in the subsequent multiple looking process only samples from the same burst fall in a multilooing
#    window.
# 2. do ionospheric correction.
# 3. modularize the procedures so that it is easier to add codes for merging additional types of bursts
#


import copy
import numpy as np 
import os
import isceobj
import datetime
import logging
from isceobj.Util.ImageUtil import ImageLib as IML


def mergeBox(frame):
    '''
    Merging using VRTs.
    '''
    
    from .VRTManager import Swath, VRTConstructor


    swaths = [Swath(x) for x in frame]

    ###Identify the 4 corners and dimensions
    topSwath = min(swaths, key = lambda x: x.sensingStart)
    botSwath = max(swaths, key = lambda x: x.sensingStop)
    leftSwath = min(swaths, key = lambda x: x.nearRange)
    rightSwath = max(swaths, key = lambda x: x.farRange)


    totalWidth = int( np.round((rightSwath.farRange - leftSwath.nearRange)/leftSwath.dr + 1))
    totalLength = int(np.round((botSwath.sensingStop - topSwath.sensingStart).total_seconds()/topSwath.dt + 1 ))

    sensingStart = topSwath.sensingStart
    nearRange = leftSwath.nearRange

    dt = topSwath.dt
    dr = leftSwath.dr

    #          box[0]      box[1]       box[2]      box[3]    box[4]    box[5]
    return [totalLength, totalWidth, sensingStart, nearRange,   dt,       dr]


def adjustValidWithLooks(swaths, box, nalks, nrlks, edge=0, avalid='strict', rvalid='strict'):
    '''
    Cunren Liang, January 2018
    adjust the position of the first valid line, last valid line, first valid sample and last valid sample,
    so that in the subsequent multiple looking process only samples from the same burst fall in a multilooing
    window.

    INPUT:
    swaths: frames
    box[0](length): length of the merged image
    box[1](width): width of merged image
    box[2](sensingStart): sensingStart of merged image
    box[3](nearRange): nearRange of merged image
    box[4](dt): timeSpacing of merged image
    box[5](dr): rangeSpacing of merged image
    nalks: number of azimuth looks to be taken
    nrlks: number of range looks to be taken
    edge: edges around valid samples to be removed

    in multiple looking
    avalid: There are three possible values:
            'strict':          avalid = nalks, this strictly follows azimuth number of looks, to make sure each 
                               resulting pixel contains exactly azimuth number of looks pixels.
            'adaptive':        this tries avalid values starting from nalks, to make sure there are no gaps
                               between bursts
            1<=avalid<=nalks:  specifying an avalid value (integer)
                               
            for all of the three cases, if there are >=avalid pixels used to do multiple looking on the upper/lower
            edge, the resulting line is considered to be valid.
            for all of teh three cases, 1<=avalid<=nalks
    
    rvalid: the same thing in range.

    RESULT OF THIS FUNCTION: the following are changed:
        swaths[i].bursts[j].firstValidLine
        swaths[i].bursts[j].firstValidSample
        swaths[i].bursts[j].numValidLines
        swaths[i].bursts[j].numValidSamples
    
    WARNING: the overlap area (valid line and valid sample) between two adjacent bursts/subswaths should
    (supposing that avalid=nalks, rvalid=nrlks)
    number of overlap valid lines - 2*edge >= 2 * nalks
    number of overlap valid samples - 2*edge >= 2 * nrlks
    otherwise, there may be blank lines or columns between adjacent bursts/subswaths.
    normally the overlap area between ajacent bursts is larger than 100 lines, and the overlap area between
    adjacent subswaths is larger than 600 samples. Therefore, this should at least support 50(az) * 300(rg)
    looks, which is enough for most applications.

    a burst of a subswath usually overlaps with two bursts of the adjacent subswaths. The two bursts may have
    different starting ranges (difference is usually about 150 samples). Even if this is the case, there are
    still more than 400 samples left, which should at least support 200 (rg) looks.
    '''

    if avalid == 'strict':
        avalidList = [nalks]
    elif avalid == 'adaptive':
        avalidList = list(range(1, nalks+1))
        avalidList.reverse()
    else:
        avalidList = [np.int(np.around(avalid))]
    
    avalidnum = len(avalidList)
    for i in range(avalidnum):
        if not (1<=avalidList[i]<=nalks):
            raise Exception('wrong avalid: {}'.format(avalidList[i]))

    if rvalid == 'strict':
        rvalidList = [nrlks]
    elif rvalid == 'adaptive':
        rvalidList = list(range(1, nrlks+1))
        rvalidList.reverse()
    else:
        rvalidList = [np.int(np.around(rvalid))]

    rvalidnum = len(rvalidList)
    for i in range(rvalidnum):
        if not (1<=rvalidList[i]<=nrlks):
            raise Exception('wrong rvalid: {}'.format(rvalidList[i]))

    length = box[0]
    width = box[1]
    sensingStart = box[2]
    nearRange = box[3]
    dt = box[4]
    dr = box[5]

    nswath = len(swaths)
    #remove edge
    for i in range(nswath):
        nburst = len(swaths[i].bursts)
        for j in range(nburst):
            swaths[i].bursts[j].firstValidLine += edge
            swaths[i].bursts[j].firstValidSample += edge
            swaths[i].bursts[j].numValidLines -= (2*edge)
            swaths[i].bursts[j].numValidSamples -= (2*edge)

            #index starts from 1
            firstline = swaths[i].bursts[j].firstValidLine + 1
            lastline = firstline + swaths[i].bursts[j].numValidLines - 1
            firstcolumn = swaths[i].bursts[j].firstValidSample + 1
            lastcolumn = firstcolumn + swaths[i].bursts[j].numValidSamples - 1
            
            if not(1 <= firstline <= swaths[i].bursts[j].numberOfLines and \
                1 <= lastline <= swaths[i].bursts[j].numberOfLines and \
                1 <= firstcolumn <= swaths[i].bursts[j].numberOfSamples and \
                1 <= lastcolumn <= swaths[i].bursts[j].numberOfSamples and \
                lastline - firstline >= 500 and \
                lastcolumn - firstcolumn >= 500):
                raise Exception('edge too large: {}'.format(edge))

    #contains first line, last line, first column, last column of each burst of each subswath
    #index in merged image, index starts from 1
    burstValidBox = []
    #index in burst, index starts from 1
    burstValidBox2 = []
    #index in burst, burst.firstValidLine, burst.numValidLines, burst.firstValidSample, burst.numValidSamples
    burstValidBox3 = []
    for i in range(nswath):
        burstValidBox.append([])
        burstValidBox2.append([])
        burstValidBox3.append([])
        nburst = len(swaths[i].bursts)
        for j in range(nburst):
            burstValidBox[i].append([0, 0, 0, 0])
            burstValidBox2[i].append([0, 0, 0, 0])
            burstValidBox3[i].append([0, 0, 0, 0])

    #adjust lines
    for ii in range(avalidnum):

        #temporary representation of burstValidBox
        burstValidBox_tmp = []
        #temporary representation of burstValidBox2
        burstValidBox2_tmp = []
        #temporary representation of burstValidBox3
        burstValidBox3_tmp = []
        for i in range(nswath):
            burstValidBox_tmp.append([])
            burstValidBox2_tmp.append([])
            burstValidBox3_tmp.append([])
            nburst = len(swaths[i].bursts)
            for j in range(nburst):
                burstValidBox_tmp[i].append([0, 0, 0, 0])
                burstValidBox2_tmp[i].append([0, 0, 0, 0])
                burstValidBox3_tmp[i].append([0, 0, 0, 0])

        messageAzimuth = ''
        for i in range(nswath):
            nburst = len(swaths[i].bursts)
            for j in range(nburst):

                #offsample = np.int(np.round( (swaths[i].bursts[j].startingRange - nearRange)/dr))
                offline = np.int(np.round( (swaths[i].bursts[j].sensingStart - sensingStart).total_seconds() / dt))

                #index in burst, index starts from 1
                firstline = swaths[i].bursts[j].firstValidLine + 1
                lastline = firstline + swaths[i].bursts[j].numValidLines - 1

                #index in merged image, index starts from 1
                #lines before first line
                #tmp = divmod((firstline + offline - 1), nalks)
                #firstlineAdj = (tmp[0] + (tmp[1]!=0)) * nalks + 1
                #tmp = divmod(lastline + offline, nalks)
                #lastlineAdj = tmp[0] * nalks

                tmp = divmod((firstline + offline - 1), nalks)
                firstlineAdj = (tmp[0] + ((nalks-tmp[1])<avalidList[ii])) * nalks + 1
                tmp = divmod(lastline + offline, nalks)
                lastlineAdj = (tmp[0] + (tmp[1]>=avalidList[ii])) * nalks

                #merge at last line of last burst
                if j != 0:
                    if burstValidBox_tmp[i][j-1][1] - firstlineAdj < -1:
                        messageAzimuth += 'WARNING: no overlap between burst %3d and burst %3d in subswath %3d\n'%(swaths[i].bursts[j-1].burstNumber, swaths[i].bursts[j].burstNumber, swaths[i].bursts[j].swathNumber)
                        messageAzimuth += '         please consider using smaller number of looks in azimuth\n'
                    else:
                        firstlineAdj = burstValidBox_tmp[i][j-1][1] + 1

                burstValidBox_tmp[i][j][0] = firstlineAdj
                burstValidBox_tmp[i][j][1] = lastlineAdj

                burstValidBox2_tmp[i][j][0] = firstlineAdj - offline
                burstValidBox2_tmp[i][j][1] = lastlineAdj - offline

                #index in burst, index starts from 0
                #consistent with def addBurst() in VRTManager.py and isce/components/isceobj/Sensor/TOPS/Sentinel1.py
                burstValidBox3_tmp[i][j][0] = firstlineAdj - offline -1
                burstValidBox3_tmp[i][j][1] = lastlineAdj - firstlineAdj + 1

        if messageAzimuth == '':
            break

    if messageAzimuth != '':
        print(messageAzimuth+'\n')

    for i in range(nswath):
        nburst = len(swaths[i].bursts)
        for j in range(nburst):
            burstValidBox[i][j][0] = burstValidBox_tmp[i][j][0]
            burstValidBox[i][j][1] = burstValidBox_tmp[i][j][1]
            
            burstValidBox2[i][j][0] = burstValidBox2_tmp[i][j][0]
            burstValidBox2[i][j][1] = burstValidBox2_tmp[i][j][1]
            
            burstValidBox3[i][j][0] = burstValidBox3_tmp[i][j][0]
            burstValidBox3[i][j][1] = burstValidBox3_tmp[i][j][1]
            
            #also change swaths
            swaths[i].bursts[j].firstValidLine = burstValidBox3_tmp[i][j][0]
            swaths[i].bursts[j].numValidLines = burstValidBox3_tmp[i][j][1]

##########################################################################################################################
    # #adjust columns
    # for i in range(nswath):
    #     nburst = len(swaths[i].bursts)

    #     #find index in merged image, index starts from 1
    #     firstcolumn0 = []
    #     lastcolumn0 = []       
    #     for j in range(nburst):
    #         offsample = np.int(np.round( (swaths[i].bursts[j].startingRange - nearRange)/dr))
    #         #index in merged image, index starts from 1
    #         firstcolumn0.append(swaths[i].bursts[j].firstValidSample + 1 + offsample)
    #         lastcolumn0.append(firstcolumn + swaths[i].bursts[j].numValidSamples - 1 + offsample)

    #     #index in merged image, index starts from 1
    #     tmp = divmod(max(firstcolumn0) - 1, nrlks)
    #     firstcolumnAdj = (tmp[0] + (tmp[1]!=0)) * nrlks + 1
    #     tmp = divmod(min(lastcolumn0), nrlks)
    #     lastcolumnAdj = tmp[0] * nrlks

    #     #merge at last column of last subswath
    #     if i != 0:
    #         #here use the lastcolumnAdj of the first (can be any, since they are the same) burst of last subswath
    #         if burstValidBox[i-1][0][3] - firstcolumnAdj  < -1:
    #             print('WARNING: no overlap between subswath %3d and subswath %3d'%(i-1, i))
    #             print('         please consider using smaller number of looks in range')
    #         else:
    #             firstcolumnAdj = burstValidBox[i-1][0][3] + 1

    #     #index in burst, index starts from 0
    #     for j in range(nburst):
    #         offsample = np.int(np.round( (swaths[i].bursts[j].startingRange - nearRange)/dr))

    #         swaths[i].bursts[j].firstValidSample = firstcolumnAdj - offsample - 1
    #         swaths[i].bursts[j].numValidSamples = lastcolumnAdj - firstcolumnAdj + 1

    #         burstValidBox[i][j] += [firstcolumnAdj, lastcolumnAdj]
    #         burstValidBox2[i][j] += [firstcolumnAdj - offsample, lastcolumnAdj - offsample]
##########################################################################################################################


    #adjust columns
    for ii in range(rvalidnum):

        #temporary representation of burstValidBox
        burstValidBox_tmp = []
        #temporary representation of burstValidBox2
        burstValidBox2_tmp = []
        #temporary representation of burstValidBox3
        burstValidBox3_tmp = []
        for i in range(nswath):
            burstValidBox_tmp.append([])
            burstValidBox2_tmp.append([])
            burstValidBox3_tmp.append([])
            nburst = len(swaths[i].bursts)
            for j in range(nburst):
                burstValidBox_tmp[i].append([0, 0, 0, 0])
                burstValidBox2_tmp[i].append([0, 0, 0, 0])
                burstValidBox3_tmp[i].append([0, 0, 0, 0])

        messageRange = ''
        for i in range(nswath):
            nburst = len(swaths[i].bursts)
            for j in range(nburst):

                offsample = np.int(np.round( (swaths[i].bursts[j].startingRange - nearRange)/dr))

                #index in burst, index starts from 1
                firstcolumn = swaths[i].bursts[j].firstValidSample + 1
                lastcolumn = firstcolumn + swaths[i].bursts[j].numValidSamples - 1

                #index in merged image, index starts from 1
                #columns before first column
                #tmp = divmod((firstcolumn + offsample - 1), nrlks)
                #firstcolumnAdj = (tmp[0] + (tmp[1]!=0)) * nrlks + 1
                #tmp = divmod(lastcolumn + offsample, nrlks)
                #lastcolumnAdj = tmp[0] * nrlks

                tmp = divmod((firstcolumn + offsample - 1), nrlks)
                firstcolumnAdj = (tmp[0] + ((nrlks-tmp[1])<rvalidList[ii])) * nrlks + 1
                tmp = divmod(lastcolumn + offsample, nrlks)
                lastcolumnAdj = (tmp[0] + (tmp[1]>=rvalidList[ii])) * nrlks

                if i != 0:
                    #find overlap burst in the left swath
                    lastcolumnLeftswath = []
                    nburst0 = len(swaths[i-1].bursts)
                    for k in range(nburst0):
                        if list(set(range(burstValidBox[i-1][k][0], burstValidBox[i-1][k][1]+1)) & set(range(burstValidBox[i][j][0], burstValidBox[i][j][1]+1))) != []:
                            lastcolumnLeftswath.append(burstValidBox_tmp[i-1][k][3])

                    #merge at last column of last subswath
                    if lastcolumnLeftswath != []:
                        #here I use minimum last column
                        lastcolumnLeftswath0 = min(lastcolumnLeftswath)
                        if lastcolumnLeftswath0 - firstcolumnAdj < -1:
                            messageRange += 'WARNING: no overlap between subswath %3d and subswath %3d at burst %3d\n'%(swaths[i-1].bursts[0].swathNumber, swaths[i].bursts[j].swathNumber, swaths[i].bursts[j].burstNumber)
                            messageRange += '         please consider using smaller number of looks in range\n'
                        else:
                            firstcolumnAdj = lastcolumnLeftswath0 + 1

                burstValidBox_tmp[i][j][2] = firstcolumnAdj
                burstValidBox_tmp[i][j][3] = lastcolumnAdj

                burstValidBox2_tmp[i][j][2] = firstcolumnAdj - offsample
                burstValidBox2_tmp[i][j][3] = lastcolumnAdj - offsample

                #index in burst, index starts from 0
                #consistent with def addBurst() in VRTManager.py and isce/components/isceobj/Sensor/TOPS/Sentinel1.py
                burstValidBox3_tmp[i][j][2] = firstcolumnAdj - offsample - 1
                burstValidBox3_tmp[i][j][3] = lastcolumnAdj - firstcolumnAdj + 1

        if messageRange == '':
            break

    if messageRange != '':
        print(messageRange+'\n')

    for i in range(nswath):
        nburst = len(swaths[i].bursts)
        for j in range(nburst):
            burstValidBox[i][j][2] = burstValidBox_tmp[i][j][2]
            burstValidBox[i][j][3] = burstValidBox_tmp[i][j][3]
            
            burstValidBox2[i][j][2] = burstValidBox2_tmp[i][j][2]
            burstValidBox2[i][j][3] = burstValidBox2_tmp[i][j][3]
            
            burstValidBox3[i][j][2] = burstValidBox3_tmp[i][j][2]
            burstValidBox3[i][j][3] = burstValidBox3_tmp[i][j][3]
            
            #also change swaths
            swaths[i].bursts[j].firstValidSample = burstValidBox3_tmp[i][j][2]
            swaths[i].bursts[j].numValidSamples = burstValidBox3_tmp[i][j][3]

    #if message != '', there are gaps
    message = messageAzimuth + messageRange

    #print result
    swath0 = max(swaths, key = lambda x: len(x.bursts))
    nburstMax = len(swath0.bursts)
    
    print('\nafter adjustment (index in merged image, index starts from 1): ')
    info = ' burst   '
    for i in range(nswath):
        info += '  fl      ll      fc      lc       '
    info +=  '\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'    
    for j in range(nburstMax):
        
        info += '%4d    '%(j+1)
        for i in range(nswath):
            if j in range(len(swaths[i].bursts)):
                info += '%7d %7d %7d %7d    '%(burstValidBox[i][j][0], burstValidBox[i][j][1], burstValidBox[i][j][2], burstValidBox[i][j][3])
        info += '\n'
    print(info)
##########################################################################################################################

    tmp = (burstValidBox, burstValidBox2, message)

    return tmp


def mergeBurstsVirtual(frame, fileList, outbox, outfile, validOnly=True):
    '''
    Merging using VRTs.
    '''
    
    from .VRTManager import Swath, VRTConstructor


    swaths = [Swath(x) for x in frame]


    ###Determine number of bands and type
    img  = isceobj.createImage()
    img.load( fileList[0][0] + '.xml')
    bands = img.bands 
    dtype = img.dataType
    img.filename = outfile


    #####Start the builder
    ###Now start building the VRT and then render it
    builder = VRTConstructor(outbox[0], outbox[1])
    builder.setReferenceTime( outbox[2])
    builder.setReferenceRange( outbox[3])
    builder.setTimeSpacing( outbox[4] )
    builder.setRangeSpacing( outbox[5])
    builder.setDataType( dtype.upper())

    builder.initVRT()


    ####Render XML and default VRT. VRT will be overwritten.
    img.width = outbox[1]
    img.length =outbox[0]
    img.renderHdr()


    for bnd in range(1,bands+1):
        builder.initBand(band = bnd)

        for ind, swath in enumerate(swaths):
            ####Relative path
            relfilelist = [os.path.relpath(x, 
                os.path.dirname(outfile))  for x in fileList[ind]]

            builder.addSwath(swath, relfilelist, band=bnd, validOnly=validOnly)

        builder.finishBand()
    builder.finishVRT()

    with open(outfile + '.vrt', 'w') as fid:
        fid.write(builder.vrt)



def mergeBursts(frame, fileList, outfile,
        method='top'):
    '''
    Merge burst products into single file.
    Simple numpy based stitching
    '''

    ###Check against metadata
    if frame.numberOfBursts != len(fileList):
        print('Warning : Number of burst products does not appear to match number of bursts in metadata')


    t0 = frame.bursts[0].sensingStart
    dt = frame.bursts[0].azimuthTimeInterval
    width = frame.bursts[0].numberOfSamples

    #######
    tstart = frame.bursts[0].sensingStart 
    tend = frame.bursts[-1].sensingStop
    nLines = int( np.round((tend - tstart).total_seconds() / dt)) + 1
    print('Expected total nLines: ', nLines)


    img = isceobj.createImage()
    img.load( fileList[0] + '.xml')
    bands = img.bands
    scheme = img.scheme
    npType = IML.NUMPY_type(img.dataType)

    azReferenceOff = []
    for index in range(frame.numberOfBursts):
        burst = frame.bursts[index]
        soff = burst.sensingStart + datetime.timedelta(seconds = (burst.firstValidLine*dt)) 
        start = int(np.round((soff - tstart).total_seconds() / dt))
        end = start + burst.numValidLines

        azReferenceOff.append([start,end])

        print('Burst: ', index, [start,end])

        if index == 0:
            linecount = start

    outMap = IML.memmap(outfile, mode='write', nchannels=bands,
            nxx=width, nyy=nLines, scheme=scheme, dataType=npType)

    for index in range(frame.numberOfBursts):
        curBurst = frame.bursts[index]
        curLimit = azReferenceOff[index]

        curMap = IML.mmapFromISCE(fileList[index], logging)

        #####If middle burst
        if index > 0:
            topBurst = frame.bursts[index-1]
            topLimit = azReferenceOff[index-1]
            topMap = IML.mmapFromISCE(fileList[index-1], logging)

            olap = topLimit[1] - curLimit[0]

            if olap <= 0:
                raise Exception('No Burst Overlap')


            for bb in range(bands):
                topData =  topMap.bands[bb][topBurst.firstValidLine: topBurst.firstValidLine + topBurst.numValidLines,:]

                curData =  curMap.bands[bb][curBurst.firstValidLine: curBurst.firstValidLine + curBurst.numValidLines,:]

                im1 = topData[-olap:,:]
                im2 = curData[:olap,:]

                if method=='avg':
                    data = 0.5*(im1 + im2)
                elif method == 'top':
                    data = im1
                elif method == 'bot':
                    data = im2
                else:
                    raise Exception('Method should be top/bot/avg')

                outMap.bands[bb][linecount:linecount+olap,:] = data

            tlim = olap
        else:
            tlim = 0

        linecount += tlim
            
        if index != (frame.numberOfBursts-1):
            botBurst = frame.bursts[index+1]
            botLimit = azReferenceOff[index+1]
            
            olap = curLimit[1] - botLimit[0]

            if olap < 0:
                raise Exception('No Burst Overlap')

            blim = botLimit[0] - curLimit[0]
        else:
            blim = curBurst.numValidLines
       
        lineout = blim - tlim
        
        for bb in range(bands):
            curData =  curMap.bands[bb][curBurst.firstValidLine: curBurst.firstValidLine + curBurst.numValidLines,:]
            outMap.bands[bb][linecount:linecount+lineout,:] = curData[tlim:blim,:] 

        linecount += lineout
        curMap = None
        topMap = None

    IML.renderISCEXML(outfile, bands,
            nLines, width,
            img.dataType, scheme)

    oimg = isceobj.createImage()
    oimg.load(outfile + '.xml')
    oimg.imageType = img.imageType
    oimg.renderHdr()
    try:
        outMap.bands[0].base.base.flush()
    except:
        pass


def multilook(infile, outname=None, alks=5, rlks=15):
    '''
    Take looks.
    '''

    from mroipac.looks.Looks import Looks

    print('Multilooking {0} ...'.format(infile))

    inimg = isceobj.createImage()
    inimg.load(infile + '.xml')

    if outname is None:
        spl = os.path.splitext(inimg.filename)
        ext = '.{0}alks_{1}rlks'.format(alks, rlks)
        outname = spl[0] + ext + spl[1]

    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inimg)
    lkObj.setOutputFilename(outname)
    lkObj.looks()

    return outname


def mergeBursts2(frames, bursts, burstIndex, box, outputfile, virtual=True, validOnly=True):
    '''
    frames:     a list of objects loaded from subswath IW*.xml files
    bursts:     burst file name with wild card
    burstIndex: swath and burst indexes
    box:        the entire merging region
    outputfile: output file
    virtual:    whether use virtual file
    '''

    burstList = []
    for [swath, minBurst, maxBurst] in burstIndex:
        burstList.append([bursts%(swath,x+1) for x in range(minBurst, maxBurst)])

    if (virtual == False) and (len(frames) == 1):
        mergeBursts(frames[0], burstList[0], outputfile)
    else:
        if (virtual == False):
            print('User requested for multi-swath stitching.')
            print('Virtual files are the only option for this.')
            print('Proceeding with virtual files.')
        mergeBurstsVirtual(frames, burstList, box, outputfile, validOnly=validOnly)


def runMergeBursts(self, adjust=1):
    '''
    Merge burst products to make it look like stripmap.
    Currently will merge interferogram, lat, lon, z and los.
    '''
    from isceobj.TopsProc.runIon import renameFile
    from isceobj.TopsProc.runIon import runCmd

    #########################################
    #totalLooksThreshold = 9
    totalLooksThreshold = 99999999999999
    #if doing ionospheric correction
    ionCorrection = self.ION_doIon
    ionDirname = 'ion/ion_burst'
    mergedIonname = 'topophase.ion'
    originalIfgname = 'topophase_ori.flat'
    #########################################

    # backing out the tigher constraints for ionosphere as it could itnroduce gabs between along track products produced seperately  
    if not ionCorrection:
        adjust=0

    #########################################
    # STEP 1. SET UP
    #########################################
    virtual = self.useVirtualFiles

    #get frames (subswaths)
    frames=[]
    burstIndex = []
    swathList = self._insar.getValidSwathList(self.swaths)
    for swath in swathList:
        minBurst, maxBurst = self._insar.commonReferenceBurstLimits(swath-1)
        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue
        ifg = self._insar.loadProduct( os.path.join(self._insar.fineIfgDirname, 'IW{0}.xml'.format(swath)))
        frames.append(ifg)
        burstIndex.append([int(swath), minBurst, maxBurst])

    #adjust valid samples
    validOnly=False
    #determine merged size
    box = mergeBox(frames)
    if adjust == 1:
        #make a copy before doing this
        frames_bak = copy.deepcopy(frames)
        #adjust valid with looks, 'frames' ARE CHANGED AFTER RUNNING THIS
        (burstValidBox, burstValidBox2, message) = adjustValidWithLooks(frames, box, self.numberAzimuthLooks, self.numberRangeLooks, edge=0, avalid='strict', rvalid='strict')
        if message != '':
            print('***********************************************************************************')
            print('doing ajustment with looks results in gaps beween bursts/swaths.')
            print('no ajustment made.')
            print('This means a multi-look pixel may contains one-look pixels from different bursts.')
            print('***********************************************************************************')
            #restore
            frames = frames_bak
        else:
            validOnly==True


    #########################################
    # STEP 2. MERGE BURSTS
    #########################################
    mergedir = self._insar.mergedDirname
    os.makedirs(mergedir, exist_ok=True)
    if (self.numberRangeLooks == 1) and (self.numberAzimuthLooks==1):
        suffix = ''
    else:
        suffix = '.full'

    #merge los, lat, lon, z
    mergeBursts2(frames, os.path.join(self._insar.geometryDirname, 'IW%d', 'los_%02d.rdr'), burstIndex, box, os.path.join(mergedir, self._insar.mergedLosName+suffix), virtual=virtual, validOnly=validOnly)
    mergeBursts2(frames, os.path.join(self._insar.geometryDirname, 'IW%d', 'lat_%02d.rdr'), burstIndex, box, os.path.join(mergedir, 'lat.rdr'+suffix), virtual=virtual, validOnly=validOnly)
    mergeBursts2(frames, os.path.join(self._insar.geometryDirname, 'IW%d', 'lon_%02d.rdr'), burstIndex, box, os.path.join(mergedir, 'lon.rdr'+suffix), virtual=virtual, validOnly=validOnly)
    mergeBursts2(frames, os.path.join(self._insar.geometryDirname, 'IW%d', 'hgt_%02d.rdr'), burstIndex, box, os.path.join(mergedir, 'z.rdr'+suffix), virtual=virtual, validOnly=validOnly)
    #merge reference and coregistered secondary slcs
    mergeBursts2(frames, os.path.join(self._insar.referenceSlcProduct, 'IW%d', 'burst_%02d.slc'), burstIndex, box, os.path.join(mergedir, 'reference.slc'+suffix), virtual=virtual, validOnly=True)
    mergeBursts2(frames, os.path.join(self._insar.fineCoregDirname, 'IW%d', 'burst_%02d.slc'), burstIndex, box, os.path.join(mergedir, 'secondary.slc'+suffix), virtual=virtual, validOnly=True)
    #merge insar products
    if self.doInSAR:
        mergeBursts2(frames, os.path.join(self._insar.fineIfgDirname, 'IW%d',  'burst_%02d.int'), burstIndex, box, os.path.join(mergedir, self._insar.mergedIfgname+suffix), virtual=virtual, validOnly=True)
        if self.numberAzimuthLooks * self.numberRangeLooks < totalLooksThreshold:
            mergeBursts2(frames, os.path.join(self._insar.fineIfgDirname, 'IW%d',  'burst_%02d.cor'), burstIndex, box, os.path.join(mergedir, self._insar.correlationFilename+suffix), virtual=virtual, validOnly=True)
        if ionCorrection == True:
            mergeBursts2(frames, os.path.join(ionDirname, 'IW%d',  'burst_%02d.ion'), burstIndex, box, os.path.join(mergedir, mergedIonname+suffix), virtual=virtual, validOnly=True)


    #########################################
    # STEP 3. MULTIPLE LOOKING MERGED IMAGES
    #########################################
    if suffix not in ['',None]:
        if self.doInSAR:
            multilook(os.path.join(mergedir, self._insar.mergedIfgname+suffix),
              outname = os.path.join(mergedir, self._insar.mergedIfgname),
              alks = self.numberAzimuthLooks, rlks=self.numberRangeLooks)

            multilook(os.path.join(mergedir, self._insar.mergedLosName+suffix),
              outname = os.path.join(mergedir, self._insar.mergedLosName),
              alks = self.numberAzimuthLooks, rlks=self.numberRangeLooks)

            if self.numberAzimuthLooks * self.numberRangeLooks < totalLooksThreshold:
                multilook(os.path.join(mergedir, self._insar.correlationFilename+suffix),
                  outname = os.path.join(mergedir, self._insar.correlationFilename),
                  alks = self.numberAzimuthLooks, rlks=self.numberRangeLooks)
            else:
                #compute coherence
                cmd = "gdal_translate -of ENVI {} {}".format(os.path.join(mergedir, 'reference.slc'+suffix+'.vrt'), os.path.join(mergedir, 'reference.slc'+suffix))
                runCmd(cmd)
                cmd = "gdal_translate -of ENVI {} {}".format(os.path.join(mergedir, 'secondary.slc'+suffix+'.vrt'), os.path.join(mergedir, 'secondary.slc'+suffix))
                runCmd(cmd)
                pwrfile = 'pwr.bil'
                cmd = "imageMath.py -e='real(a)*real(a)+imag(a)*imag(a);real(b)*real(b)+imag(b)*imag(b)' --a={} --b={} -o {} -t float -s BIL".format(os.path.join(mergedir, 'reference.slc'+suffix), os.path.join(mergedir, 'secondary.slc'+suffix), os.path.join(mergedir, pwrfile+suffix))
                runCmd(cmd)
                cmd = "looks.py -i {} -o {} -r {} -a {}".format(os.path.join(mergedir, pwrfile+suffix), os.path.join(mergedir, pwrfile), self.numberRangeLooks, self.numberAzimuthLooks)
                runCmd(cmd)
                cmd = "imageMath.py -e='((abs(a))!=0)*((b_0*b_1)!=0)*sqrt(b_0*b_1);((abs(a))!=0)*((b_0*b_1)!=0)*abs(a)/(sqrt(b_0*b_1)+((b_0*b_1)==0))' --a={} --b={} -o {} -t float -s BIL".format(os.path.join(mergedir, self._insar.mergedIfgname), os.path.join(mergedir, pwrfile), os.path.join(mergedir, self._insar.correlationFilename))
                runCmd(cmd)
                #remove intermediate files
                os.remove(os.path.join(mergedir, 'reference.slc'+suffix))
                os.remove(os.path.join(mergedir, 'secondary.slc'+suffix))
                os.remove(os.path.join(mergedir, pwrfile+suffix))
                os.remove(os.path.join(mergedir, pwrfile+suffix+'.xml'))
                os.remove(os.path.join(mergedir, pwrfile+suffix+'.vrt'))
                os.remove(os.path.join(mergedir, pwrfile))
                os.remove(os.path.join(mergedir, pwrfile+'.xml'))
                os.remove(os.path.join(mergedir, pwrfile+'.vrt'))

            if ionCorrection:
                multilook(os.path.join(mergedir, mergedIonname+suffix),
                  outname = os.path.join(mergedir, mergedIonname),
                  alks = self.numberAzimuthLooks, rlks=self.numberRangeLooks)
    else:
        print('Skipping multi-looking ....')


    #########################################
    # STEP 4. APPLY CORRECTIONS
    #########################################
    #do ionospheric and other corrections here
    #should also consider suffix, but usually we use multiple looks, so I ignore it for now.
    if self.doInSAR:
        if ionCorrection:
            print('user choose to do ionospheric correction')

            #define file names
            interferogramFilename = os.path.join(mergedir, self._insar.mergedIfgname)
            originalInterferogramFilename = os.path.join(mergedir, originalIfgname)
            ionosphereFilename = os.path.join(mergedir, mergedIonname)

            #rename original interferogram to make a backup copy
            if os.path.isfile(originalInterferogramFilename):
                pass
            else:
                renameFile(interferogramFilename, originalInterferogramFilename)
                
            #do correction
            cmd = "imageMath.py -e='a*exp(-1.0*J*b)' --a={} --b={} -o {} -t cfloat".format(originalInterferogramFilename, ionosphereFilename, interferogramFilename)
            runCmd(cmd)


if __name__ == '__main__' :
    '''
    Merge products burst-by-burst.
    '''

    main()
