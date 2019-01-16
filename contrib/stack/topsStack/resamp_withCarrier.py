#!/usr/bin/env python3

import isce
import isceobj
import stdproc
from stdproc.stdproc import crossmul
import numpy as np
from isceobj.Util.Poly2D import Poly2D
import argparse
import os
import copy
import s1a_isce_utils as ut
from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct


def createParser():
    parser = argparse.ArgumentParser( description='Resampling burst by burst SLCs ')

    parser.add_argument('-m', '--master', dest='master', type=str, required=True,
            help='Directory with master acquisition')

    parser.add_argument('-s', '--slave', dest='slave', type=str, required=True,
            help='Directory with slave acquisition')

    parser.add_argument('-o', '--coregdir', dest='coreg', type=str, default='coreg_slave',
            help='Directory with coregistered SLCs and IFGs')

    parser.add_argument('-a', '--azimuth_misreg', dest='misreg_az', type=str, default=0.0,
            help='File name with the azimuth misregistration')

    parser.add_argument('-r', '--range_misreg', dest='misreg_rng', type=str, default=0.0,
            help='File name with the range misregistration')

    parser.add_argument('--noflat', dest='noflat', action='store_true', default=False,
            help='To turn off flattening. False: flattens the SLC. True: turns off flattening.')

    parser.add_argument('-v', '--overlap', dest='overlap', action='store_true', default=False,
            help='Is this an overlap burst slc. default: False')

    parser.add_argument('-d', '--overlapDir', dest='overlapDir', type=str, default='overlap',
            help='master overlap directory')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

def resampSlave(mas, slv, rdict, outname, flatten):
    '''
    Resample burst by burst.
    '''

    azpoly = rdict['azpoly']
    rgpoly = rdict['rgpoly']
    azcarrpoly = rdict['carrPoly']
    dpoly = rdict['doppPoly']

    rngImg = isceobj.createImage()
    rngImg.load(rdict['rangeOff'] + '.xml')
    rngImg.setAccessMode('READ')

    aziImg = isceobj.createImage()
    aziImg.load(rdict['azimuthOff'] + '.xml')
    aziImg.setAccessMode('READ')

    inimg = isceobj.createSlcImage()
    inimg.load(slv.image.filename + '.xml')
    inimg.setAccessMode('READ')


    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = slv.rangePixelSize
    rObj.radarWavelength = slv.radarWavelength
    rObj.azimuthCarrierPoly = azcarrpoly
    rObj.dopplerPoly = dpoly

    rObj.azimuthOffsetsPoly = azpoly
    rObj.rangeOffsetsPoly = rgpoly
    rObj.imageIn = inimg

    width = mas.numberOfSamples
    length = mas.numberOfLines
    imgOut = isceobj.createSlcImage()
    imgOut.setWidth(width)
    imgOut.filename = outname
    imgOut.setAccessMode('write')

    rObj.outputWidth = width
    rObj.outputLines = length
    rObj.residualRangeImage = rngImg
    rObj.residualAzimuthImage = aziImg
    rObj.flatten = flatten
    print(rObj.flatten)
    rObj.resamp_slc(imageOut=imgOut)

    imgOut.renderHdr()
    imgOut.renderVRT()
    return imgOut

def main(iargs=None):
    '''
    Create coregistered overlap slaves.
    '''
    inps = cmdLineParse(iargs)
    masterSwathList = ut.getSwathList(inps.master)
    slaveSwathList = ut.getSwathList(inps.slave)

    swathList = list(sorted(set(masterSwathList+slaveSwathList)))

    for swath in swathList:
    
        ####Load slave metadata
        master = ut.loadProduct( os.path.join(inps.master , 'IW{0}.xml'.format(swath)))
        slave = ut.loadProduct( os.path.join(inps.slave , 'IW{0}.xml'.format(swath)))
        if inps.overlap:
            masterTop = ut.loadProduct(os.path.join(inps.master, inps.overlapDir , 'IW{0}_top.xml'.format(swath)))
            masterBottom = ut.loadProduct(os.path.join(inps.master, inps.overlapDir , 'IW{0}_bottom.xml'.format(swath)))

    
        dt = slave.bursts[0].azimuthTimeInterval
        dr = slave.bursts[0].rangePixelSize

        if os.path.exists(str(inps.misreg_az)):
             with open(inps.misreg_az, 'r') as f:
                misreg_az = float(f.readline())
        else:
             misreg_az = 0.0

        if os.path.exists(str(inps.misreg_rng)):
             with open(inps.misreg_rng, 'r') as f:
                misreg_rg = float(f.readline())
        else:
             misreg_rg = 0.0

        ###Output directory for coregistered SLCs
        if not inps.overlap:
            outdir = os.path.join(inps.coreg,'IW{0}'.format(swath))
            offdir = os.path.join(inps.coreg,'IW{0}'.format(swath)) 
        else:
            outdir = os.path.join(inps.coreg, inps.overlapDir, 'IW{0}'.format(swath))
            offdir = os.path.join(inps.coreg, inps.overlapDir, 'IW{0}'.format(swath))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    
        ####Indices w.r.t master
        burstoffset, minBurst, maxBurst = master.getCommonBurstLimits(slave)
        slaveBurstStart = minBurst +  burstoffset
        slaveBurstEnd = maxBurst
    
        relShifts = ut.getRelativeShifts(master, slave, minBurst, maxBurst, slaveBurstStart)
        if inps.overlap:
            maxBurst = maxBurst - 1 ###For overlaps
    
        print('Shifts: ', relShifts)
    
    ####Can corporate known misregistration here
    
        apoly = Poly2D()
        apoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])
    
        rpoly = Poly2D()
        rpoly.initPoly(rangeOrder=0,azimuthOrder=0,coeffs=[[0.]])

    
        #topCoreg = createTOPSSwathSLCProduct()
        topCoreg = ut.coregSwathSLCProduct()
        topCoreg.configure()

        if inps.overlap:    
            botCoreg = ut.coregSwathSLCProduct()
            botCoreg.configure()

        for ii in range(minBurst, maxBurst):
            jj = slaveBurstStart + ii - minBurst

            if inps.overlap:
                botBurst = masterBottom.bursts[ii]
                topBurst = masterTop.bursts[ii]
            else:
                topBurst = master.bursts[ii]


            slvBurst = slave.bursts[jj]

            #####Top burst processing
            try:
                offset = relShifts[jj]
            except:
                raise Exception('Trying to access shift for slave burst index {0}, which may not overlap with master'.format(jj))
        
            if inps.overlap:         
                outname = os.path.join(outdir, 'burst_top_%02d_%02d.slc'%(ii+1,ii+2))
        
                ####Setup initial polynomials
                ### If no misregs are given, these are zero
                ### If provided, can be used for resampling without running to geo2rdr again for fast results
                rdict = {'azpoly' : apoly,
                     'rgpoly' : rpoly,
                     'rangeOff' : os.path.join(offdir, 'range_top_%02d_%02d.off'%(ii+1,ii+2)),
                     'azimuthOff': os.path.join(offdir, 'azimuth_top_%02d_%02d.off'%(ii+1,ii+2))}

        
            ###For future - should account for azimuth and range misreg here .. ignoring for now. 
                azCarrPoly, dpoly = slave.estimateAzimuthCarrierPolynomials(slvBurst, offset = -1.0 * offset)
        
                rdict['carrPoly'] = azCarrPoly
                rdict['doppPoly'] = dpoly
        
                outimg = resampSlave(topBurst, slvBurst, rdict, outname, (not inps.noflat))
        
                copyBurst = copy.deepcopy(topBurst)
                ut.adjustValidSampleLine(copyBurst)
                copyBurst.image.filename = outimg.filename 
                print('After: ', copyBurst.firstValidLine, copyBurst.numValidLines)
                topCoreg.bursts.append(copyBurst)
            #######################################################

        
                slvBurst = slave.bursts[jj+1]
                outname = os.path.join(outdir, 'burst_bot_%02d_%02d.slc'%(ii+1,ii+2))
        
        ####Setup initial polynomials
        ### If no misregs are given, these are zero
        ### If provided, can be used for resampling without running to geo2rdr again for fast results
                rdict = {'azpoly' : apoly,
                     'rgpoly' : rpoly,
                     'rangeOff' : os.path.join(offdir, 'range_bot_%02d_%02d.off'%(ii+1,ii+2)),
                     'azimuthOff': os.path.join(offdir, 'azimuth_bot_%02d_%02d.off'%(ii+1,ii+2))}
        
                azCarrPoly, dpoly = slave.estimateAzimuthCarrierPolynomials(slvBurst, offset = -1.0 * offset)
        
                rdict['carrPoly'] = azCarrPoly
                rdict['doppPoly'] = dpoly
        
                outimg = resampSlave(botBurst, slvBurst, rdict, outname, (not inps.noflat))
        
                copyBurst = copy.deepcopy(botBurst)
                ut.adjustValidSampleLine(copyBurst)
                copyBurst.image.filename = outimg.filename
                print('After: ', copyBurst.firstValidLine, copyBurst.numValidLines)
                botCoreg.bursts.append(copyBurst)
           #######################################################

            else:

                outname = os.path.join(outdir, 'burst_%02d.slc'%(ii+1))  
        
        ####Setup initial polynomials
        ### If no misregs are given, these are zero
        ### If provided, can be used for resampling without running to geo2rdr again for fast results
                rdict = {'azpoly' : apoly,
                     'rgpoly' : rpoly,
                      'rangeOff' : os.path.join(offdir, 'range_%02d.off'%(ii+1)),
                      'azimuthOff': os.path.join(offdir, 'azimuth_%02d.off'%(ii+1))}
                 

        ###For future - should account for azimuth and range misreg here .. ignoring for now.
                azCarrPoly, dpoly = slave.estimateAzimuthCarrierPolynomials(slvBurst, offset = -1.0 * offset)
        
                rdict['carrPoly'] = azCarrPoly
                rdict['doppPoly'] = dpoly
        
                outimg = resampSlave(topBurst, slvBurst, rdict, outname, (not inps.noflat))
                minAz, maxAz, minRg, maxRg = ut.getValidLines(slvBurst, rdict, outname,
                    misreg_az = misreg_az - offset, misreg_rng = misreg_rg)
                

                copyBurst = copy.deepcopy(topBurst)
                ut.adjustValidSampleLine_V2(copyBurst, slvBurst, minAz=minAz, maxAz=maxAz,
                                         minRng=minRg, maxRng=maxRg)
                copyBurst.image.filename = outimg.filename
                print('After: ', copyBurst.firstValidLine, copyBurst.numValidLines)
                topCoreg.bursts.append(copyBurst)
        #######################################################       

 
        topCoreg.numberOfBursts = len(topCoreg.bursts)
        topCoreg.source = ut.asBaseClass(slave)

        if inps.overlap:
            botCoreg.numberOfBursts = len(botCoreg.bursts)
            topCoreg.reference = ut.asBaseClass(masterTop)
            botCoreg.reference = ut.asBaseClass(masterBottom)
            botCoreg.source = ut.asBaseClass(slave)
            ut.saveProduct(topCoreg, outdir + '_top.xml')
            ut.saveProduct(botCoreg, outdir + '_bottom.xml')

        else:
            topCoreg.reference = master
            ut.saveProduct(topCoreg, outdir + '.xml')    

if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()



