#!/usr/bin/env python3

import numpy as np
import argparse
import os
import isce
import isceobj
import glob
import s1a_isce_utils as ut

def createParser():
    parser = argparse.ArgumentParser( description='Estimate azimuth misregistration using overlap ifgs')
    parser.add_argument('-i', '--overlap_dir', type=str, dest='esdDirname', default='overlap',
            help='Directory with the combined overlap interferograms')
    parser.add_argument('-o', '--out_azimuth', type=str, dest='output', default='misreg.txt',
            help='Textfile with the constant azimuth offset')
    parser.add_argument('-t', '--coh_threshold', type=float, dest='esdCoherenceThreshold', default=0.95,
            help='Coherence threshold for overlap masking')

    parser.add_argument('-a', '--azimuth_looks', type=str, dest='esdAzimuthLooks', default=5,
            help='Azimuth looks')
    parser.add_argument('-r', '--range_looks', type=str, dest='esdRangeLooks', default=15,
            help='Range looks')

    return parser

def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''
    parser = createParser()
    return parser.parse_args(args=iargs)

def main(iargs=None):

    inps = cmdLineParse(iargs)

    '''
    Estimate azimuth misregistration.
    '''

    #catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    #master = self._insar.loadProduct( self._insar.masterSlcProduct + '.xml' )

    #minBurst, maxBurst = self._insar.commonMasterBurstLimits
    #slaveBurstStart, slaveBurstEnd = self._insar.commonSlaveBurstLimits

    esdPath = inps.esdDirname
    swathList = ut.getSwathList(esdPath)


    alks = inps.esdAzimuthLooks
    rlks = inps.esdRangeLooks

    #esdPath = esdPath.split()
    val = []
    #for esddir in esdPath:
    for swath in swathList:
        esddir = os.path.join(esdPath, 'IW{0}'.format(swath))
        freqFiles = glob.glob(os.path.join(esddir,'freq_??.bin'))
        freqFiles.sort()

        minBurst = int(os.path.basename(freqFiles[0]).split('.')[0][-2:])
        maxBurst = int(os.path.basename(freqFiles[-1]).split('.')[0][-2:])

    #maxBurst = maxBurst - 1

        combIntName = os.path.join(esddir, 'combined.int')
        combFreqName = os.path.join(esddir, 'combined_freq.bin')
        combCorName = os.path.join(esddir, 'combined.cor')
        combOffName = os.path.join(esddir, 'combined.off')


        for ff in [combIntName, combFreqName, combCorName, combOffName]:
           if os.path.exists(ff):
              os.remove(ff)

  #  val = []
        lineCount = 0
        for ii in range(minBurst, maxBurst):
          intname = os.path.join(esddir, 'overlap_%02d.%dalks_%drlks.int'%(ii+1, alks,rlks))
          freqname = os.path.join(esddir, 'freq_%02d.%dalks_%drlks.bin'%(ii+1,alks,rlks))
          corname = os.path.join(esddir, 'overlap_%02d.%dalks_%drlks.cor'%(ii+1, alks, rlks))


          img = isceobj.createImage()
          img.load(intname + '.xml')
          width = img.getWidth()
          length = img.getLength()

          ifg = np.fromfile(intname, dtype=np.complex64).reshape((-1,width))
          freq = np.fromfile(freqname, dtype=np.float32).reshape((-1,width))
          cor = np.fromfile(corname, dtype=np.float32).reshape((-1,width))

          with open(combIntName, 'ab') as fid:
              ifg.tofile(fid)

          with open(combFreqName, 'ab') as fid:
              freq.tofile(fid)

          with open(combCorName, 'ab') as fid:
              cor.tofile(fid)

          off = np.angle(ifg) / freq

          with open(combOffName, 'ab') as fid:
              off.astype(np.float32).tofile(fid)

          lineCount += length


          mask = (np.abs(ifg) > 0) * (cor > inps.esdCoherenceThreshold)

          vali = off[mask]
          val = np.hstack((val, vali))



        img = isceobj.createIntImage()
        img.filename = combIntName
        img.setWidth(width)
        img.setAccessMode('READ')
        img.renderHdr()

        for fname in [combFreqName, combCorName, combOffName]:
          img = isceobj.createImage()
          img.bands = 1
          img.scheme = 'BIP'
          img.dataType = 'FLOAT'
          img.filename = fname
          img.setWidth(width)
          img.setAccessMode('READ')
          img.renderHdr()

    if val.size == 0 :
        raise Exception('Coherence threshold too strict. No points left for reliable ESD estimate')

    medianval = np.median(val)
    meanval = np.mean(val)
    stdval = np.std(val)

    hist, bins = np.histogram(val, 50, normed=1)
    center = 0.5*(bins[:-1] + bins[1:])


    debugplot = True
    try:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
    except:
        print('Matplotlib could not be imported. Skipping debug plot...')
        debugplot = False

    if debugplot:
        ####Plotting
        plt.figure()
        plt.bar(center, hist, align='center', width = 0.7*(bins[1] - bins[0]))
        plt.xlabel('Azimuth shift in pixels')
        plt.savefig( os.path.join(esddir, 'ESDmisregistration.png'))
        plt.close()


#    catalog.addItem('Median', medianval, 'esd')
#    catalog.addItem('Mean', meanval, 'esd')
#    catalog.addItem('Std', stdval, 'esd')
#    catalog.addItem('coherence threshold', self.esdCoherenceThreshold, 'esd')
#    catalog.addItem('number of coherent points', val.size, 'esd')

#    catalog.printToLog(logger, "runESD")
#    self._insar.procDoc.addAllFromCatalog(catalog)

#    slaveTimingCorrection = medianval * master.bursts[0].azimuthTimeInterval

    outputDir = os.path.dirname(inps.output)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    with open(inps.output, 'w') as f:
         f.write('median : '+str(medianval) +'\n')
         f.write('mean : '+str(meanval)+'\n')
         f.write('std : '+str(stdval)+'\n')
         f.write('coherence threshold : '+str(inps.esdCoherenceThreshold)+'\n')
         f.write('mumber of coherent points : '+str(len(val))+'\n')


if __name__ == '__main__':
    '''
    The main driver.
    '''

    main()    





