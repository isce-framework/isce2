from coregSwathSLCProduct import coregSwathSLCProduct
import isce
import isceobj
import os
#from isceobj.Sensor.TOPS.coregSwathSLCProduct import coregSwathSLCProduct

class catalog(object):
      def __init__(self):
         pass

      def addItem(self,*args):
          print(' '.join([str(x) for x in args]))
          


def loadProduct(xmlname):
        '''
        Load the product using Product Manager.
        '''

        from iscesys.Component.ProductManager import ProductManager as PM

        pm = PM()
        pm.configure()

        obj = pm.loadProduct(xmlname)

        return obj


def saveProduct( obj, xmlname):
        '''
        Save the product to an XML file using Product Manager.
        '''
       # import shelve
       # import os
       # with shelve.open(os.path.dirname(xmlname) + '/'+ os.path.basename(xmlname)  +'.data') as db:
       #     db['data'] = obj

        from iscesys.Component.ProductManager import ProductManager as PM

        pm = PM()
        pm.configure()

        pm.dumpProduct(obj, xmlname)

        return None

def getRelativeShifts(mFrame, sFrame, minBurst, maxBurst, slaveBurstStart):
    '''
    Estimate the relative shifts between the start of the bursts.
    '''
    import numpy as np    
    azMasterOff = {}
    azSlaveOff = {}
    azRelOff = {}
    tm = mFrame.bursts[minBurst].sensingStart
    dt = mFrame.bursts[minBurst].azimuthTimeInterval
    ts = sFrame.bursts[slaveBurstStart].sensingStart
    
    for index in range(minBurst, maxBurst):
        burst = mFrame.bursts[index]
        azMasterOff[index] = int(np.round((burst.sensingStart - tm).total_seconds() / dt))
        
        burst = sFrame.bursts[slaveBurstStart + index - minBurst]
        azSlaveOff[slaveBurstStart + index - minBurst] =  int(np.round((burst.sensingStart - ts).total_seconds() / dt))
        
        azRelOff[slaveBurstStart + index - minBurst] = azSlaveOff[slaveBurstStart + index - minBurst] - azMasterOff[index]

    
    return azRelOff


def adjustValidSampleLine(master,  minAz=0, maxAz=0, minRng=0, maxRng=0):
    import numpy as np
    import isce
    import isceobj
    # Valid region in the resampled slc based on offsets
    ####Adjust valid samples and first valid sample here
    print ("Adjust valid samples")
    print('Before: ', master.firstValidSample, master.numValidSamples)
    print('Offsets : ', minRng, maxRng)
    if (minRng > 0) and (maxRng > 0):
            master.numValidSamples -= (int(np.ceil(maxRng)) + 8)
            master.firstValidSample += 4
    elif (minRng < 0) and  (maxRng < 0):
            master.firstValidSample -= int(np.floor(minRng) - 4)
            master.numValidSamples += int(np.floor(minRng) - 8)
    elif (minRng < 0) and (maxRng > 0):
            master.firstValidSample -= int(np.floor(minRng) - 4)
            master.numValidSamples += int(np.floor(minRng) - 8) - int(np.ceil(maxRng))

    print('After: ', master.firstValidSample, master.numValidSamples)

    ###Adjust valid lines and first valid line here
    print ("Adjust valid lines")
    print('Before: ', master.firstValidLine, master.numValidLines)
    print('Offsets : ', minAz, maxAz)
    if (minAz > 0) and (maxAz > 0):
            master.numValidLines -= (int(np.ceil(maxAz)) + 8)
            master.firstValidLine += 4
    elif (minAz < 0) and  (maxAz < 0):
            master.firstValidLine -= int(np.floor(minAz) - 4)
            master.numValidLines += int(np.floor(minAz) - 8)
    elif (minAz < 0) and (maxAz > 0):
            master.firstValidLine -= int(np.floor(minAz) - 4)
            master.numValidLines += int(np.floor(minAz) - 8) - int(np.ceil(maxAz))
    print('After:', master.firstValidLine, master.numValidLines)


def adjustValidSampleLine_V2(master, slave, minAz=0, maxAz=0, minRng=0, maxRng=0): 
    import numpy as np
    import isce
    import isceobj
    ####Adjust valid samples and first valid sample here
    print ("Adjust valid samples")
    print('Before: ', master.firstValidSample, master.numValidSamples)
    print('Offsets : ', minRng, maxRng)

    if (minRng > 0) and (minRng > 0):
        master.firstValidSample = slave.firstValidSample - int(np.floor(maxRng)-4)
        lastValidSample = master.firstValidSample - 8 + slave.numValidSamples

        if lastValidSample < master.numberOfSamples:
            master.numValidSamples = slave.numValidSamples - 8
        else:
            master.numValidSamples = master.numberOfSamples - master.firstValidSample

    elif (minRng < 0) and (maxRng < 0):
        master.firstValidSample = slave.firstValidSample - int(np.floor(minRng) - 4)
        lastValidSample = master.firstValidSample + slave.numValidSamples  - 8
        if lastValidSample < master.numberOfSamples:
            master.numValidSamples = slave.numValidSamples - 8
        else:
            master.numValidSamples = master.numberOfSamples - master.firstValidSample
    elif (minRng < 0) and (maxRng > 0):
        master.firstValidSample = slave.firstValidSample - int(np.floor(minRng) - 4)
        lastValidSample = master.firstValidSample + slave.numValidSamples + int(np.floor(minRng) - 8) - int(np.ceil(maxRng))
        if lastValidSample < master.numberOfSamples:
            master.numValidSamples = slave.numValidSamples + int(np.floor(minRng) - 8) - int(np.ceil(maxRng))
        else:
            master.numValidSamples = master.numberOfSamples - master.firstValidSample

    master.firstValidSample = np.max([0, master.firstValidSample])
 
    print('After: ', master.firstValidSample, master.numValidSamples)

    ###Adjust valid lines and first valid line here
    print ("Adjust valid lines")
    print('Before: ', master.firstValidLine, master.numValidLines)
    print('Offsets : ', minAz, maxAz)

    if (minAz > 0) and (maxAz > 0):

            master.firstValidLine = slave.firstValidLine - int(np.floor(maxAz) - 4)
            lastValidLine = master.firstValidLine - 8  + slave.numValidLines
            if lastValidLine < master.numberOfLines:
               master.numValidLines = slave.numValidLines - 8
            else:
               master.numValidLines = master.numberOfLines - master.firstValidLine

    elif (minAz < 0) and  (maxAz < 0):
            master.firstValidLine = slave.firstValidLine - int(np.floor(minAz) - 4)
            lastValidLine = master.firstValidLine + slave.numValidLines + int(np.floor(minAz) - 8)
            lastValidLine = master.firstValidLine + slave.numValidLines  - 8
            if lastValidLine < master.numberOfLines:
               master.numValidLines = slave.numValidLines - 8
            else:
               master.numValidLines = master.numberOfLines - master.firstValidLine

    elif (minAz < 0) and (maxAz > 0):
            master.firstValidLine = slave.firstValidLine - int(np.floor(minAz) - 4)
            lastValidLine = master.firstValidLine + slave.numValidLines + int(np.floor(minAz) - 8) - int(np.ceil(maxAz))
            if lastValidLine < master.numberOfLines:
               master.numValidLines = slave.numValidLines + int(np.floor(minAz) - 8) - int(np.ceil(maxAz))
            else:
               master.numValidLines = master.numberOfLines - master.firstValidLine

    return master

def adjustCommonValidRegion(master,slave):
    # valid lines between master and slave


    master_lastValidLine = master.firstValidLine + master.numValidLines - 1
    master_lastValidSample = master.firstValidSample + master.numValidSamples - 1
    slave_lastValidLine = slave.firstValidLine + slave.numValidLines - 1
    slave_lastValidSample = slave.firstValidSample + slave.numValidSamples - 1

    igram_lastValidLine = min(master_lastValidLine, slave_lastValidLine)
    igram_lastValidSample = min(master_lastValidSample, slave_lastValidSample)

    master.firstValidLine = max(master.firstValidLine, slave.firstValidLine)
    master.firstValidSample = max(master.firstValidSample, slave.firstValidSample)

    master.numValidLines = igram_lastValidLine - master.firstValidLine + 1
    master.numValidSamples = igram_lastValidSample - master.firstValidSample + 1


def getValidLines(slave, rdict, inname, misreg_az=0.0, misreg_rng=0.0):
    '''
    Looks at the master, slave and azimuth offsets and gets the Interferogram valid lines 
    '''
    import numpy as np
    import isce
    import isceobj

    dimg = isceobj.createSlcImage()
    dimg.load(inname + '.xml')
    shp = (dimg.length, dimg.width)
    az = np.fromfile(rdict['azimuthOff'], dtype=np.float32).reshape(shp)
    az += misreg_az
    aa = np.zeros(az.shape)
    aa[:,:] = az
    aa[aa < -10000.0] = np.nan
    amin = np.nanmin(aa)
    amax = np.nanmax(aa)

    rng = np.fromfile(rdict['rangeOff'], dtype=np.float32).reshape(shp)
    rng += misreg_rng
    rr = np.zeros(rng.shape)
    rr[:,:] = rng
    rr[rr < -10000.0] = np.nan
    rmin = np.nanmin(rr)
    rmax = np.nanmax(rr)

    return amin, amax, rmin, rmax



def asBaseClass(inobj):
    '''
    Return as TOPSSwathSLCProduct.
    '''
    from isceobj.Sensor.TOPS.TOPSSwathSLCProduct import TOPSSwathSLCProduct
    
    
    def topsproduct(cobj):
        obj = TOPSSwathSLCProduct()
        obj.configure()

        for x in obj.parameter_list:
            val = getattr(cobj, x.attrname)
            setattr(obj, x.attrname, val)

        for x in obj.facility_list:
            attrname = x.public_name
            val = getattr(cobj, x.attrname)
            setattr(obj, x.attrname, val)
        
        return obj


    if isinstance(inobj, coregSwathSLCProduct):
        return topsproduct(inobj)

    elif isinstance(inobj, TOPSSwathSLCProduct):
        return inobj
    else:
        raise Exception('Cannot be converted to TOPSSwathSLCProduct')

def getSwathList(indir):

    swathList = []
    for x in [1,2,3]:
       SW = os.path.join(indir,'IW{0}'.format(x))
       if os.path.exists(SW):
          swathList.append(x)

    return swathList



