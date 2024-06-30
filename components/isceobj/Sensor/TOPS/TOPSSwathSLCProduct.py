#!/usr/bin/env python3

import isce
import datetime
import isceobj
import numpy as np
from iscesys.Component.Component import Component
from iscesys.Traits import datetimeType


####List of parameters
IMAGING_MODE = Component.Parameter('mode',
        public_name = 'imaging mode',
        default = 'TOPS',
        type = str,
        mandatory = False,
        doc = 'Imaging mode')

FOLDER = Component.Parameter('folder',
        public_name = 'folder',
        default = None,
        type = str,
        mandatory = True,
        doc = 'Folder corresponding to single swath of TOPS SLC')

SPACECRAFT_NAME = Component.Parameter('spacecraftName',
    public_name='spacecraft name',
    default=None,
    type = str,
    mandatory = True,
    doc = 'Name of the space craft')

MISSION = Component.Parameter('mission',
        public_name = 'mission',
        default = None,
        type = str,
        mandatory = True,
        doc = 'Mission name')

PROCESSING_FACILITY = Component.Parameter('processingFacility',
    public_name='processing facility',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Processing facility information')

PROCESSING_SYSTEM = Component.Parameter('processingSystem',
    public_name='processing system',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Processing system information')

PROCESSING_SYSTEM_VERSION = Component.Parameter('processingSoftwareVersion',
    public_name='processing software version',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Processing system software version')

ASCENDING_NODE_TIME = Component.Parameter('ascendingNodeTime',
        public_name='ascending node time',
        default=None,
        type=datetimeType,
        mandatory=True,
        doc='Ascending node time corresponding to the acquisition')

NUMBER_BURSTS = Component.Parameter('numberOfBursts',
        public_name = 'number of bursts',
        default = None,
        type = int,
        mandatory = True,
        doc = 'Number of bursts in the product')

####List of facilities
BURSTS = Component.Facility('bursts',
        public_name='bursts',
        module = 'iscesys.Component',
        factory = 'createTraitSeq',
        args=('burst',),
        mandatory = False,
        doc = 'Trait sequence of burst SLCs')

class TOPSSwathSLCProduct(Component):
    """A class to represent a burst SLC along a radar track"""
    
    family = 'topsswathslc'
    logging_name = 'isce.tops.swath.slc'

    facility_list = (BURSTS,)


    parameter_list = (IMAGING_MODE,
                      FOLDER,
                      SPACECRAFT_NAME,
                      MISSION,
                      PROCESSING_FACILITY,
                      PROCESSING_SYSTEM,
                      PROCESSING_SYSTEM_VERSION,
                      ASCENDING_NODE_TIME,
                      NUMBER_BURSTS
                      )


    facility_list = (BURSTS,)


    def __init__(self,name=''):
        super(TOPSSwathSLCProduct, self).__init__(family=self.__class__.family, name=name)
        return None

    @property
    def sensingStart(self):
        return self.bursts[0].sensingStart

    @property
    def sensingStop(self):
        return self.bursts[-1].sensingStop

    @property
    def sensingMid(self):
        return self.sensingStart + 0.5 * (self.sensingStop - self.sensingStart)

    @property
    def startingRange(self):
        return self.bursts[0].startingRange

    @property
    def farRange(self):
        return self.bursts[0].farRange

    @property
    def midRange(self):
        return 0.5 * (self.startingRange + self.farRange)

    @property
    def orbit(self):
        '''
        For now all bursts have same state vectors.
        This will be the case till we build mechanisms for bursts to share metadata.
        '''
        return self.bursts[0].orbit

    def getBbox(self ,hgtrange=[-500,9000]):
        '''
        Bounding box estimate.
        '''

        ts = [self.sensingStart, self.sensingStop]
        rngs = [self.startingRange, self.farRange]
       
        pos = []
        for ht in hgtrange:
            for tim in ts:
                for rng in rngs:
                    llh = self.orbit.rdr2geo(tim, rng, height=ht)
                    pos.append(llh)

        pos = np.array(pos)

        bbox = [np.min(pos[:,0]), np.max(pos[:,0]), np.min(pos[:,1]), np.max(pos[:,1])]
        return bbox

    ####Functions to assist with deramping
    def computeAzimuthCarrier(self, burst, offset=0.0, position=None):
        '''
        Returns the ramp function as a numpy array.
        '''
        Vs = np.linalg.norm(burst.orbit.interpolateOrbit(burst.sensingMid, method='hermite').getVelocity())
        Ks =   2 * Vs * burst.azimuthSteeringRate / burst.radarWavelength 


        if position is None:
            rng = np.arange(burst.numberOfSamples) * burst.rangePixelSize + burst.startingRange

## Seems to work best for basebanding data
            eta =( np.arange(0, burst.numberOfLines) - (burst.numberOfLines//2)) * burst.azimuthTimeInterval +  offset * burst.azimuthTimeInterval

            f_etac = burst.doppler(rng)
            Ka     = burst.azimuthFMRate(rng)

            eta_ref = (burst.doppler(burst.startingRange) / burst.azimuthFMRate(burst.startingRange) ) - (f_etac / Ka)

#            eta_ref *= 0.0
            Kt = Ks / (1.0 - Ks/Ka)


            carr = np.pi * Kt[None,:] * ((eta[:,None] - eta_ref[None,:])**2)

        else:
            ####y and x need to be zero index
            y,x = position

            eta = (y - (burst.numberOfLines//2)) * burst.azimuthTimeInterval + offset * burst.azimuthTimeInterval
            rng = burst.startingRange + x * burst.rangePixelSize 
            f_etac = burst.doppler(rng)
            Ka  = burst.azimuthFMRate(rng)

            eta_ref = (burst.doppler(burst.startingRange) / burst.azimuthFMRate(burst.startingRange)) - (f_etac / Ka)
#            eta_ref *= 0.0
            Kt = Ks / (1.0 - Ks/Ka)

            carr = np.pi * Kt * ((eta - eta_ref)**2)

        return carr


    def computeRamp(self, burst, offset=0.0, position=None):
        '''
        Compute the phase ramp.
        '''
        cJ = np.complex64(1.0j)
        carr = self.computeAzimuthCarrier(burst,offset=offset, position=position)
        ramp = np.exp(-cJ * carr)
        return ramp



    ####Functions to help with finding overlap between products
    def getBurstOffset(self, sframe):
        '''
        Identify integer burst offset between 2 products.
        Compare the mid frames to start. Returns the integer offset between frame indices.
        '''


        if (len(sframe.bursts) < len(self.bursts)):
            return -sframe.getBurstOffset(self)

        checkBursts = [0.5, 0.25, 0.75, 0, 1]
        offset = []
        for bfrac in checkBursts:
            mind = int(self.numberOfBursts * bfrac)
            mind = np.clip(mind, 0, self.numberOfBursts - 1) 
            
            frame = self.bursts[mind]
            tmid = frame.sensingMid
            sv = frame.orbit.interpolateOrbit(tmid, method='hermite')
            mpos = np.array(sv.getPosition())
            mvel = np.array(sv.getVelocity())
        
            mdist = 0.2 * np.linalg.norm(mvel) * frame.azimuthTimeInterval * frame.numberOfLines

            arr = []
            for burst in sframe.bursts:
                tmid = burst.sensingMid
                sv = burst.orbit.interpolateOrbit(tmid, method='hermite')
                dr = np.array(sv.getPosition()) - mpos
                alongtrackdist = np.abs(np.dot(dr, mvel)) / np.linalg.norm(mvel)
                arr.append(alongtrackdist)

            arr = np.array(arr)
            ind = np.argmin(arr)
       
            if arr[ind] < mdist:
                return ind-mind
        
        raise  Exception('Could not determine a suitable burst offset')
        return 

    def getCommonBurstLimits(self, sFrame):
        '''
        Get range of min to max bursts w.r.t another swath product.
        minBurst, maxBurst can together be put into a slice object.
        '''
        burstoffset = self.getBurstOffset(sFrame)
        print('Estimated burst offset: ', burstoffset)

        minBurst = max(0, -burstoffset)
        maxBurst = min(self.numberOfBursts, sFrame.numberOfBursts - burstoffset)

        return burstoffset, minBurst, maxBurst


    def estimateAzimuthCarrierPolynomials(self, burst, offset=0.0,
                            xstep=500, ystep=50,
                            azorder=5, rgorder=3, plot=False):
        '''
        Estimate a polynomial that represents the carrier on a given burst. To be used with resampling.
        '''

        from isceobj.Util.Poly2D import Poly2D

        ####TOPS steering component of the azimuth carrier
        x = np.arange(0, burst.numberOfSamples,xstep,dtype=int)
        y = np.arange(0, burst.numberOfLines, ystep, dtype=int)

        xx,yy = np.meshgrid(x,y)


        data = self.computeAzimuthCarrier(burst, offset=offset, position=(yy,xx))


        ###Compute the doppler component of the azimuth carrier
        dop = burst.doppler
        dpoly = Poly2D()
        dpoly._meanRange = (dop._mean - burst.startingRange)/ burst.rangePixelSize
        dpoly._normRange = dop._norm / burst.rangePixelSize
        coeffs = [2*np.pi*val*burst.azimuthTimeInterval for val in dop._coeffs]
        zcoeffs = [0. for val in coeffs]
        dpoly.initPoly(rangeOrder=dop._order, azimuthOrder=0)
        dpoly.setCoeffs([coeffs])

    
        ####Need to account for 1-indexing in Fortran code
        poly = Poly2D()
        poly.initPoly(rangeOrder = rgorder, azimuthOrder = azorder)
        poly.polyfit(xx.flatten()+1, yy.flatten()+1, data.flatten()) #, maxOrder=True)  
        poly.createPoly2D() # Cpointer created 

        ###Run some diagnostics to raise warning
        fit = poly(yy+1,xx+1)
        diff = data - fit
        maxdiff = np.max(np.abs(diff))
        print('Misfit radians - Max: {0} , Min : {1} '.format(np.max(diff), np.min(diff)))

        if (maxdiff > 0.01):
            print('Warning: The azimuth carrier polynomial may not be accurate enough')

        if plot:   ####For debugging only

            import matplotlib.pyplot as plt

            plt.figure('Original')
            plt.imshow(data)
            plt.colorbar()

            plt.figure('Fit')
            plt.imshow(fit)
            plt.colorbar()

            plt.figure('diff')
            plt.imshow(diff)
            plt.colorbar()


            plt.show()

        return poly, dpoly
