#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import isce
import stdproc
import isceobj
import logging
import numpy as np
from Poly2d import Polynomial
from stdproc.stdproc.offsetpoly.Offsetpoly import Offsetpoly

logger = logging.getLogger('dense')
def load_pickle(step='outliers1'):
    import cPickle

    insarObj = cPickle.load(open('PICKLE/{0}'.format(step), 'rb'))
    return insarObj

def runOffPolyISCE(offField):
    '''
    Estimate polynomial here.
    '''

    inArr = np.array(offField.unpackOffsets())
    x = inArr[:,0]
    y = inArr[:,2]
    dx = inArr[:,1]
    dy = inArr[:,3]
    sig = inArr[:,4]

    obj = Offsetpoly()
    obj.setLocationAcross(list(x))
    obj.setLocationDown(list(y))
    obj.setSNR(list(sig))
    obj.setOffset(list(dy))
    obj.offsetpoly()
    val = obj.offsetPoly

#    print('Range: ', val)
    azpol = Polynomial(rangeOrder=2, azimuthOrder=2)
    azpol.setCoeffs([[val[0],val[1],val[4]],
                     [val[2], val[3]],
                     [val[5]]])


    obj.setOffset(list(dx))
    obj.offsetpoly()
    val = obj.offsetPoly

#    print('Azimuth: ', val)

    rgpol = Polynomial(rangeOrder=2, azimuthOrder=2)
    rgpol.setCoeffs([[val[0],val[1],val[4]],
                     [val[2], val[3]],
                     [val[5]]])

    return azpol, rgpol



def runOffPoly(offField):
    '''
    Estimate polynomial here.
    '''

    inArr = np.array(offField.unpackOffsets())
    x = inArr[:,0]
    y = inArr[:,2]
    dx = inArr[:,1]
    dy = inArr[:,3]
    sig = inArr[:,4]
    snr = 1.0 + 1.0/sig

    xOrder = 2
    yOrder = 2

    #####Normalization factors
    ymin = np.min(y)
    ynorm = np.max(y) - ymin
    if ynorm == 0:
        ynorm = 1.0

    yoff = np.int(np.round(np.mean(dy)))
    y = (y - ymin)/ynorm


    xmin = np.min(x)
    xnorm = np.max(x) - xmin
    if xnorm == 0:
        xnorm = 1.0

    x = (x-xmin)/xnorm

    arrList = []
    for ii in range(yOrder + 1):
        yfact = np.power(y, ii)
        for jj in range(yOrder + 1-ii):
            temp = np.power(x,jj)* yfact
            arrList.append(temp.reshape((temp.size,1)))

    A = np.hstack(arrList)

    A = A / snr[:,None]
    b = dy / snr

    val, res, rank, eigs = np.linalg.lstsq(A,b, rcond=1.0e-12)
    print('Az Chi : ', np.sqrt(res/(1.0*len(b))))
    
    azpol = Polynomial(rangeOrder=2, azimuthOrder=2)
    azpol.setCoeffs([val[0:3],val[3:5],val[5:]])
    azpol._meanRange = xmin
    azpol._normRange = xnorm
    azpol._meanAzimuth = ymin
    azpol._normAzimuth = ynorm

    b = dx/snr
    val,res, rank, eigs = np.linalg.lstsq(A,b, rcond=1.0e-12)
    print('Rg chi : ', np.sqrt(res/(1.0*len(b))))

    rgpol = Polynomial(rangeOrder=2, azimuthOrder=2)
    rgpol.setCoeffs([val[0:3],val[3:5],val[5:]])
    rgpol._meanRange = xmin
    rgpol._normRange = xnorm
    rgpol._meanAzimuth = ymin
    rgpol._normAzimuth = ynorm


    return azpol, rgpol

if __name__ == '__main__':
    iObj = load_pickle()
    print('Done loading pickle')

    width = iObj.getReferenceSlcImage().getWidth()
    length = iObj.getReferenceSlcImage().getLength()
    print('Image Dimensions: ', length, width)

    print('Results from numpy code')
    azpol, rgpol = runOffPoly(iObj.getRefinedOffsetField())

    print('Upper Left: ', rgpol(1,0), azpol(1,0))
    print('Upper Right: ', rgpol(1,width-1), azpol(1,width-1))
    print('Lower Left: ', rgpol(length+1,0), azpol(length+1,0))
    print('Lower Right: ', rgpol(length+1,width-1), azpol(length+1,width-1))


    print('Results from old method')
    az1, rg1 = runOffPolyISCE(iObj.getRefinedOffsetField())
    print('Upper Left: ', rg1(1,0), az1(1,0))
    print('Upper Right: ', rg1(1,width-1), az1(1,width-1))
    print('Lower Left: ', rg1(length+1,0), az1(length+1,0))
    print('Lower Right: ', rg1(length+1,width-1), az1(length+1,width-1))

