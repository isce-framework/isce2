#!/usr/bin/env python3
#
# Author: Piyush Agram
# Copyright 2016
#
# Heresh Fattahi, updated for stack processing


import numpy as np 
import os
import isce
import isceobj
import datetime
import logging
import argparse
from isceobj.Util.ImageUtil import ImageLib as IML
from isceobj.Util.decorators import use_api
import s1a_isce_utils as ut
import glob
def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-i', '--master', type=str, dest='master', required=True, help='Path to folder')
    parser.add_argument('-k', '--kml', type=str, dest='shapefile', default=None, help='Path to kml')
    parser.add_argument('-f', '--figure', type=str, dest='figure', default=None, help='Path to output PDF')

    return parser


def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args=iargs)

    return inps

def main(iargs=None):
    '''
    Merge burst products to make it look like stripmap.
    Currently will merge interferogram, lat, lon, z and los.
    '''


    inps=cmdLineParse(iargs)
    from osgeo import ogr, osr
    import matplotlib
    if inps.shapefile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


    swathList = ut.getSwathList(inps.master)

    swathColors = ['r', 'g', 'b']
    shapeColors = ['FF0000','00FF00','0000FF']

    fig = plt.figure('Burst map')
    ax = fig.add_subplot(111,aspect='equal')
    
    tmin = None
    rmin = None
    
    xmin = 1e10
    ymin = 1e10
    xmax = -1e10
    ymax = -1e10


    if inps.shapefile is not None:
        ds = ogr.GetDriverByName('KML').CreateDataSource(inps.shapefile)
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS('WGS84')
        layer = ds.CreateLayer('bursts', srs=srs)
        field_name = ogr.FieldDefn("Name", ogr.OFTString)
        field_name.SetWidth(16)
        layer.CreateField(field_name)
        field_name = ogr.FieldDefn("OGR_STYLE", ogr.OFTString)
        layer.CreateField(field_name)


    for swath in swathList:
        ifg = ut.loadProduct(os.path.join(inps.master , 'IW{0}.xml'.format(swath)))
        minBurst = ifg.bursts[0].burstNumber
        maxBurst = ifg.bursts[-1].burstNumber

        if tmin is None:
            tmin = ifg.bursts[0].sensingStart
            dtime = ifg.bursts[0].azimuthTimeInterval
            rmin = ifg.bursts[0].startingRange
            drange = ifg.bursts[0].rangePixelSize


        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue
    
        if not inps.shapefile:
            for ii in range(minBurst, maxBurst + 1):
                burst = ifg.bursts[ii-minBurst]
                x0 = np.round( (burst.startingRange - rmin)/drange)
                y0 = np.round( (burst.sensingStart - tmin).total_seconds()/ dtime)
                if ii %2 == 0:
                    style = 'solid'
                else:
                    style = 'dashdot'

                ax.add_patch( patches.Rectangle(
                    (x0,y0),
                    burst.numValidSamples,
                    burst.numValidLines,
                    edgecolor=swathColors[swath-1],
                    facecolor=swathColors[swath-1],
                    alpha=0.2,
                linestyle=style))

                xmin = min(xmin, x0)
                xmax = max(xmax, x0 + burst.numValidSamples)
                ymin = min(ymin, y0)
                ymax = max(ymax, y0 + burst.numValidLines)
        else:
            for ii in range(minBurst, maxBurst+1):
                burst = ifg.bursts[ii-minBurst]
                t0 = burst.sensingStart + datetime.timedelta(seconds = burst.firstValidLine * burst.azimuthTimeInterval)
                t1 = t0 + datetime.timedelta(seconds = burst.numValidLines * burst.azimuthTimeInterval)
                r0 = burst.startingRange + burst.firstValidSample * burst.rangePixelSize
                r1 = r0 + burst.numValidSamples * burst.rangePixelSize

                earlyNear = burst.orbit.rdr2geo(t0,r0)
                earlyFar = burst.orbit.rdr2geo(t0,r1)
                lateFar = burst.orbit.rdr2geo(t1,r1)
                lateNear = burst.orbit.rdr2geo(t1,r0)

                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(earlyNear[1], earlyNear[0])
                ring.AddPoint(earlyFar[1], earlyFar[0])
                ring.AddPoint(lateFar[1], lateFar[0])
                ring.AddPoint(lateNear[1], lateNear[0])
                ring.AddPoint(earlyNear[1], earlyNear[0])

                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField('Name', 'IW{0}-{1}'.format(swath, ii))
                feature.SetField('OGR_STYLE', "PEN(c:#{0},w:8px)".format(shapeColors[swath-1]))
                feature.SetGeometry(ring)
                layer.CreateFeature(feature)
                feature = None

        
    if not inps.shapefile:   
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])

        if inps.figure is not None:
            plt.savefig(inps.figure, format='pdf')
        else:
            plt.show()

    else:
        ds = None
        print('Wrote KML file: ', inps.shapefile)

if __name__ == '__main__' :
    '''
    Merge products burst-by-burst.
    '''

    main()
