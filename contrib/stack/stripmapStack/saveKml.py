#!/usr/bin/env python3
#
# Author: Heresh Fattahi
# Copyright 2016
#
import argparse
import isce
import isceobj
import os
import gdal
import matplotlib as mpl;  #mpl.use('Agg')
import matplotlib.pyplot as plt
from pykml.factory import KML_ElementMaker as KML
import numpy as np
mpl.use('Agg')

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser( description='Create a kml file from geocoded products.')
    parser.add_argument('-f', '--filelist', dest='prodlist', type=str, required=True,
            help='Input file to be geocoded')
#    parser.add_argument('-b', '--bbox', dest='bbox', type=str, required=True,
#            help='Bounding box (SNWE)')
    parser.add_argument('-m', '--min', dest='min', type=float, default=None,
            help='minimum value of colorscale')
    parser.add_argument('-M', '--max', dest='max', type=float, default=None,
            help='maximum value of color scale')
    parser.add_argument('-d', '--dpi', dest='dpi', type=int, default=500,
            help='dpi of the png image')
    parser.add_argument('-c', '--color_map', dest='color_map', type=str, default='jet',
            help='matplotlib colormap')
    parser.add_argument('-u', '--unit', dest='unit', type=str, default='',
            help='unit in which data is displayed')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
            help= 'scale factor to scale the data before display')
    parser.add_argument('-b', '--start_time', dest='startTime', type=str, default='',
            help= 'start time of the observation')
    parser.add_argument('-e', '--end_time', dest='endTime', type=str, default='',
            help= 'end time of the observation')    
    parser.add_argument('-r', '--reverse_color_map', dest='reverseColorMap', type=str, default='no',
            help= 'reverse color map (default: no)') 
    parser.add_argument('-w', '--rewrap', dest='rewrap', type=str, default='no',
            help= 'reverse color map (default: no)')
    parser.add_argument('-n', '--band_number', dest='bandNumber', type=int, default=1,
            help='band number if multiple bands exist')
    return parser

def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps =  parser.parse_args(args = iargs)

    inps.prodlist = inps.prodlist.split()
    return inps

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r

def get_lat_lon(file):

    ds=gdal.Open(file)
    b=ds.GetRasterBand(1)

    width=b.XSize
    length = b.YSize

    minLon = ds.GetGeoTransform()[0]
    deltaLon = ds.GetGeoTransform()[1]
    maxLon = minLon + width*deltaLon    

    maxLat = ds.GetGeoTransform()[3]
    deltaLat = ds.GetGeoTransform()[5]
    minLat = maxLat + length*deltaLat

    return minLat, maxLat, minLon, maxLon

def rewrap(unw):
   rewrapped = unw - np.round(unw/(2*np.pi)) * 2*np.pi
   return rewrapped

def display(file,inps):
    ds = gdal.Open(file)
    b = ds.GetRasterBand(inps.bandNumber)
    data = b.ReadAsArray()
    data = data*inps.scale
    data[data==0]=np.nan
    #data = np.ma.masked_where(data == 0, data)
    if inps.rewrap=='yes':
       data = rewrap(data)

    if inps.min is None:
       inps.min = np.nanmin(data)

    if inps.max is None:
       inps.max = np.nanmax(data)

    width = b.XSize
    length = b.YSize    

    fig = plt.figure()
    fig = plt.figure(frameon=False)
    # fig.set_size_inches(width/1000,length/1000)
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
   # ax.patch.set_alpha(0.0)
    ax.set_axis_off()
    fig.add_axes(ax)

    aspect = width/(length*1.0)
    # ax.imshow(data,aspect='normal')
    cmap = plt.get_cmap(inps.color_map)
    if inps.reverseColorMap=='yes':
       cmap = reverse_colourmap(cmap) 
    cmap.set_bad(alpha=0.0)
 #   cmap.set_under('k', alpha=0)
    try:     ax.imshow(data, aspect = 'normal', vmax = inps.max, vmin = inps.min, cmap = cmap)
    except:  ax.imshow(data, aspect = 'normal', cmap = cmap)

    ax.set_xlim([0,width])
    ax.set_ylim([length,0])

    # figName = k+'.png'
    figName = file + '.png'
    plt.savefig(figName, pad_inches=0.0, transparent=True, dpi=inps.dpi)

    #############################
    #pc = plt.figure(figsize=(1,4))
    pc = plt.figure(figsize=(1.3,2))
    axc = pc.add_subplot(111)
    cmap=mpl.cm.get_cmap(name=inps.color_map)
    if inps.reverseColorMap=='yes':     
       cmap = reverse_colourmap(cmap)
    norm = mpl.colors.Normalize(vmin=inps.min, vmax=inps.max)
    clb = mpl.colorbar.ColorbarBase(axc,cmap=cmap,norm=norm, orientation='vertical')
    clb.set_label(inps.unit)
    pc.subplots_adjust(left=0.25,bottom=0.1,right=0.4,top=0.9)
    #pc.subplots_adjust(left=0.0,bottom=0.0,right=1.0,top=1.0)
   # pc.savefig(file+'_colorbar.png',transparent=True,dpi=300)
    pc.savefig(file+'_colorbar.png',dpi=300)

    return file + '.png' , file+'_colorbar.png'

def writeKML(file, img, colorbarImg,inps):
  South, North, West, East = get_lat_lon(file)  
  ############## Generate kml file
  print ('generating kml file')
  doc = KML.kml(KML.Folder(KML.name(os.path.basename(file))))
  slc = KML.GroundOverlay(KML.name(os.path.basename(img)),KML.Icon(KML.href(os.path.basename(img))),\
                          KML.TimeSpan(KML.begin(),KML.end()),\
                          KML.LatLonBox(KML.north(str(North)),KML.south(str(South)),\
                                        KML.east(str(East)),  KML.west(str(West))))
  doc.Folder.append(slc)

  #############################
  print ('adding colorscale')
  latdel = North-South
  londel = East-West
  
  slc1   = KML.ScreenOverlay(KML.name('colorbar'),KML.Icon(KML.href(os.path.basename(colorbarImg))),
        KML.overlayXY(x="0.0",y="1",xunits="fraction",yunits="fraction",),
        KML.screenXY(x="0.0",y="1",xunits="fraction",yunits="fraction",),
        KML.rotationXY(x="0.",y="1.",xunits="fraction",yunits="fraction",),
        KML.size(x="0",y="0.3",xunits="fraction",yunits="fraction",),
      )


  doc.Folder.append(slc1)

  

  #############################
  from lxml import etree
  kmlstr = etree.tostring(doc, pretty_print=True)
  print (kmlstr)
  kmlname = file + '.kml'
  print ('writing '+kmlname)
  kmlfile = open(kmlname,'wb')
  kmlfile.write(kmlstr)
  kmlfile.close()

  kmzName = file + '.kmz'
  print ('writing '+kmzName)
  cmdKMZ = 'zip ' + kmzName +' '+ os.path.basename(kmlname) +' ' + os.path.basename(img) + ' ' + os.path.basename(colorbarImg)
  os.system(cmdKMZ)



def runKml(inps):

    for file in inps.prodlist:
       file = os.path.abspath(file)
       img,colorbar = display(file,inps) 
       writeKML(file,img,colorbar,inps)

def main(iargs=None):
    '''
    Main driver.
    '''
    inps = cmdLineParse(iargs)
    runKml(inps)


if __name__ == '__main__':
    main()


