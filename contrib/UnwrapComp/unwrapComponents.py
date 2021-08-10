#!/usr/bin/env python3

import os
import numpy as np 
import argparse
import isce
import isceobj
from imageMath import IML
from osgeo import gdal
from osgeo.gdalconst import *

import logging
import scipy.spatial as SS

import string
import random

from .phaseUnwrap import Vertex, Edge, PhaseUnwrap
from iscesys.Component.Component import Component

SOLVER = Component.Parameter(
    'solver',
    public_name='SOLVER',
    default='pulp',
    type=str,
    mandatory=False,
    intent='input',
    doc='Linear Programming Solver'
)

REDUNDANT_ARCS = Component.Parameter(
    'redArcs',
    public_name='REDUNDANT_ARCS',
    default=-1,
    type=int,
    mandatory=False,
    intent='input',
    doc='Redundant Arcs for Unwrapping'
)

INP_FILE = Component.Parameter(
    'inpFile',
    public_name='INPUT',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='Input File Name'
)

CONN_COMP_FILE = Component.Parameter(
    'ccFile',
    public_name='CONN_COMP_FILE',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='Ouput File Name'
)

OUT_FILE = Component.Parameter(
    'outFile',
    public_name='OUTPUT',
    default=None,
    type=str,
    mandatory=False,
    intent='input',
    doc='Ouput File Name'
)

solverTypes = ['pulp', 'glpk', 'gurobi']
redarcsTypes = {'MCF':-1, 'REDARC0':0, 'REDARC1':1, 'REDARC2':2}

class UnwrapComponents(Component):
    ''' 
    2-Stage Phase Unwrapping
    '''
    family='unwrapComps'
    logging_name='contrib.unwrapComponents'
    parameter_list = (
                      SOLVER,
                      REDUNDANT_ARCS,
                      INP_FILE,
                      CONN_COMP_FILE,
                      OUT_FILE
                     )
    facility_list = ()

    def unwrapComponents(self):
        
        if self.inpFile is None:
            print("Error. Input interferogram file not set.")
            raise Exception

        if self.ccFile is None:
            print("Error. Connected Components file not set.")
            raise Exception

        if self.solver not in solverTypes:
            raise ValueError(self.treeType + ' must be in ' + str(unwTreeTypes))

        if self.redArcs not in redarcsTypes.keys():
            raise ValueError(self.redArcs + ' must be in ' + str(redarcsTypes))

        if self.outFile is None:
            self.outFile = self.inpFile.split('.')[0] + '_2stage.xml' 

        self.__getAccessor__()
        self.__run__()
        self.__finalizeImages__()

        # -- D. Bekaert - Make the unwrap file consistent with default ISCE convention.
        # unwrap file is two band with one band amplitude and other being phase.
        if self.inpAmp:
            command ="imageMath.py -e='a_0;b_0' --a={0} --b={1} -o={2} -s BIL -t float"
            commandstr = command.format(self.inpFile,self.outFile,os.path.abspath(self.outFile_final))
            os.system(commandstr)
            
            # update the xml file to indicate this is an unwrap file
            unwImg = isceobj.createImage()
            unwImg = self.unwImg
            unwImg.dataType = 'FLOAT'
            unwImg.scheme = 'BIL'
            unwImg.bands = 2
            unwImg.imageType = 'unw'
            unwImg.setFilename(self.outFile_final)
            unwImg.dump(self.outFile_final + ".xml")
           
            # remove the temp file
            os.remove(self.outFile)
            if os.path.isfile(self.outFile + '.hdr'):
                os.remove(self.outFile + '.hdr')
            if os.path.isfile(self.outFile + '.vrt'):
                os.remove(self.outFile + '.vrt')
            if os.path.isfile(self.outFile + '.xml'):
                os.remove(self.outFile + '.xml')
        # -- Done

        return

    def setRedArcs(self, redArcs):
        """ Set the Redundant Arcs to use for LP unwrapping """
        self.redArcs = redArcs

    def setSolver(self, solver):
        """ Set the solver to use for unwrapping """
        self.solver = solver

    def setInpFile(self, input):
        """ Set the input Filename for 2-stage unwrapping """
        self.inpFile = input

    def setOutFile(self, output): 
        """ Set the output File name """
        self.outFile = output
    
    def setConnCompFile(self, connCompFile):
        """ Set the connected Component file """
        self.ccFile = connCompFile

    def __getAccessor__(self):
        """ This reads in the input unwrapped file from Snaphu and Connected Components """

        # Snaphu Unwrapped Phase
        inphase = IML.mmapFromISCE(self.inpFile, logging)
        if len(inphase.bands) == 1:
            self.inpAcc = inphase.bands[0]
            self.inpAmp = False         # D. Bekaert track if two-band file or not
        else:
            self.inpAcc = inphase.bands[1]
            self.inpAmp = True
            self.outFile_final = self.outFile
            self.outFile = self.outFile + "_temp"

       

        # Connected Component
        inConnComp = IML.mmapFromISCE(self.ccFile, logging)
        if len(inConnComp.bands) == 1:
            # --D. Bekaert
            # problem with using .hdr files see next item. Below is no longer needed
            # Gdal dependency for computing proximity
            # self.__createGDALHDR__(inConnComp.name, inConnComp.width, inConnComp.length)
            # --Done

            # --D. Bekaert - make sure gdal is using the vrt file to load the data.
            # i.e. gdal will default to envi headers first, for which the convention is filename.dat => filename.hdr.
            # for the connected component this load the wrong header: topophase.unw.conncomp => topophase.unw.hdr
            # force therefore to use the vrt file instead.
            inConnComp_filename, inConnComp_ext = os.path.splitext(inConnComp.name)
            # fix the file to be .vrt
            if inConnComp_ext != '.vrt':
                if inConnComp_ext != '.conncomp' and inConnComp_ext != '.geo': 
                    inConnComp.name = inConnComp_filename + ".vrt"
                else:
                    inConnComp.name = inConnComp.name + ".vrt"
            if not os.path.isfile(inConnComp.name):
                raise Exception("Connected Component vrt file does not exist")
            print("GDAL using " + inConnComp.name)
            # --Done
            self.cc_ds = gdal.Open(inConnComp.name, GA_ReadOnly)
            self.ccband = self.cc_ds.GetRasterBand(1)
            self.conncompAcc = self.ccband.ReadAsArray()
        else:
            raise Exception("Connected Component Input File has 2 bands: Expected only one")
 
        return

    @staticmethod
    def __createGDALHDR__(connfile, width, length):
        '''
        Creates an ENVI style HDR file for use with GDAL API.
        '''
        import isceobj

        tempstring = """ENVI
description = {{Snaphu connected component file}}
samples = {0}
lines   = {1}
bands   = 1
header offset = 0
file type = ENVI Standard
data type = 1
interleave = bsq
byte order = 0
band names = {{component (Band 1) }}
"""

        outstr = tempstring.format(width, length)
        
        with open(connfile + '.hdr', 'w') as f:
            f.write(outstr)

        return

    def __run__(self):
        '''
        The main driver.
        '''
        # Get number of Components in the output from Snaphu
        self.nComponents = self.__getNumberComponents__()

        # Get the nearest neighbors between connected components
        self.uniqVertices = self.__getNearestNeighbors__()

        ####Further reduce number of vertices using clustering
        x = [pt.x for pt in self.uniqVertices]
        y = [pt.y for pt in self.uniqVertices]
        compnum = [pt.compNumber for pt in self.uniqVertices]

        # plotTriangulation(inps.connfile, uniqVertices, tris)
        phaseunwrap = PhaseUnwrap(x=x, y=y, phase=self.inpAcc[y,x], compNum=compnum, redArcs=redarcsTypes[self.redArcs])
        phaseunwrap.solve(self.solver)
        cycles = phaseunwrap.unwrapLP()

        # Map component to integer
        compMap = self.__compToCycle__(cycles, compnum)
        compCycles = np.array([0] + list(compMap.values()))
        cycleAdjust = compCycles[self.conncompAcc]

        self.outAcc = self.inpAcc - cycleAdjust * (2*np.pi)
        self.outAcc[self.conncompAcc == 0] = 0.0

        # Create xml arguments 
        self.__createImages__()

        return

    # Maps component number to n
    def __compToCycle__(self, cycle, compnum):
        compMap = {} 
        for n, c in zip(cycle, compnum):
          try:
            compN = compMap[c]
            # Test if same cycle
            if (compN == n):
              continue
            else:
              raise ValueError("Incorrect phaseunwrap output: Different cycles in same components")
          except:
            # Component cycle doesn;t exist in the dictionary
            compMap[c] = n  
      
        return compMap

    def __getNumberComponents__(self):
        #Determine number of connected components
        numComponents = np.nanmax(self.conncompAcc)
        print('Number of connected components: %d'%(numComponents))

        if numComponents == 1:
            print('Single connected component in image. 2 Stage will have effect')
            sys.exit(0)
        elif numComponents == 0:
            print('Appears to be a null image. No connected components')
            sys.exit(0)

        return numComponents

    # Get unique vertices
    def __getUniqueVertices__(self, vertices):
        ####Find unique vertices
        uniqVertices = []

        # Simple unique point determination
        for point in vertices:
            if point not in uniqVertices:
                uniqVertices.append(point)

        print('Number of unique vertices: %d'%(len(uniqVertices)))
        return uniqVertices

    def __getNearestNeighbors__(self):
        '''
        Find the nearest neighbors of particular component amongst other components.
        '''
        #Initialize list of vertices
        vertices = []

        mem_drv = gdal.GetDriverByName('MEM')
        for compNumber in range(1, self.nComponents+1):
          options = []
          options.append('NODATA=0')
          options.append('VALUES=%d'%(compNumber))

          dst_ds  = mem_drv.Create('', self.cc_ds.RasterXSize, self.cc_ds.RasterYSize, 1, gdal.GDT_Int16)
          dst_ds.SetGeoTransform( self.cc_ds.GetGeoTransform() )
          dst_ds.SetProjection( self.cc_ds.GetProjectionRef() )
          dstband = dst_ds.GetRasterBand(1)
          print('Estimating neighbors of component: %d'%(compNumber))

          gdal.ComputeProximity(self.ccband, dstband, options, callback = gdal.TermProgress)
          width = self.cc_ds.RasterXSize
          dist = dstband.ReadAsArray()


          #For each components, find the closest neighbor
          #from the other components
          ptList = []
          for comp in range(1, self.nComponents+1):
              if comp != compNumber:
                  marr = np.ma.array(dist, mask=(self.conncompAcc !=comp)).argmin()
                  point = Vertex()
                  point.y,point.x = np.unravel_index(marr, self.conncompAcc.shape)
                  point.compNumber = comp
                  point.source = compNumber
                  point.dist = dist[point.y, point.x]
                  ptList.append(point)

          vertices += ptList

          # Emptying dst_ds
          dst_ds = None

        # Emptying src_ds
        src_ds = None
        uniqVertices = self.__getUniqueVertices__(vertices)

        return uniqVertices 

    def __createImages__(self):
        '''
        Create any outputs that need to be generated always here.
        '''
        # write corresponding xml
        unwImg = isceobj.createImage()
        unwImg.dataType = 'FLOAT'
        unwImg.scheme = 'BIL'
        unwImg.imageType = 'unw'
        unwImg.bands = 1
        unwImg.setAccessMode('WRITE')
        unwImg.setWidth(self.inpAcc.shape[1])
        unwImg.setLength(self.inpAcc.shape[0])

        ## - D. Bekaert:  adding the geo-information too using the original unwrap file
        # gives gdal as input the vrt files
        inFilepart1, inFilepart2 = os.path.splitext(self.inpFile)
        if inFilepart2 != '.vrt' and inFilepart2 != '.hdr':
            inFile = self.inpFile + ".vrt"
        else:
            inFile = inFilepart1 + ".vrt"
        data_or =  gdal.Open(inFile, gdal.GA_ReadOnly)
        # transformation (lines/pixels or lonlat and the spacing 1/-1 or deltalon/deltalat)
        transform_or = data_or.GetGeoTransform()
        unwImg.firstLongitude = transform_or[0]
        unwImg.firstLatitude = transform_or[3] 
        unwImg.deltaLatitude = transform_or[5] 
        unwImg.deltaLongitude = transform_or[1] 
        # store the information for later as this does not change
        self.unwImg = unwImg
        ## - DONE

        unwImg.setFilename(self.outFile)




        # Bookmarking for rendering later
        self._createdHere.append((unwImg, self.outAcc))

    def __finalizeImages__(self):
        '''
        Close any images that were created here and 
        not provided by user.
        '''
            
        for img, buf in self._createdHere:
            # Create Xml
            img.renderHdr()
            
            # Write Buffer
            buf.astype(dtype=np.float32).tofile(self.outFile)

        self._createdHere = []
        return

    def __init__(self, family='', name=''):
        super().__init__(family=self.__class__.family, name=name if name else self.__class__.family)
        self.configure()

        # Local Variables
        self.nComponents    = None
        self.uniqVertices   = None
        self.width          = None

        # Files related
        self.inpFile        = None
        self.ccFile         = None
        self.outFile        = None
        self._createdHere   = []

        # Accessors for the images 
        self.inpAcc         = 0
        self.conncompAcc    = 0
        self.outAcc         = 0

        return

#end class
if __name__ == "__main__":

    import isceobj

    unw = UnwrapComponents()
    unw.setInpFile(inpFile)
    unw.setConnCompFile(ccFile)
    unw.unwrapComponents()
