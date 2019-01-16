#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                  Giangi Sacco
#                        NASA Jet Propulsion Laboratory
#                      California Institute of Technology
#                        (C) 2009  All Rights Reserved
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import sys
import os
import math
import struct
sys.path.append(os.environ['HOME'] + '/LineAccessor/') 
import array
import fortranSrc
from install.LineAccessorPy import LineAccessorPy

def main():
    
    choice = 1
    if len(sys.argv) > 1:
        choice = int(sys.argv[1])
    
    if choice == 1 or choice == 2:
        LAGet = LineAccessorPy()
        LAGet.createLineAccessorObject()
        infile  = "testFile"
        fileSize = 4*4*10#10 lines 4 columns float type i.e. sizeof(float) = 4
        #create test file 160 bytes with integer from 0 to 159
        buffer = array.array('B')
        for i in range(fileSize):
            buffer.append(i)
        fout = open(infile, "wb")
        buffer.tofile(fout)
        fout.close()
        filemode = 'read'
        endian = 'l'
        machineEnd =  LAGet.getMachineEndianness()
        #choice = 1 test machine and file endianness being the same
        if choice == 1:
            if machineEnd == 'b':
                endian = 'b'
        else :
            if machineEnd == 'l':
                endian = 'b'
        type = "FLOAT";# using numpy nomenclature for variable type
        col = 4;# width of the tile. this means 10 lines in total. each line is col*sizeof(float) = 4*4 = 16 bytes.
        row = 3;# height of the tile. 
        # create image object to read from
        LAGet.initLineAccessor(infile,filemode,endian,type,row,col)
        
        LASet = LineAccessorPy()
        LASet.createLineAccessorObject()
        outfile = "testOutP";
        if  choice == 1:
            outfile += "1"
        else:
            outfile += "2"
        filemode1 = "writeread";
        # create image objet to write into
        LASet.initLineAccessor(outfile, filemode1, endian, type, row,col);
        #get the address of teh objects.
        addressGet = LAGet.getLineAccessorPointer()
        addressSet = LASet.getLineAccessorPointer()
        fortranSrc.testImageSetGet(addressGet,addressSet,choice);
        #need to do flushing and free memory
        LASet.finalizeLineAccessor();
        LAGet.finalizeLineAccessor();


    elif choice == 3:
        LAGet = LineAccessorPy()
        LAGet.createLineAccessorObject()
        infile  = "testSwap"
        outfile  = "testSwapOutP"
        type = "FLOAT";
        fileSize = 4*4*10#10 lines 4 columns float type i.e. sizeof(float) = 4
        #create test file 160 bytes with integer from 0 to 159
        buffer = array.array('B')
        for i in range(fileSize):
            buffer.append(i)
        fout = open(infile, "wb")
        buffer.tofile(fout)
        fout.close()
        LAGet.convertFileEndianness(infile,outfile,type);
    elif choice == 4:
        
        LAGet = LineAccessorPy()
        LAGet.createLineAccessorObject()
        infile  = "testFileBand"
        outfile  = "testFileBandOutP"
        fileSize = 4*4*10#10 lines 4 columns float type i.e. sizeof(float) = 4
        #create test file 160 bytes with integer from 0 to 159
        buffer = array.array('B')
        for i in range(fileSize):
            buffer.append(i)
        fout = open(infile, "wb")
        buffer.tofile(fout)
        fout.close()
        type = "FLOAT";# using numpy nomenclature for variable type
        col = 4;# width of the tile. this means 10 lines in total. each line is col*sizeof(float) = 4*4 = 16 bytes.
        bandIn = 1 # BSQ
        bandOut = 3 # BIL
        numBands = 2
        LAGet.changeBandScheme(infile,outfile,type,col,numBands,bandIn,bandOut)
    else:
        raise("Error. Wrong selection")



if __name__ == "__main__":
    sys.exit(main())
