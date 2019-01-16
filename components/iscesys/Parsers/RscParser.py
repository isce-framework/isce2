#!/usr/bin/env python3 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                  Giangi Sacco
#                        NASA Jet Propulsion Laboratory
#                      California Institute of Technology
#                        (C) 2009  All Rights Reserved
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from __future__ import print_function
import sys
import os
import math
import logging
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
class RscParser:

    
    def parse(self,filename):
        try:
            file = open(filename)
        except IOError:
            self.logger.error("Error in RscParser. Cannot open file %s " %(filename))
            raise IOError
        allLines = file.readlines()
        dictionary = {}
        for line in allLines:
            
            if len(line) == 0:# empty line
                continue
            if line.startswith('#'):# comment
                continue
            if line.count('#'):# remove comments from line
                pos = line.find('#')
                line = line[0:pos]
            
            splitLine = line.split()
            if ((len(splitLine) < 2)):# remove lines that do not have at least two values
                continue
            if(len(splitLine) == 2):  #just key and value value
                
                dictionary[splitLine[0]] = splitLine[1]
            else:
                # the value is a list
                valList = []
                for i in range(1,len(splitLine)):
                    valList.append(splitLine[i])
                
                dictionary[splitLine[0]] = valList
        #for now call to top node as the filename 
        return {os.path.split(filename)[1]:dictionary}


    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d
    def __setstate__(self,d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('isce.iscesys.Parsers.RscParser')
    def __init__(self):
        self.logger = logging.getLogger('isce.iscesys.Parsers.RscParser')

def main(argv):
    PA = RscParser()
    print(PA.parse(argv[0]))

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


