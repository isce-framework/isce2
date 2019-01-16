#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Giangi Sacco
# Copyright 2010, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
#
# This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S.
# export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before 
# exporting such information to foreign countries or providing access to foreign persons.
#
#                               Jet Propulsion Lab
#                        California Institute of Technology
#                        (C) 2004-2006  All Rights Reserved
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import os
#import InitSlcImageComponent
from isceobj.SlcImage.SlcImage import SlcImage

def main():
        '''
        home = os.environ['HOME']
        filename = home + "/TEST_DIR/930110/930110.slc"
        accessmode = 'read'
        endian = 'l'
        width = 5700
        obj = SlcImage()
        obj.initImage(filename,accessmode,endian,width)
        image = obj.getImage()
        image.printObjectInfo()
        '''

if __name__ == "__main__":
    sys.exit(main()) 


# End of file
