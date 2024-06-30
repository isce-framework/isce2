#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import sys
import isce
from iscesys.Display.Display import Display
##
# Call mdx.py argv.
# The first element in argv must be the metadata file (i.e. metadata.rsc or metadata.xml) when displaying an image (could be something else when printing help info). If the file does not end by .rsc or .xml, then one needs to specify
# the -type flag that could be  rsc or xml. For rsc type of metadata the rsc ROI_PAC format is assumed. For xml type the ISCE xml format is assumed.
# In case the data file name is not simply the metadata file name with the extension  removed (for instance metadata file image.int.rsc and data file image.int)
# then use the -image flag and specify the filename.
# If the type of image that needs to be displayed cannot be inferred from the extension (for ROI_PAC type) or from the metadata doc string (ISCE type) then specify the -ext flag.
# To print a list of extensions run mdx.py -ext.
# To print the usage with the list of options just run mdx.py with no arguments.
# The flags -cw,-e,-amp1,-amp2,-chdr,-RMG-Mag,-RMG_Hgt -wrap,-wrap and -cmap have some defaults value depending on the image type. By specifying these flags in the command line the default values can be overwritten.
# Whatever flags in the argv that are not part of the abovementioned ones, will be passed to mdx as arguments at the end of the command.
##
def main(argv = None):
    DS = Display()
    DS.mdx(argv)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit(main())
    else:
        sys.exit(main(sys.argv[1:]))
