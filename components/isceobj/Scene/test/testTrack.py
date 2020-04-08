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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from isce import logging
from isceobj.Sensor.ERS import ERS
from isceobj.Scene.Track import Track
logger = logging.getLogger("testTrack")

def main():
    output = 'test.raw'
    frame1 = createERSFrame(leaderFile='/Users/szeliga/data/InSAR/raw/ers/track134/frame2961/930913/SARLEADER199309132961f134t',
                   imageryFile='/Users/szeliga/data/InSAR/raw/ers/track134/frame2961/930913/IMAGERY199309132961f134t',
                   output='frame2961.raw')
    frame2 = createERSFrame(leaderFile='/Users/szeliga/data/InSAR/raw/ers/track134/frame2979/930913/SARLEADER199309132979f134t',
                   imageryFile='/Users/szeliga/data/InSAR/raw/ers/track134/frame2979/930913/IMAGERY199309132979f134t',
                   output='frame2979.raw')

    track = Track()
    track.addFrame(frame1)
    track.addFrame(frame2)
    track.createTrack(output)

def createERSFrame(leaderFile=None,imageryFile=None,output=None):
    logger.info("Extracting ERS frame %s" % leaderFile)
    ers = ERS()
    ers._leaderFile = leaderFile
    ers._imageFile = imageryFile
    ers.output = output

    ers.extractImage()

    return ers.getFrame()

if __name__ == "__main__":
    main()
