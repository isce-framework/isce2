#
# Author: Eric Gurrola
# Date: 2016
#

import os
from isceobj import createDemImage
from iscesys.DataManager import createManager

def createDem(usnwe, info, insar, demStitcher, useHighResOnly=False, useZeroTiles=False):
    """
    Create a dem object given a user specified snwe lat, lon bounding box (usnwe),
    a frame information object (info),
    an insar container object (insar),
    a configured demStitcher object,
    an option useHighResOnly (default False) to accept only high resolution dem with zero fill
    and option useZeroTiles (default False) to proceed with zero filled dem tiles if unavailable
    The insar object contains a configured demStitcher,
    """

    #get the south, north latitude and west, east longitude extremes (snwe) from the frame
    #information with additional padding of 0.2 degrees in each direction
    snwe = info.getExtremes(0.2)
    #take the larger bounding region from these frame snwe values and the user's specified usnwe,
    #if given
    if usnwe:
        op1 = (min, max)
        snwe = [op1[i%2](usnwe[i], snwe[i]) for i in range(4)]
    #round outwards (relative to bounding box) to nearest integer latitude, longitudes
    import math
    op2 = (math.floor, math.ceil)
    snwe = [op2[i%2](snwe[i]) for i in range(4)]

    #Record the user's (or default) preference for using zeroed out tiles when the DEM is not
    #available (should really be done before coming here)
    demStitcher.proceedIfZeroDem = useZeroTiles

    #get the name of the wgs84 dem and its metadata file
    demName = demStitcher.defaultName(snwe)
    demNameXml = demName + '.xml'
    wgs84demName = demName + '.wgs84'
    wgs84demNameXml = wgs84demName + '.xml'

    #save the name just in case
    insar.demInitFile = wgs84demNameXml

    #check to see if the demStitcher has a valid DEM image instance we can use
    demImage = demStitcher.image
    if demImage:
        #get the wgs84 version
        wgs84dem = get_wgs84dem(demStitcher, demImage)
        insar.demImage = wgs84dem
        return

    #If not, check if there is already one in the local directory to load from
    #wgs84 version?
    if os.path.isfile(wgs84demNameXml):
        wgs84demImage  = createDemImage()
        wgs84demImage.load(wgs84demNameXml)
        insar.demImage = wgs84demImage
        return

    #or create one from the non-wgs84 version
    if os.path.isfile(demNameXml):
        demImage  = createDemImage()
        demImage.load(demNameXml)
        wgs84demImage = get_wgs84dem(demStitcher, demImage)
        insar.demImage = wgs84demImage
        return

    #or in the DEMDB directory
    #the wgs84dem
    if "DEMDB" in os.environ and os.path.isfile(os.path.join(os.environ["DEMDB"], wgs84demNameXml)):
        dbwgs84dem = os.path.join(os.environ["DEMDB"], wgs84demNameXml)
        wgs84demImage  = createDemImage()
        wgs84demImage.load(dbwgs84dem)
        insar.demImage = wgs84demImage
        return

    #or from the non-wgs84 version
    if "DEMDB" in os.environ and os.path.isfile(os.path.join(os.environ["DEMDB"], demNameXml)):
        dbdem = os.path.join(os.environ["DEMDB"], demNameXml)
        demImage  = createDemImage()
        demImage.load(dbdem)
        wgs84demImage = get_wgs84dem(demStitcher, demImage)
        insar.demImage = wgs84demImage
        return

    #or finally, have the demStitcher download and stitch a new one.
    #stitch
    if useHighResOnly:
        #use the high res DEM. Fill the missing tiles
        demStitcher.noFilling = False
        stitchOk = demStitcher.stitch(snwe[0:2], snwe[2:4])
    else:
        #try to use the demStitcher (high resolution DEM by default)
        #and do not allow missing tiles
        demStitcher.noFilling = True
        stitchOk = demStitcher.stitch(snwe[0:2], snwe[2:4])
        #check if high resolution stitching was not OK
        if not stitchOk:
            #then maybe try the lower resolution DEM
            newDemStitcher = createManager('dem3')
            #and do not allow missing tiles
            newDemStitcher.noFilling = True
            #set the preference for proceeding if the server does not return a tile
            newDemStitcher.proceedIfNoServer = useZeroTiles
            #try again only if it's not already a low res instance
            if type(newDemStitcher) != type(demStitcher):
                stitchOk = newDemStitcher.stitch(snwe[0:2], snwe[2:4])
                if stitchOk:
                    #if low res was ok change the stitcher to dem3
                    demStitcher = newDemStitcher

            #if cannot form a full dem with either high and low res
            #then use what ever with have with high res
            if not stitchOk:
                 demStitcher.noFilling = False
                 stitchOk = demStitcher.stitch(snwe[0:2], snwe[2:4])

    #check if stitching worked
    if stitchOk:
        #get the image
        demImage = demStitcher.image
        #set the metadatalocation and _extraFilename attributes
        demImage.metadatalocation = demImage.filename + ".xml"
        demImage._extraFilename = demImage.metadatalocation.replace(".xml", ".vrt")

        #get the wgs84 dem
        wgs84demImage = get_wgs84dem(demStitcher, demImage)

        #if there is a global store, move the dem files to it
        if "DEMDB" in os.environ and os.path.exists(os.environ["DEMDB"]):
            #modify filename in the meta data to include
            #path to the global store

            #the demImage
            demImage.filename = os.path.join(os.environ["DEMDB"],
                demImage.filename)
            demImage.metadatalocation = os.path.join(os.environ["DEMDB"],
                demImage.metadatalocation)
            demImage._extraFilename = os.path.join(os.environ["DEMDB"],
                demImage._extraFilename)
            demImage.dump(demNameXml)

            #the wgs84demImage
            wgs84demImage.load(wgs84demNameXml)
            wgs84demImage.filename = os.path.join(os.environ["DEMDB"],
                wgs84demImage.filename)
            wgs84demImage.metadatalocation = os.path.join(os.environ["DEMDB"],
                wgs84demImage.metadatalocation)
            wgs84demImage._extraFilename = os.path.join(os.environ["DEMDB"],
                wgs84demImage._extraFilename)
            wgs84demImage.dump(wgs84demNameXml)

            #remove the demLat*.vrt file from the local directory because
            #a side effect of the demImage.dump() above was to create the
            #vrt in the location indicated by the path in the xml file.
            os.remove(demNameXml.replace('.xml','.vrt'))
            os.remove(wgs84demNameXml.replace('.xml','.vrt'))

            #move the demfiles to the global store
            #make list of dem file names to be moved to the global store
            import glob
            dwlist = glob.glob(demName+"*")
            import shutil
            #move the dem files to the global store
            for dwfile in dwlist:
                shutil.move(dwfile,os.environ["DEMDB"])

        #put the wgs84demImage in the InsarProc object
        insar.demImage = wgs84demImage
        #that's all
        return

    #exhausted all options; ask the user for help
    else:
        logger.error(
            "Cannot form the DEM for the region of interest. "+
            "If you have one, set the appropriate DEM "+
            "component in the input file.")
        raise Exception

    return


def get_wgs84dem(demStitcher, demImage):

    #check to see if demImage is actually an EGM96 referenced dem as expected
    if demImage.reference.upper() == 'EGM96':
        #look for wgs84 version of the dem with the expected name
        wgs84demName = demImage.filename + ".wgs84"
        wgs84demNameXml = wgs84demName + ".xml"
        if os.path.isfile(wgs84demName) and os.path.isfile(wgs84demNameXml):
            #create a DemImage instance
            wgs84demImage  = createDemImage()
            #load its state
            wgs84demImage.load(wgs84demNameXml)
        else:
            #correct the dem reference to the WGS84 ellipsoid
            wgs84demImage = demStitcher.correct(demImage)
            #set the metadatalocation
            wgs84demImage.metadatalocation = wgs84demNameXml
            #set the vrt filename (even though it is not yet created)
            wgs84demImage._extraFilename = wgs84demImage.metadatalocation.replace('.xml', '.vrt')

    #Check if the demImage is already referenced to WGS84
    elif demImage.reference.upper() == 'WGS84':
        wgs84demImage = demImage

    #all expectations have been exhausted; give up
    else:
        wgs84demImage = None
        logger.error(
            "Cannot form the WGS84 DEM for the region of interest. "+
            "If you have one, set the appropriate DEM "+
            "component in the input file.")
        raise Exception

    #return the wgs84demImage
    return wgs84demImage

#end-of-file
