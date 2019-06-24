import sys
import isceobj
import os
from isceobj.TopsProc.runUnwrapSnaphu import runUnwrapMcf 
from contrib.downsample_unwrapper.downsample_unwrapper import DownsampleUnwrapper 
def runUnwrap(self,costMode = None,initMethod = None, defomax = None, initOnly = None):
    #generate inputs from insar obj
    inps = {
            "flat_name":self._insar.filtFilename,
            "unw_name":self._insar.unwrappedIntFilename,
            "cor_name":self._insar.coherenceFilename,
            "range_looks":self.numberRangeLooks,
            "azimuth_looks":self.numberAzimuthLooks,
            "data_dir":self._insar.mergedDirname
            }
    
    du = DownsampleUnwrapper(inps)
    #modify the filenames so it uses the downsampled versions
    self._insar.filtFilename = du._dflat_name
    self._insar.unwrappedIntFilename = du._dunw_name
    self._insar.coherenceFilename = du._dcor_name
    self.numberRangeLooks = int(du._resamp*du._range_looks)
    self.numberAzimuthLooks = int(du._resamp*du._azimuth_looks)
    
    du.downsample_images(du._ddir,du._flat_name,du._cor_name,du._resamp)
    runUnwrapMcf(self)
    du.upsample_unw(du._ddir,du._flat_name,du._dunw_name,du._dccomp_name,upsamp=du._resamp,filt_sizes=(3,4))
    #put back the original values
    self._insar.filtFilename = du._flat_name
    self._insar.unwrappedIntFilename = du._unw_name
    self._insar.coherenceFilename = du._cor_name
    self.numberRangeLooks = int(du._range_looks)
    self.numberAzimuthLooks = int(du._azimuth_looks)