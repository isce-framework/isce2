#ifndef ampcormoduleFortTrans_h
#define ampcormoduleFortTrans_h

  #if defined(NEEDS_F77_TRANSLATION)

        #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)

              #define ampcor_f ampcor_

              #define setImageDatatype1_f setimagedatatype1_
              #define setLineLength1_f setlinelength1_
              #define setImageLength1_f setimagelength1_
              #define setImageDatatype2_f setimagedatatype2_
              #define setLineLength2_f setlinelength2_
              #define setImageLength2_f setimagelength2_
              #define setFirstSampleDown_f setfirstsampledown_
              #define setLastSampleDown_f setlastsampledown_
              #define setSkipSampleDown_f setskipsampledown_
              #define setFirstSampleAcross_f setfirstsampleacross_
              #define setLastSampleAcross_f setlastsampleacross_
              #define setSkipSampleAcross_f setskipsampleacross_
              #define setWindowSizeWidth_f setwindowsizewidth_
              #define setWindowSizeHeight_f setwindowsizeheight_
              #define setSearchWindowSizeWidth_f setsearchwindowsizewidth_
              #define setSearchWindowSizeHeight_f setsearchwindowsizeheight_
              #define setAcrossLooks_f setacrosslooks_
              #define setDownLooks_f setdownlooks_
              #define setOversamplingFactor_f setoversamplingfactor_
              #define setZoomWindowSize_f setzoomwindowsize_
              #define setAcrossGrossOffset_f setacrossgrossoffset_
              #define setDownGrossOffset_f setdowngrossoffset_
              #define setThresholdSNR_f setthresholdsnr_
              #define setThresholdCov_f setthresholdcov_
              #define setDebugFlag_f setdebugflag_
              #define setDisplayFlag_f setdisplayflag_
              #define setScaleFactorX_f setscalefactorx_
              #define setScaleFactorY_f setscalefactory_

              #define ampcorPrintState_f ampcorprintstate_

              #define getNumRows_f getnumrows_
              #define getCov1_f getcov1_
              #define getCov2_f getcov2_
              #define getCov3_f getcov3_
              #define getSNR_f  getsnr_
              #define getLocationAcross_f getlocationacross_
              #define getLocationAcrossOffset_f getlocationacrossoffset_
              #define getLocationDown_f getlocationdown_
              #define getLocationDownOffset_f getlocationdownoffset_


              #define allocate_locationAcross_f allocate_locationacross_
              #define allocate_locationDown_f allocate_locationdown_
              #define allocate_locationAcrossOffset_f allocate_locationacrossoffset_
              #define allocate_locationDownOffset_f allocate_locationdownoffset_
              #define allocate_snrRet_f allocate_snrret_
              #define allocate_cov1Ret_f allocate_cov1ret_
              #define allocate_cov2Ret_f allocate_cov2ret_
              #define allocate_cov3Ret_f allocate_cov3ret_

              #define deallocate_locationAcross_f deallocate_locationacross_
              #define deallocate_locationDown_f deallocate_locationdown_
              #define deallocate_locationAcrossOffset_f deallocate_locationacrossoffset_
              #define deallocate_locationDownOffset_f deallocate_locationdownoffset_
              #define deallocate_snrRet_f deallocate_snrret_
              #define deallocate_cov1Ret_f deallocate_cov1ret_
              #define deallocate_cov2Ret_f deallocate_cov2ret_
              #define deallocate_cov3Ret_f deallocate_cov3ret_

              #define setWinsizeFilt_f setwinsizefilt_
              #define setOversamplingFactorFilt_f setoversamplingfactorfilt_

        #else
            #error Unknown translation for FORTRAN external symbols
        #endif
  #endif
#endif //ampcormoduleFortTrans_h
