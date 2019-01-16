//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//#
//#
//# Author: Piyush Agram
//# Copyright 2013, by the California Institute of Technology. ALL RIGHTS RESERVED.
//# United States Government Sponsorship acknowledged.
//# Any commercial use must be negotiated with the Office of Technology Transfer at
//# the California Institute of Technology.
//# This software may be subject to U.S. export control laws.
//# By accepting this software, the user agrees to comply with all applicable U.S.
//# export laws and regulations. User has the responsibility to obtain export licenses,
//# or other export authority as may be required before exporting such information to
//# foreign countries or providing access to foreign persons.
//#
//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifndef fitoffmoduleFortTrans_h
#define fitoffmoduleFortTrans_h

    #if defined(NEEDS_F77_TRANSLATION)

        #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
            #define fitoff_f fitoff_
            #define setMaxRms_f setmaxrms_
            #define setMinPoint_f setminpoint_
            #define setNSig_f setnsig_
            #define setNumberLines_f setnumberlines_
            #define setMinIter_f setminiter_
            #define setMaxIter_f setmaxiter_
            #define setL1normFlag_f setl1normflag_
            #define setLocationAcross_f setlocationacross_
            #define setLocationDown_f setlocationdown_
            #define setLocationAcrossOffset_f setlocationacrossoffset_
            #define setLocationDownOffset_f setlocationdownoffset_
            #define setSNR_f setsnr_
            #define setCovAcross_f setcovacross_
            #define setCovDown_f setcovdown_
            #define setCovCross_f setcovcross_
            #define setStdWriter_f setstdwriter_
            #define getAffineVector_f getaffinevector_

            #define allocate_LocationAcross_f allocate_locationacross_
            #define allocate_LocationDown_f allocate_locationdown_
            #define allocate_SNR_f allocate_snr_
            #define allocate_Covariance_f allocate_covariance_
            #define allocate_LocationAcrossOffset_f allocate_locationacrossoffset_
            #define allocate_LocationDownOffset_f allocate_locationdownoffset_
            #define deallocate_LocationAcross_f deallocate_locationacross_
            #define deallocate_LocationDown_f deallocate_locationdown_
            #define deallocate_LocationAcrossOffset_f deallocate_locationacrossoffset_
            #define deallocate_LocationDownOffset_f deallocate_locationdownoffset_
            #define deallocate_SNR_f deallocate_snr_
            #define deallocate_Covariance_f deallocate_covariance_
            #define getNumberOfRefinedOffsets_f getnumberofrefinedoffsets_
            #define getRefinedLocationAcross_f getrefinedlocationacross_
            #define getRefinedLocationDown_f getrefinedlocationdown_
            #define getRefinedLocationAcrossOffset_f getrefinedlocationacrossoffset_
            #define getRefinedLocationDownOffset_f getrefinedlocationdownoffset_
            #define getRefinedSNR_f getrefinedsnr_
            #define getRefinedCovAcross_f getrefinedcovacross_
            #define getRefinedCovDown_f getrefinedcovdown_
            #define getRefinedCovCross_f getrefinedcovcross_
        #else
            #error Unknown traslation for FORTRAN external symbols
        #endif
    #endif

#endif //fitoffmoduleFortTrans_h
