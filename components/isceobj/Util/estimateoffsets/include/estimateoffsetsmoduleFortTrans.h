//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// United States Government Sponsorship acknowledged. This software is subject to
// U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
// (No [Export] License Required except when exporting to an embargoed country,
// end user, or in support of a prohibited end use). By downloading this software,
// the user agrees to comply with all applicable U.S. export laws and regulations.
// The user has the responsibility to obtain export licenses, or other export
// authority as may be required before exporting this software to any 'EAR99'
// embargoed foreign country or citizen of those countries.
//
// Author: Giangi Sacco
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





#ifndef estimateoffsetsmoduleFortTrans_h
#define estimateoffsetsmoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define allocate_locationAcrossOffset_f allocate_locationacrossoffset_
                        #define allocate_locationAcross_f allocate_locationacross_
                        #define allocate_locationDownOffset_f allocate_locationdownoffset_
                        #define allocate_locationDown_f allocate_locationdown_
                        #define allocate_snrRet_f allocate_snrret_
                        #define deallocate_locationAcrossOffset_f deallocate_locationacrossoffset_
                        #define deallocate_locationAcross_f deallocate_locationacross_
                        #define deallocate_locationDownOffset_f deallocate_locationdownoffset_
                        #define deallocate_locationDown_f deallocate_locationdown_
                        #define deallocate_snrRet_f deallocate_snrret_
                        #define getLocationAcrossOffset_f getlocationacrossoffset_
                        #define getLocationAcross_f getlocationacross_
                        #define getLocationDownOffset_f getlocationdownoffset_
                        #define getLocationDown_f getlocationdown_
                        #define getSNR_f getsnr_
                        #define estimateoffsets_f estimateoffsets_
                        #define setAcrossGrossOffset_f setacrossgrossoffset_
                        #define setDebugFlag_f setdebugflag_
                        #define setDownGrossOffset_f setdowngrossoffset_
                        #define setFileLength1_f setfilelength1_
                        #define setFileLength2_f setfilelength2_
                        #define setFirstPRF_f setfirstprf_
                        #define setFirstSampleAcross_f setfirstsampleacross_
                        #define setFirstSampleDown_f setfirstsampledown_
                        #define setLastSampleAcross_f setlastsampleacross_
                        #define setLastSampleDown_f setlastsampledown_
                        #define setLineLength1_f setlinelength1_
                        #define setLineLength2_f setlinelength2_
                        #define setNumberLocationAcross_f setnumberlocationacross_
                        #define setNumberLocationDown_f setnumberlocationdown_
                        #define setSecondPRF_f setsecondprf_
                        #define setWindowSize_f setwindowsize_
                        #define setSearchWindowSize_f setsearchwindowsize_
                        #define setZoomWindowSize_f setzoomwindowsize_
                        #define setOversamplingFactor_f setoversamplingfactor_
                        #define setIsComplex1_f setiscomplex1_
                        #define setIsComplex2_f setiscomplex2_
                        #define setBand1_f setband1_
                        #define setBand2_f setband2_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //estimateoffsetsmoduleFortTrans_h
