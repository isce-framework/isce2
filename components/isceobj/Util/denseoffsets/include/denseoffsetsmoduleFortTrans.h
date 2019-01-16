//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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
// Author: Piyush Agram
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





#ifndef denseoffsetsmoduleFortTrans_h
#define denseoffsetsmoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define denseoffsets_f denseoffsets_
                        #define setAcrossGrossOffset_f setacrossgrossoffset_
                        #define setDebugFlag_f setdebugflag_
                        #define setDownGrossOffset_f setdowngrossoffset_
                        #define setFileLength1_f setfilelength1_
                        #define setFileLength2_f setfilelength2_
                        #define setScaleFactorX_f setscalefactorx_
                        #define setFirstSampleAcross_f setfirstsampleacross_
                        #define setFirstSampleDown_f setfirstsampledown_
                        #define setLastSampleAcross_f setlastsampleacross_
                        #define setLastSampleDown_f setlastsampledown_
                        #define setLineLength1_f setlinelength1_
                        #define setLineLength2_f setlinelength2_
                        #define setSkipSampleAcross_f setskipsampleacross_
                        #define setSkipSampleDown_f setskipsampledown_
                        #define setScaleFactorY_f setscalefactory_
                        #define setWindowSizeWidth_f setwindowsizewidth_
                        #define setWindowSizeHeight_f setwindowsizeheight_
                        #define setSearchWindowSizeWidth_f setsearchwindowsizewidth_
                        #define setSearchWindowSizeHeight_f setsearchwindowsizeheight_
                        #define setZoomWindowSize_f setzoomwindowsize_
                        #define setOversamplingFactor_f setoversamplingfactor_
                        #define setIsComplex1_f setiscomplex1_
                        #define setIsComplex2_f setiscomplex2_
                        #define setBand1_f setband1_
                        #define setBand2_f setband2_
                        #define setNormalizeFlag_f setnormalizeflag_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //denseoffsetsmoduleFortTrans_h
