//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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





#ifndef offoutliersmoduleFortTrans_h
#define offoutliersmoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
            #define setStdWriter_f setstdwriter_
                        #define allocate_acshift_f allocate_acshift_
                        #define allocate_dnshift_f allocate_dnshift_
                        #define allocate_indexArray_f allocate_indexarray_
                        #define allocate_s_f allocate_s_
                        #define allocate_sig_f allocate_sig_
                        #define allocate_xd_f allocate_xd_
                        #define allocate_yd_f allocate_yd_
                        #define deallocate_acshift_f deallocate_acshift_
                        #define deallocate_dnshift_f deallocate_dnshift_
                        #define deallocate_indexArray_f deallocate_indexarray_
                        #define deallocate_s_f deallocate_s_
                        #define deallocate_sig_f deallocate_sig_
                        #define deallocate_xd_f deallocate_xd_
                        #define deallocate_yd_f deallocate_yd_
                        #define getAverageOffsetAcross_f getaverageoffsetacross_
                        #define getAverageOffsetDown_f getaverageoffsetdown_
                        #define getIndexArraySize_f getindexarraysize_
                        #define getIndexArray_f getindexarray_
                        #define offoutliers_f offoutliers_
                        #define setDistance_f setdistance_
                        #define setLocationAcrossOffset_f setlocationacrossoffset_
                        #define setLocationAcross_f setlocationacross_
                        #define setLocationDownOffset_f setlocationdownoffset_
                        #define setLocationDown_f setlocationdown_
                        #define setNumberOfPoints_f setnumberofpoints_
                        #define setSNR_f setsnr_
                        #define setSign_f setsign_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //offoutliersmoduleFortTrans_h
