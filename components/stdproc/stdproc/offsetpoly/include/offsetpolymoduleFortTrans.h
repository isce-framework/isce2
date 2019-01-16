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





#ifndef offsetpolymoduleFortTrans_h
#define offsetpolymoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define allocateFieldArrays_f allocatefieldarrays_
                        #define deallocateFieldArrays_f deallocatefieldarrays_
                        #define allocatePolyArray_f allocatepolyarray_
                        #define deallocatePolyArray_f deallocatepolyarray_
                        #define getOffsetPoly_f getoffsetpoly_
                        #define offsetpoly_f offsetpoly_
                        #define setLocationAcross_f setlocationacross_
                        #define setOffset_f setoffset_
                        #define setLocationDown_f setlocationdown_
                        #define setSNR_f setsnr_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //offsetpolymoduleFortTrans_h
