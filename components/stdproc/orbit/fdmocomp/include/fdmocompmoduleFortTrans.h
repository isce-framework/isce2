//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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





#ifndef fdmocompmoduleFortTrans_h
#define fdmocompmoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define allocate_fdArray_f allocate_fdarray_
                        #define allocate_vsch_f allocate_vsch_
                        #define deallocate_fdArray_f deallocate_fdarray_
                        #define deallocate_vsch_f deallocate_vsch_
                        #define fdmocomp_f fdmocomp_
                        #define getCorrectedDoppler_f getcorrecteddoppler_
                        #define setDopplerCoefficients_f setdopplercoefficients_
                        #define setHeigth_f setheigth_
                        #define setPRF_f setprf_
                        #define setPlatformHeigth_f setplatformheigth_
                        #define setRadarWavelength_f setradarwavelength_
                        #define setRadiusOfCurvature_f setradiusofcurvature_
                        #define setRangeSamplingRate_f setrangesamplingrate_
                        #define setSchVelocity_f setschvelocity_
                        #define setStartingRange_f setstartingrange_
                        #define setWidth_f setwidth_
                        #define setLookSide_f setlookside_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //fdmocompmoduleFortTrans_h
