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





#ifndef correctmoduleFortTrans_h
#define correctmoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define allocate_midpoint_f allocate_midpoint_
                        #define allocate_mocbaseArray_f allocate_mocbasearray_
                        #define allocate_s1sch_f allocate_s1sch_
                        #define allocate_s2sch_f allocate_s2sch_
                        #define allocate_s_mocompArray_f allocate_s_mocomparray_
                        #define allocate_smsch_f allocate_smsch_
                        #define correct_f correct_
                        #define deallocate_midpoint_f deallocate_midpoint_
                        #define deallocate_mocbaseArray_f deallocate_mocbasearray_
                        #define deallocate_s1sch_f deallocate_s1sch_
                        #define deallocate_s2sch_f deallocate_s2sch_
                        #define deallocate_s_mocompArray_f deallocate_s_mocomparray_
                        #define deallocate_smsch_f deallocate_smsch_
                        #define setBodyFixedVelocity_f setbodyfixedvelocity_
                        #define setEllipsoidEccentricitySquared_f setellipsoideccentricitysquared_
                        #define setEllipsoidMajorSemiAxis_f setellipsoidmajorsemiaxis_
                        #define setISMocomp_f setismocomp_
                        #define setLength_f setlength_
                        #define setMidpoint_f setmidpoint_
                        #define setMocompBaseline_f setmocompbaseline_
                        #define setNumberAzimuthLooks_f setnumberazimuthlooks_
                        #define setNumberRangeLooks_f setnumberrangelooks_
                        #define setPRF_f setprf_
                        #define setPegHeading_f setpegheading_
                        #define setPegLatitude_f setpeglatitude_
                        #define setPegLongitude_f setpeglongitude_
                        #define setPlanetLocalRadius_f setplanetlocalradius_
                        #define setRadarWavelength_f setradarwavelength_
                        #define setRangeFirstSample_f setrangefirstsample_
                        #define setRangePixelSpacing_f setrangepixelspacing_
                        #define setReferenceOrbit_f setreferenceorbit_
                        #define setSc_f setsc_
                        #define setSch1_f setsch1_
                        #define setSch2_f setsch2_
                        #define setSpacecraftHeight_f setspacecraftheight_
                        #define setWidth_f setwidth_
                        #define setLookSide_f setlookside_
                        #define setDopCoeff_f setdopcoeff_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //correctmoduleFortTrans_h
