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





#ifndef topomoduleFortTrans_h
#define topomoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define allocate_s_mocompArray_f allocate_s_mocomparray_
                        #define allocate_squintshift_f allocate_squintshift_
                        #define deallocate_s_mocompArray_f deallocate_s_mocomparray_
                        #define deallocate_squintshift_f deallocate_squintshift_
                        #define getAzimuthSpacing_f getazimuthspacing_
                        #define getMaximumLatitude_f getmaximumlatitude_
                        #define getMaximumLongitude_f getmaximumlongitude_
                        #define getMinimumLatitude_f getminimumlatitude_
                        #define getMinimumLongitude_f getminimumlongitude_
                        #define getPlanetLocalRadius_f getplanetlocalradius_
                        #define getSCoordinateFirstLine_f getscoordinatefirstline_
                        #define getSCoordinateLastLine_f getscoordinatelastline_
                        #define getSquintShift_f getsquintshift_
                        #define setBodyFixedVelocity_f setbodyfixedvelocity_
                        #define setDeltaLatitude_f setdeltalatitude_
                        #define setDeltaLongitude_f setdeltalongitude_
                        #define setDemLength_f setdemlength_
                        #define setDemWidth_f setdemwidth_
                        #define setEllipsoidEccentricitySquared_f setellipsoideccentricitysquared_
                        #define setEllipsoidMajorSemiAxis_f setellipsoidmajorsemiaxis_
                        #define setFirstLatitude_f setfirstlatitude_
                        #define setFirstLongitude_f setfirstlongitude_
                        #define setHeightRPointer_f setheightrpointer_
                        #define setHeightSchPointer_f setheightschpointer_
                        #define setISMocomp_f setismocomp_
                        #define setLatitudePointer_f setlatitudepointer_
                        #define setLength_f setlength_
                        #define setLongitudePointer_f setlongitudepointer_
                        #define setLosPointer_f setlospointer_
                        #define setIncPointer_f setincpointer_
                        #define setNumberAzimuthLooks_f setnumberazimuthlooks_
                        #define setNumberIterations_f setnumberiterations_
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
                        #define setSpacecraftHeight_f setspacecraftheight_
                        #define setWidth_f setwidth_
                        #define setLookSide_f setlookside_
                        #define setMethod_f setmethod_
                        #define topo_f topo_
                        #define getLength_f getlength_
                        #define setSensingStart_f setsensingstart_
                        #define setOrbit_f setorbit_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //topomoduleFortTrans_h
