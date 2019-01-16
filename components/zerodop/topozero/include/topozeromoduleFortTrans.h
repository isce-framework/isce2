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





#ifndef topozeromoduleFortTrans_h
#define topozeromoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define getMaximumLatitude_f getmaximumlatitude_
                        #define getMaximumLongitude_f getmaximumlongitude_
                        #define getMinimumLatitude_f getminimumlatitude_
                        #define getMinimumLongitude_f getminimumlongitude_
                        #define setDeltaLatitude_f setdeltalatitude_
                        #define setDeltaLongitude_f setdeltalongitude_
                        #define setDemLength_f setdemlength_
                        #define setDemWidth_f setdemwidth_
                        #define setEllipsoidEccentricitySquared_f setellipsoideccentricitysquared_
                        #define setEllipsoidMajorSemiAxis_f setellipsoidmajorsemiaxis_
                        #define setFirstLatitude_f setfirstlatitude_
                        #define setFirstLongitude_f setfirstlongitude_
                        #define setHeightPointer_f setheightpointer_
                        #define setLatitudePointer_f setlatitudepointer_
                        #define setLength_f setlength_
                        #define setLongitudePointer_f setlongitudepointer_
                        #define setLosPointer_f setlospointer_
                        #define setIncPointer_f setincpointer_
                        #define setMaskPointer_f setmaskpointer_
                        #define setNumberAzimuthLooks_f setnumberazimuthlooks_
                        #define setNumberIterations_f setnumberiterations_
                        #define setNumberRangeLooks_f setnumberrangelooks_
                        #define setPRF_f setprf_
                        #define setSensingStart_f setsensingstart_
                        #define setPegHeading_f setpegheading_
                        #define setRadarWavelength_f setradarwavelength_
                        #define setRangeFirstSample_f setrangefirstsample_
                        #define setRangePixelSpacing_f setrangepixelspacing_
                        #define setOrbit_f setorbit_
                        #define setWidth_f setwidth_
                        #define setLookSide_f setlookside_
                        #define topo_f topo_
                        #define setSecondaryIterations_f setsecondaryiterations_
                        #define setThreshold_f setthreshold_
                        #define setMethod_f setmethod_
                        #define setOrbitMethod_f setorbitmethod_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //topozeromoduleFortTrans_h
