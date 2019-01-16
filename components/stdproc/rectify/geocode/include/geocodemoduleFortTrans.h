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





#ifndef geocodemoduleFortTrans_h
#define geocodemoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
            #define setStdWriter_f setstdwriter_
                        #define allocate_s_mocomp_f allocate_s_mocomp_
                        #define deallocate_s_mocomp_f deallocate_s_mocomp_
                        #define geocode_f geocode_
                        #define getGeoLength_f getgeolength_
                        #define getGeoWidth_f getgeowidth_
                        #define getLatitudeSpacing_f getlatitudespacing_
                        #define getLongitudeSpacing_f getlongitudespacing_
                        #define getMaximumGeoLatitude_f getmaximumgeolatitude_
                        #define getMaxmumGeoLongitude_f getmaxmumgeolongitude_
                        #define getMinimumGeoLatitude_f getminimumgeolatitude_
                        #define getMinimumGeoLongitude_f getminimumgeolongitude_
                        #define setDeltaLatitude_f setdeltalatitude_
                        #define setDeltaLongitude_f setdeltalongitude_
                        #define setDemLength_f setdemlength_
                        #define setDemWidth_f setdemwidth_
                        #define setLookSide_f setlookside_
                        #define setDopplerAccessor_f setdoppleraccessor_
                        #define setEllipsoidEccentricitySquared_f setellipsoideccentricitysquared_
                        #define setEllipsoidMajorSemiAxis_f setellipsoidmajorsemiaxis_
                        #define setFirstLatitude_f setfirstlatitude_
                        #define setFirstLongitude_f setfirstlongitude_
                        #define setHeight_f setheight_
                        #define setISMocomp_f setismocomp_
                        #define setLength_f setlength_
                        #define setMaximumLatitude_f setmaximumlatitude_
                        #define setMaximumLongitude_f setmaximumlongitude_
                        #define setMinimumLatitude_f setminimumlatitude_
                        #define setMinimumLongitude_f setminimumlongitude_
                        #define setNumberAzimuthLooks_f setnumberazimuthlooks_
                        #define setNumberPointsPerDemPost_f setnumberpointsperdempost_
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
                        #define setSCoordinateFirstLine_f setscoordinatefirstline_
                        #define setVelocity_f setvelocity_
                        #define setWidth_f setwidth_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //geocodemoduleFortTrans_h
