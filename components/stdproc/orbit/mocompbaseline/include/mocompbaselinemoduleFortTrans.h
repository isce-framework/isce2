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





#ifndef mocompbaselinemoduleFortTrans_h
#define mocompbaselinemoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
            #define setStdWriter_f setstdwriter_
                        #define allocate_baselineArray_f allocate_baselinearray_
                        #define allocate_is1_f allocate_is1_
                        #define allocate_is2_f allocate_is2_
                        #define allocate_s1_f allocate_s1_
                        #define allocate_s2_f allocate_s2_
                        #define allocate_sch1_f allocate_sch1_
                        #define allocate_sch2_f allocate_sch2_
                        #define deallocate_baselineArray_f deallocate_baselinearray_
                        #define deallocate_is1_f deallocate_is1_
                        #define deallocate_is2_f deallocate_is2_
                        #define deallocate_s1_f deallocate_s1_
                        #define deallocate_s2_f deallocate_s2_
                        #define deallocate_sch1_f deallocate_sch1_
                        #define deallocate_sch2_f deallocate_sch2_
                        #define getBaseline_f getbaseline_
                        #define mocompbaseline_f mocompbaseline_
                        #define setEllipsoidEccentricitySquared_f setellipsoideccentricitysquared_
                        #define setEllipsoidMajorSemiAxis_f setellipsoidmajorsemiaxis_
                        #define setMocompPosition1_f setmocompposition1_
                        #define setMocompPosition2_f setmocompposition2_
                        #define setMocompPositionIndex1_f setmocomppositionindex1_
                        #define setMocompPositionIndex2_f setmocomppositionindex2_
                        #define setPlanetLocalRadius_f setplanetlocalradius_
                        #define setPegHeading_f setpegheading_
                        #define setPegLatitude_f setpeglatitude_
                        #define setPegLongitude_f setpeglongitude_
                        #define setHeight_f setheight_
                        #define setSchPosition1_f setschposition1_
                        #define setSchPosition2_f setschposition2_
                        #define allocate_baselineArray1_f allocate_baselinearray1_
                        #define allocate_baselineArray2_f allocate_baselinearray2_
                        #define allocate_midPointArray1_f allocate_midpointarray1_
                        #define allocate_midPointArray2_f allocate_midpointarray2_
                        #define allocate_midPointArray_f allocate_midpointarray_
                        #define allocate_scArray_f allocate_scarray_
                        #define allocate_schArray_f allocate_scharray_
                        #define deallocate_baselineArray1_f deallocate_baselinearray1_
                        #define deallocate_baselineArray2_f deallocate_baselinearray2_
                        #define deallocate_midPointArray1_f deallocate_midpointarray1_
                        #define deallocate_midPointArray2_f deallocate_midpointarray2_
                        #define deallocate_midPointArray_f deallocate_midpointarray_
                        #define deallocate_scArray_f deallocate_scarray_
                        #define deallocate_schArray_f deallocate_scharray_
                        #define getBaseline1_f getbaseline1_
                        #define getBaseline2_f getbaseline2_
                        #define getMidpoint1_f getmidpoint1_
                        #define getMidpoint2_f getmidpoint2_
                        #define getMidpoint_f getmidpoint_
                        #define getSc_f getsc_
                        #define get_dim1_s1_f get_dim1_s1_
                        #define getSch_f getsch_

                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //mocompbaselinemoduleFortTrans_h
