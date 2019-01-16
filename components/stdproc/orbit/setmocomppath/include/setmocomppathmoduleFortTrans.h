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





#ifndef setmocomppathmoduleFortTrans_h
#define setmocomppathmoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define allocate_vxyz1_f allocate_vxyz1_
                        #define allocate_vxyz2_f allocate_vxyz2_
                        #define allocate_xyz1_f allocate_xyz1_
                        #define allocate_xyz2_f allocate_xyz2_
                        #define deallocate_vxyz1_f deallocate_vxyz1_
                        #define deallocate_vxyz2_f deallocate_vxyz2_
                        #define deallocate_xyz1_f deallocate_xyz1_
                        #define deallocate_xyz2_f deallocate_xyz2_
                        #define getFirstAverageHeight_f getfirstaverageheight_
                        #define getFirstProcVelocity_f getfirstprocvelocity_
                        #define getPegHeading_f getpegheading_
                        #define getPegLatitude_f getpeglatitude_
                        #define getPegLongitude_f getpeglongitude_
                        #define getPegRadiusOfCurvature_f getpegradiusofcurvature_
                        #define getSecondAverageHeight_f getsecondaverageheight_
                        #define getSecondProcVelocity_f getsecondprocvelocity_
                        #define setEllipsoidEccentricitySquared_f setellipsoideccentricitysquared_
                        #define setEllipsoidMajorSemiAxis_f setellipsoidmajorsemiaxis_
                        #define setFirstPosition_f setfirstposition_
                        #define setFirstVelocity_f setfirstvelocity_
                        #define setStdWriter_f setstdwriter_
                        #define setPlanetGM_f setplanetgm_
                        #define setSecondPosition_f setsecondposition_
                        #define setSecondVelocity_f setsecondvelocity_
                        #define setmocomppath_f setmocomppath_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //setmocomppathmoduleFortTrans_h
