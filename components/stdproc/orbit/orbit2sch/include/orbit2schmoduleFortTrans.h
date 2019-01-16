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





#ifndef orbit2schmoduleFortTrans_h
#define orbit2schmoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
            #define setStdWriter_f setstdwriter_
                        #define allocate_asch_f allocate_asch_
                        #define allocate_sch_f allocate_sch_
                        #define allocate_vsch_f allocate_vsch_
                        #define allocate_vxyz_f allocate_vxyz_
                        #define allocate_xyz_f allocate_xyz_
                        #define deallocate_asch_f deallocate_asch_
                        #define deallocate_sch_f deallocate_sch_
                        #define deallocate_vsch_f deallocate_vsch_
                        #define deallocate_vxyz_f deallocate_vxyz_
                        #define deallocate_xyz_f deallocate_xyz_
                        #define getSchGravitationalAcceleration_f getschgravitationalacceleration_
                        #define getSchPosition_f getschposition_
                        #define getSchVelocity_f getschvelocity_
                        #define orbit2sch_f orbit2sch_
                        #define setAverageHeight_f setaverageheight_
                        #define setComputePegInfoFlag_f setcomputepeginfoflag_
                        #define setEllipsoidEccentricitySquared_f setellipsoideccentricitysquared_
                        #define setEllipsoidMajorSemiAxis_f setellipsoidmajorsemiaxis_
                        #define setOrbitPosition_f setorbitposition_
                        #define setOrbitVelocity_f setorbitvelocity_
                        #define setPegHeading_f setpegheading_
                        #define setPegLatitude_f setpeglatitude_
                        #define setPegLongitude_f setpeglongitude_
                        #define setPlanetGM_f setplanetgm_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //orbit2schmoduleFortTrans_h
