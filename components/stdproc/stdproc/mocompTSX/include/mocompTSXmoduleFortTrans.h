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





#ifndef mocompTSXmoduleFortTrans_h
#define mocompTSXmoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define allocate_dopplerCentroidCoefficients_f allocate_dopplercentroidcoefficients_
                        #define allocate_sch_f allocate_sch_
                        #define allocate_time_f allocate_time_
                        #define deallocate_dopplerCentroidCoefficients_f deallocate_dopplercentroidcoefficients_
                        #define deallocate_sch_f deallocate_sch_
                        #define deallocate_time_f deallocate_time_
                        #define getMocompIndex_f getmocompindex_
                        #define getMocompPositionSize_f getmocomppositionsize_
                        #define getMocompPosition_f getmocompposition_
                        #define mocompTSX_f mocomptsx_
                        #define setBodyFixedVelocity_f setbodyfixedvelocity_
                        #define setDopplerCentroidCoefficients_f setdopplercentroidcoefficients_
                        #define setNumberAzLines_f setnumberazlines_
                        #define setNumberRangeBins_f setnumberrangebins_
                        #define setPRF_f setprf_
                        #define setPlanetLocalRadius_f setplanetlocalradius_
                        #define setPosition_f setposition_
                        #define setRadarWavelength_f setradarwavelength_
                        #define setRangeFisrtSample_f setrangefisrtsample_
                        #define setRangeSamplingRate_f setrangesamplingrate_
                        #define setSpacecraftHeight_f setspacecraftheight_
                        #define setStdWriter_f setstdwriter_
                        #define setTime_f settime_
                        #define setVelocity_f setvelocity_
                        #define setLookSide_f setlookside_
                        #define getStartingRange_f getstartingrange_
                        #define setOrbit_f setorbit_
                        #define setMocompOrbit_f setmocomporbit_
                        #define setPlanet_f setplanet_
                        #define setPegPoint_f setpegpoint_
                        #define setSensingStart_f setsensingstart_
                        #define getSlcSensingStart_f getslcsensingstart_
                        #define getMocompRange_f getmocomprange_
                        #define setEllipsoid_f setellipsoid_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //mocompTSXmoduleFortTrans_h
