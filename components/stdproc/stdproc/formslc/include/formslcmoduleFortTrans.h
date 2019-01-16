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





#ifndef formslcmoduleFortTrans_h
#define formslcmoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define setStdWriter_f setstdwriter_
                        #define allocate_dopplerCoefficients_f allocate_dopplercoefficients_
                        #define allocate_sch_f allocate_sch_
                        #define allocate_vsch_f allocate_vsch_
                        #define allocate_time_f allocate_time_
                        #define deallocate_dopplerCoefficients_f deallocate_dopplercoefficients_
                        #define deallocate_sch_f deallocate_sch_
                        #define deallocate_vsch_f deallocate_vsch_
                        #define deallocate_time_f deallocate_time_
                        #define formslc_f formslc_
                        #define getMocompIndex_f getmocompindex_
                        #define getMocompPosition_f getmocompposition_
                        #define getMocompPositionSize_f getmocomppositionsize_
                        #define setAzimuthPatchSize_f setazimuthpatchsize_
                        #define setAzimuthResolution_f setazimuthresolution_
                        #define setBodyFixedVelocity_f setbodyfixedvelocity_
                        #define setCaltoneLocation_f setcaltonelocation_
                        #define setChirpSlope_f setchirpslope_
                        #define setDebugFlag_f setdebugflag_
                        #define setDeskewFlag_f setdeskewflag_
                        #define setDopplerCentroidCoefficients_f setdopplercentroidcoefficients_
                        #define setEllipsoid_f setellipsoid_
                        #define setFirstLine_f setfirstline_
                        #define setFirstSample_f setfirstsample_
                        #define setIMMocomp_f setimmocomp_
                        #define setIMRC1_f setimrc1_
                        #define setIMRCAS1_f setimrcas1_
                        #define setIMRCRM1_f setimrcrm1_
                        #define setInPhaseValue_f setinphasevalue_
                        #define setIQFlip_f setiqflip_
                        #define setNumberAzimuthLooks_f setnumberazimuthlooks_
                        #define setNumberBytesPerLine_f setnumberbytesperline_
                        #define setNumberGoodBytes_f setnumbergoodbytes_
                        #define setNumberPatches_f setnumberpatches_
                        #define setNumberRangeBin_f setnumberrangebin_
                        #define setNumberValidPulses_f setnumbervalidpulses_
                        #define setOverlap_f setoverlap_
                        #define setPlanetLocalRadius_f setplanetlocalradius_
                        #define setPosition_f setposition_
                        #define setVelocity_f setvelocity_
                        #define setPegPoint_f setpegpoint_
                        #define setPlanet_f setplanet_
                        #define setPRF_f setprf_
                        #define setQuadratureValue_f setquadraturevalue_
                        #define setRadarWavelength_f setradarwavelength_
                        #define setRanfftiq_f setranfftiq_
                        #define setRanfftov_f setranfftov_
                        #define setRangeChirpExtensionPoints_f setrangechirpextensionpoints_
                        #define setRangeFirstSample_f setrangefirstsample_
                        #define setRangePulseDuration_f setrangepulseduration_
                        #define setRangeSamplingRate_f setrangesamplingrate_
                        #define setRangeSpectralWeighting_f setrangespectralweighting_
                        #define setSecondaryRangeMigrationFlag_f setsecondaryrangemigrationflag_
                        #define setSpacecraftHeight_f setspacecraftheight_
                        #define setSpectralShiftFraction_f setspectralshiftfraction_
                        #define setStartRangeBin_f setstartrangebin_
                        #define setTime_f settime_
                        #define setTransDat_f settransdat_
                        #define setSlcWidth_f setslcwidth_
                        #define getStartingRange_f getstartingrange_
                        #define setStartingRange_f setstartingrange_
                        #define setLookSide_f setlookside_
                        #define setShift_f setshift_
                        #define setOrbit_f setorbit_
                        #define setSensingStart_f setsensingstart_
                        #define setMocompOrbit_f setmocomporbit_
                        #define getSlcSensingStart_f getslcsensingstart_
                        #define getMocompRange_f getmocomprange_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //formslcmoduleFortTrans_h
