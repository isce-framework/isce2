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





#ifndef resampmoduleFortTrans_h
#define resampmoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
            #define setStdWriter_f setstdwriter_
                        #define allocate_acrossOffset_f allocate_acrossoffset_
                        #define allocate_dopplerCoefficients_f allocate_dopplercoefficients_
                        #define allocate_downOffset_f allocate_downoffset_
                        #define allocate_r_azoff2_f allocate_r_azoff2_
                        #define allocate_r_azoff_f allocate_r_azoff_
                        #define allocate_r_azpos2_f allocate_r_azpos2_
                        #define allocate_r_azpos_f allocate_r_azpos_
                        #define allocate_r_ranoff2_f allocate_r_ranoff2_
                        #define allocate_r_ranoff_f allocate_r_ranoff_
                        #define allocate_r_ranpos2_f allocate_r_ranpos2_
                        #define allocate_r_ranpos_f allocate_r_ranpos_
                        #define allocate_r_sig2_f allocate_r_sig2_
                        #define allocate_r_sig_f allocate_r_sig_
                        #define deallocate_acrossOffset_f deallocate_acrossoffset_
                        #define deallocate_dopplerCoefficients_f deallocate_dopplercoefficients_
                        #define deallocate_downOffset_f deallocate_downoffset_
                        #define deallocate_r_azoff2_f deallocate_r_azoff2_
                        #define deallocate_r_azoff_f deallocate_r_azoff_
                        #define deallocate_r_azpos2_f deallocate_r_azpos2_
                        #define deallocate_r_azpos_f deallocate_r_azpos_
                        #define deallocate_r_ranoff2_f deallocate_r_ranoff2_
                        #define deallocate_r_ranoff_f deallocate_r_ranoff_
                        #define deallocate_r_ranpos2_f deallocate_r_ranpos2_
                        #define deallocate_r_ranpos_f deallocate_r_ranpos_
                        #define deallocate_r_sig2_f deallocate_r_sig2_
                        #define deallocate_r_sig_f deallocate_r_sig_
                        #define getLocationAcrossOffset_f getlocationacrossoffset_
                        #define getLocationDownOffset_f getlocationdownoffset_
                        #define resamp_f resamp_
                        #define setDopplerCentroidCoefficients_f setdopplercentroidcoefficients_
                        #define setFirstLineOffset_f setfirstlineoffset_
                        #define setFlattenWithOffsetFitFlag_f setflattenwithoffsetfitflag_
                        #define setLocationAcross1_f setlocationacross1_
                        #define setLocationAcross2_f setlocationacross2_
                        #define setLocationAcrossOffset1_f setlocationacrossoffset1_
                        #define setLocationAcrossOffset2_f setlocationacrossoffset2_
                        #define setLocationDown1_f setlocationdown1_
                        #define setLocationDown2_f setlocationdown2_
                        #define setLocationDownOffset1_f setlocationdownoffset1_
                        #define setLocationDownOffset2_f setlocationdownoffset2_
                        #define setNumberAzimuthLooks_f setnumberazimuthlooks_
                        #define setNumberFitCoefficients_f setnumberfitcoefficients_
                        #define setNumberLines_f setnumberlines_
                        #define setNumberLinesImage2_f setnumberlinesimage2_
                        #define setNumberRangeBin1_f setnumberrangebin1_
                        #define setNumberRangeBin2_f setnumberrangebin2_
                        #define setNumberRangeLooks_f setnumberrangelooks_
                        #define setRadarWavelength_f setradarwavelength_
                        #define setSNR1_f setsnr1_
                        #define setSNR2_f setsnr2_
                        #define setSlantRangePixelSpacing_f setslantrangepixelspacing_
                        #define setStartLine_f setstartline_
                #else
                        #error Unknown translation for FORTRAN external symbols
                #endif

        #endif

#endif //resampmoduleFortTrans_h
