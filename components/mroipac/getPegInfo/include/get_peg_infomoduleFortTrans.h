//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef get_peg_infomoduleFortTrans_h
#define get_peg_infomoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define allocate_r_af_f allocate_r_af_
                        #define allocate_r_afdot_f allocate_r_afdot_
                        #define allocate_r_axyz1_f allocate_r_axyz1_
                        #define allocate_r_cf_f allocate_r_cf_
                        #define allocate_r_cfdot_f allocate_r_cfdot_
                        #define allocate_r_intPos_f allocate_r_intpos_
                        #define allocate_r_intVel_f allocate_r_intvel_
                        #define allocate_r_platacc_f allocate_r_platacc_
                        #define allocate_r_platvel_f allocate_r_platvel_
                        #define allocate_r_sfdot_f allocate_r_sfdot_
                        #define allocate_r_time_f allocate_r_time_
                        #define allocate_r_transVect_f allocate_r_transvect_
                        #define allocate_r_transfMat_f allocate_r_transfmat_
                        #define allocate_r_vxyz1_f allocate_r_vxyz1_
                        #define allocate_r_vxyzpeg_f allocate_r_vxyzpeg_
                        #define allocate_r_xyz1_f allocate_r_xyz1_
                        #define deallocate_r_af_f deallocate_r_af_
                        #define deallocate_r_afdot_f deallocate_r_afdot_
                        #define deallocate_r_axyz1_f deallocate_r_axyz1_
                        #define deallocate_r_cf_f deallocate_r_cf_
                        #define deallocate_r_cfdot_f deallocate_r_cfdot_
                        #define deallocate_r_intPos_f deallocate_r_intpos_
                        #define deallocate_r_intVel_f deallocate_r_intvel_
                        #define deallocate_r_platacc_f deallocate_r_platacc_
                        #define deallocate_r_platvel_f deallocate_r_platvel_
                        #define deallocate_r_sfdot_f deallocate_r_sfdot_
                        #define deallocate_r_time_f deallocate_r_time_
                        #define deallocate_r_transVect_f deallocate_r_transvect_
                        #define deallocate_r_transfMat_f deallocate_r_transfmat_
                        #define deallocate_r_vxyz1_f deallocate_r_vxyz1_
                        #define deallocate_r_vxyzpeg_f deallocate_r_vxyzpeg_
                        #define deallocate_r_xyz1_f deallocate_r_xyz1_
                        #define getAlongTrackVelocityFit_f getalongtrackvelocityfit_
                        #define getCrossTrackVelocityFit_f getcrosstrackvelocityfit_
                        #define getGroundSpacing_f getgroundspacing_
                        #define getHorizontalFit_f gethorizontalfit_
                        #define getIntPosition_f getintposition_
                        #define getIntVelocity_f getintvelocity_
                        #define getPegHeading_f getpegheading_
                        #define getPegHeight_f getpegheight_
                        #define getPegLat_f getpeglat_
                        #define getPegLon_f getpeglon_
                        #define getPegRadius_f getpegradius_
                        #define getPegVelocity_f getpegvelocity_
                        #define getPlatformSCHAcceleration_f getplatformschacceleration_
                        #define getPlatformSCHVelocity_f getplatformschvelocity_
                        #define getTimeFirstScene_f gettimefirstscene_
                        #define getTransformationMatrix_f gettransformationmatrix_
                        #define getTranslationVector_f gettranslationvector_
                        #define getVerticalFit_f getverticalfit_
                        #define getVerticalVelocityFit_f getverticalvelocityfit_
                        #define get_peg_info_f get_peg_info_
                        #define setAccelerationVector_f setaccelerationvector_
                        #define setNumAzimuthLooksInt_f setnumazimuthlooksint_
                        #define setNumLinesInt_f setnumlinesint_
                        #define setNumLinesSlc_f setnumlinesslc_
                        #define setNumObservations_f setnumobservations_
                        #define setPlanetGM_f setplanetgm_
                        #define setPlanetSpinRate_f setplanetspinrate_
                        #define setPositionVector_f setpositionvector_
                        #define setPrfSlc_f setprfslc_
                        #define setStartLineSlc_f setstartlineslc_
                        #define setTimeSlc_f settimeslc_
                        #define setTime_f settime_
                        #define setVelocityVector_f setvelocityvector_
                #else
                        #error Unknown traslation for FORTRAN external symbols
                #endif

        #endif

#endif //get_peg_infomoduleFortTrans_h
