/* provides structures to read ALOS SAR tapes	*/
/* reall just a CEOS reader			*/

/*
include files were modified from the rceos.c programs
written by C. Tomassini & F. Lorenna

other format information from:
from CERS (RAW) CCT format specifications STD-TM#92-767F
   Canada Centre for Remote Sensing (CCRS)
   Surveys, Mapping and Remote Sensing Sector
   Energy, Mines and Resources Canada

Table 6.1.2.2 "SARLEADER" FILE POINTER RECORD CONTENTS
page 6.

	R. J. Mellors 
	July 1997, IGPP-SIO

from esa annex A (Document ER-IS-EPO-GS-5902.I)
     Issue 2.1.1

        Paul F. Jamason
        25-FEB-1997, IGPP-SIO

Modified to read ALOS format
Product Format Description
(PALSAR Level 1.0)

	R. J. Mellors
	June 2007, SDSU

6/1/07 SARLEADER_DSS_RCS_ALOS 
7th line changed to "%4c%4c%16c%16c%16c%16c%16c"\
(2 16c at end rather than 1 32c)
*/

/* ALOS raw data set summary record format */
#define SARLEADER_DSS_RCS_ALOS "%4c%4c%16c%32c%32c%16c%16c%16c%16c%16c%16c"\
"%16c%16c%16c%16c%16c%16c%16c%16c%8c%8c%16c%16c%16c%4c%4c%16c%32c%8c%8c"\
"%8c%8c%8c%8c%8c%16c%2c%16c%16c%16c%16c%16c%16c%16c%16c%16c%16c%16c%8c%8c"\
"%16c%16c%16c%4c%4c%32c%8c%12c%16c%16c%16c%32c%16c%16c%4c%16c%32c%16c%32c"\
"%8c%8c%16c%8c%8c%32c%32c%32c%16c%16c%16c%16c%16c%16c%32c%32c%16c%16c%16c"\
"%32c%16c%16c%16c%16c%16c%16c%16c%8c%8c%16c%16c%16c%16c%16c%16c%16c%16c%8c"\
"%4c%4c%16c%16c%16c%16c%16c"\
"%4c%8c%8c%8c%8c%4c%8c%16c%4c%4c%16c%4c%28c%120c%8c%8c%2048c%26c"

/* ALOS raw data set summary corresponding log file output */
#define SARLEADER_DSS_RVL_ALOS(SP)\
(SP)->dss_rec_seq_num,\
(SP)->chan_ind,\
(SP)->reserved1 ,\
(SP)->scene_number ,\
(SP)->input_scene_center_time,\
(SP)->spare1,\
(SP)->center_lat,\
(SP)->center_long,\
(SP)->center_heading,\
(SP)->ellipsoid_designator,\
(SP)->ellipsoid_semimajor_axis,\
(SP)->ellipsoid_semiminor_axis,\
(SP)->earth_constant,\
(SP)->spare2,\
(SP)->ellipsoid_j2,\
(SP)->ellipsoid_j3,\
(SP)->ellipsoid_j4,\
(SP)->spare,\
(SP)->reserved_new,\
(SP)->scene_centre_line_number,\
(SP)->scene_centre_pixel_number,\
(SP)->scene_length,\
(SP)->scene_width,\
(SP)->spare3,\
(SP)->nchan,\
(SP)->spare4,\
(SP)->mission_identifier,\
(SP)->sensor_id_and_mode,\
(SP)->orbit_number,\
(SP)->lat_nadir_center,\
(SP)->long_nadir_center,\
(SP)->heading_nadir_center,\
(SP)->clock_angle,\
(SP)->incidence_angle_center,\
(SP)->radar_freq,\
(SP)->radar_wavelength,\
(SP)->motion_compensation,\
(SP)->range_pulse_code_specifier,\
(SP)->range_pulse_amplitude_const,\
(SP)->range_pulse_amplitude_lin,\
(SP)->range_pulse_amplitude_quad,\
(SP)->range_pulse_amplitude_cube,\
(SP)->range_pulse_amplitude_quart,\
(SP)->range_pulse_phase_const,\
(SP)->range_pulse_phase_lin,\
(SP)->range_pulse_phase_quad,\
(SP)->range_pulse_phase_cube,\
(SP)->range_pulse_phase_quart,\
(SP)->chirp_extraction_index,\
(SP)->spare5,\
(SP)->sampling_rate,\
(SP)->range_gate_early_edge_start_image,\
(SP)->range_pulse_length,\
(SP)->reserved2,\
(SP)->range_compressed_flag,\
(SP)->reserved3,\
(SP)->quantisation_in_bits,\
(SP)->quantizer_descriptor,\
(SP)->dc_bias_i,\
(SP)->dc_bias_q,\
(SP)->gain_imbalance,\
(SP)->spare6,\
(SP)->reserved4,\
(SP)->antenna_mech_bor,\
(SP)->reserved5,\
(SP)->nominal_prf,\
(SP)->reserved6,\
(SP)->satelite_encoded_binary_time,\
(SP)->satelite_clock_time,\
(SP)->satelite_clock_increment,\
(SP)->spare7,\
(SP)->processing_facility_identifier,\
(SP)->processing_system_id,\
(SP)->processing_version_id,\
(SP)->reserved7,\
(SP)->product_type_id,\
(SP)->alg_id,\
(SP)->nlooks_az,\
(SP)->neff_looks_range,\
(SP)->bandwidth_look_az,\
(SP)->bandwidth_look_range,\
(SP)->total_look_bandwidth_az,\
(SP)->total_look_bandwidth_range,\
(SP)->w_func_designator_az,\
(SP)->w_func_designator_range,\
(SP)->data_input_source,\
(SP)->nom_res_3db_range,\
(SP)->nom_res_az,\
(SP)->reserved8,\
(SP)->a_track_dop_freq_const_early_image,\
(SP)->a_track_dop_freq_lin_early_image,\
(SP)->a_track_dop_freq_quad_early_image,\
(SP)->spare8,\
(SP)->c_track_dop_freq_const_early_image,\
(SP)->c_track_dop_freq_lin_early_image,\
(SP)->c_track_dop_freq_quad_early_image,\
(SP)->time_direction_along_pixel,\
(SP)->time_direction_along_line,\
(SP)->a_track_dop_freq_rate_const_early_image,\
(SP)->a_track_dop_freq_rate_lin_early_image,\
(SP)->a_track_dop_freq_rate_quad_early_image,\
(SP)->spare9,\
(SP)->c_track_dop_freq_rate_const_early_image,\
(SP)->c_track_dop_freq_rate_lin_early_image,\
(SP)->c_track_dop_freq_rate_quad_early_image,\
(SP)->spare10,\
(SP)->line_content_indicator,\
(SP)->clut_lock_flag,\
(SP)->autofocussing_flag,\
(SP)->line_spacing,\
(SP)->pixel_spacing_range,\
(SP)->range_compression_designator,\
(SP)->spare11,\
(SP)->spare12,\
(SP)->calibration_data_indicator,\
(SP)->start_line_upper_image,\
(SP)->stop_line_upper_image,\
(SP)->start_line_bottom_image,\
(SP)->stop_line_bottom_image,\
(SP)->PRF_switch,\
(SP)->PRF_switch_line,\
(SP)->spare13,\
(SP)->yaw_steering_mode,\
(SP)->parameter_table,\
(SP)->nom_offnadir_angle,\
(SP)->antenna_beam_number,\
(SP)->spare14,\
(SP)->spare15,\
(SP)->num_anno_points,\
(SP)->spare16,\
(SP)->image_annotation,\
(SP)->spare17

struct sarleader_dss_ALOS {
	char    dss_rec_seq_num[4];   	/*dss record sequence number (1)*/
	char    chan_ind[4];            /*sar channel indicator (1)*/
	char    reserved1[16] ;         /* scene identifier*/
	char    scene_number[32] ;
	char    input_scene_center_time[32];
	char    spare1[16];
	char    center_lat[16];
	char    center_long[16];
	char    center_heading[16];
	char    ellipsoid_designator[16];
	char    ellipsoid_semimajor_axis[16];
	char    ellipsoid_semiminor_axis[16];
	char    earth_constant[16];
	char    spare2[16];
	char    ellipsoid_j2[16];
	char    ellipsoid_j3[16];
	char    ellipsoid_j4[16];
	char    spare[16];
	char    reserved_new[16];
	char    scene_centre_line_number[8];
	char    scene_centre_pixel_number[8];
	char    scene_length[16];
	char    scene_width[16];
	char    spare3[16];
	char    nchan[4];
	char    spare4[4];
	char    mission_identifier[16];
	char    sensor_id_and_mode[32];
	char    orbit_number[8];
	char    lat_nadir_center[8];
	char    long_nadir_center[8];
	char    heading_nadir_center[8];
	char    clock_angle[8];
	char    incidence_angle_center[8];
	char    radar_freq[8];
	char    radar_wavelength[16];
	char    motion_compensation[2];
	char    range_pulse_code_specifier[16];
	char    range_pulse_amplitude_const[16];
	char    range_pulse_amplitude_lin[16];
	char    range_pulse_amplitude_quad[16];
	char    range_pulse_amplitude_cube[16];
	char    range_pulse_amplitude_quart[16];
	char    range_pulse_phase_const[16];
	char    range_pulse_phase_lin[16];
	char    range_pulse_phase_quad[16];
	char    range_pulse_phase_cube[16];
	char    range_pulse_phase_quart[16];
	char    chirp_extraction_index[8];
	char    spare5[8];
	char    sampling_rate[16];
	char    range_gate_early_edge_start_image[16];
	char    range_pulse_length[16];
	char    reserved2[4];
	char    range_compressed_flag[4];
	char    reserved3[32];
	char    quantisation_in_bits[8];
	char    quantizer_descriptor[12];
	char    dc_bias_i[16];
	char    dc_bias_q[16];
	char    gain_imbalance[16];
	char    spare6[32];
	char    reserved4[16];
	char    antenna_mech_bor[16];
	char    reserved5[4];
	char    nominal_prf[16];
	char    reserved6[32];
	char    satelite_encoded_binary_time[16];
	char    satelite_clock_time[32];
	char    satelite_clock_increment[8];
	char    spare7[8];
	char    processing_facility_identifier[16];
	char    processing_system_id[8];
	char    processing_version_id[8];
	char    reserved7[32];
	char    product_type_id[32];
	char    alg_id[32];
	char    nlooks_az[16];
	char    neff_looks_range[16];
	char    bandwidth_look_az[16];
	char    bandwidth_look_range[16];
	char    total_look_bandwidth_az[16];
	char    total_look_bandwidth_range[16];
	char    w_func_designator_az[32];
	char    w_func_designator_range[32];
	char    data_input_source[16];
	char    nom_res_3db_range[16];
	char    nom_res_az[16];
	char    reserved8[32];
	char    a_track_dop_freq_const_early_image[16];
	char    a_track_dop_freq_lin_early_image[16];
	char    a_track_dop_freq_quad_early_image[16];
	char    spare8[16];
	char    c_track_dop_freq_const_early_image[16];
	char    c_track_dop_freq_lin_early_image[16];
	char    c_track_dop_freq_quad_early_image[16];
	char    time_direction_along_pixel[8];
	char    time_direction_along_line[8];
	char    a_track_dop_freq_rate_const_early_image[16];
	char    a_track_dop_freq_rate_lin_early_image[16];
	char    a_track_dop_freq_rate_quad_early_image[16];
	char    spare9[16];
	char    c_track_dop_freq_rate_const_early_image[16];
	char    c_track_dop_freq_rate_lin_early_image[16];
	char    c_track_dop_freq_rate_quad_early_image[16];
	char    spare10[16];
	char    line_content_indicator[8];
	char    clut_lock_flag[4];
	char    autofocussing_flag[4];
	char    line_spacing[16];
	char    pixel_spacing_range[16];
	char    range_compression_designator[16];
	char    spare11[16];
	char    spare12[16];
	char	calibration_data_indicator[4];
	char	start_line_upper_image[8];
	char	stop_line_upper_image[8];
	char	start_line_bottom_image[8];
	char	stop_line_bottom_image[8];
	char	PRF_switch[4];
	char	PRF_switch_line[8];
	char	spare13[16];
	char	yaw_steering_mode[4];
	char	parameter_table[4];
	char	nom_offnadir_angle[16];
	char	antenna_beam_number[4];
	char	spare14[28];
	char	spare15[120];
	char	num_anno_points[8];
	char	spare16[8];
	char	image_annotation[2048];
	char	spare17[26];
} ;

#define SARLEADER_DSS_WCS_ALOS "*********** DSS RECORD ***********\n"\
"dss_rec_seq_num  			==>	%.4s\n" \
"chan_ind  				==>	%.4s\n"\
"reserved1  				==>	%.16s\n" \
"scene_number  				==>	%.32s\n" \
"input_scene_center_time  		==>	%.32s\n"\
"spare1  				==>	%.16s\n"\
"center_lat  				==>	%.16s\n"\
"center_long  				==>	%.16s\n"\
"center_heading  			==>	%.16s\n"\
"ellipsoid_designator  			==>	%.16s\n"\
"ellipsoid_semimajor_axis  		==>	%.16s\n"\
"ellipsoid_semiminor_axis  		==>	%.16s\n"\
"earth_constant  			==>	%.16s\n"\
"spare2  				==>	%.16s\n"\
"ellipsoid_j2  				==>	%.16s\n"\
"ellipsoid_j3  				==>	%.16s\n"\
"ellipsoid_j4  				==>	%.16s\n"\
"spare					==>	%.16s\n"\
"reserved_new  				==>	%.16s\n"\
"scene_centre_line_number  		==>	%.8s\n"\
"scene_centre_pixel_number  		==>	%.8s\n"\
"scene_length  				==>	%.16s\n"\
"scene_width  				==>	%.16s\n"\
"spare3  				==>	%.16s\n"\
"nchan					==>	%.4s\n"\
"spare4  				==>	%.4s\n"\
"mission_identifier  			==>	%.16s\n"\
"sensor_id_and_mode  			==>	%.32s\n"\
"orbit_number  				==>	%.8s\n"\
"lat_nadir_center  			==>	%.8s\n"\
"long_nadir_center  			==>	%.8s\n"\
"heading_nadir_center  			==>	%.8s\n"\
"clock_angle  				==>	%.8s\n"\
"incidence_angle_center  		==>	%.8s\n"\
"radar_freq  				==>	%.8s\n"\
"radar_wavelength  			==>	%.16s\n"\
"motion_compensation  			==>	%.2s\n"\
"range_pulse_code_specifier  		==>	%.16s\n"\
"range_pulse_amplitude_const  		==>	%.16s\n"\
"range_pulse_amplitude_lin  		==>	%.16s\n"\
"range_pulse_amplitude_quad  		==>	%.16s\n"\
"range_pulse_amplitude_cube  		==>	%.16s\n"\
"range_pulse_amplitude_quart  		==>	%.16s\n"\
"range_pulse_phase_const  		==>	%.16s\n"\
"range_pulse_phase_lin			==>	%.16s\n"\
"range_pulse_phase_quad  		==>	%.16s\n"\
"range_pulse_phase_cube  		==>	%.16s\n"\
"range_pulse_phase_quart  		==>	%.16s\n"\
"chirp_extraction_index  		==>	%.8s\n"\
"spare5  				==>	%.8s\n"\
"sampling_rate  			==>	%.16s\n"\
"range_gate_early_edge_start_image 	==>	%.16s\n"\
"range_pulse_length  			==>	%.16s\n"\
"reserved2  				==>	%.4s\n"\
"range_compressed_flag			==>	%.4s\n"\
"reserved3  				==>	%.32s\n"\
"quantisation_in_bits  			==>	%.8s\n"\
"quantizer_descriptor  			==>	%.12s\n"\
"dc_bias_i  				==>	%.16s\n"\
"dc_bias_q  				==>	%.16s\n"\
"gain_imbalance  			==>	%.16s\n"\
"spare6  				==>	%.32s\n"\
"reserved4  				==>	%.16s\n"\
"antenna_mech_bor  			==>	%.16s\n"\
"reserved5  				==>	%.4s\n"\
"nominal_prf  				==>	%.16s\n"\
"reserved6  				==>	%.32s\n"\
"satelite_encoded_binary_time  		==>	%.16s\n"\
"satelite_clock_time  			==>	%.32s\n"\
"satelite_clock_increment  		==>	%.8s\n"\
"spare7  				==>	%.8s\n"\
"processing_facility_identifier		==>	%.16s\n"\
"processing_system_id  			==>	%.8s\n"\
"processing_version_id			==>	%.8s\n"\
"reserved7  				==>	%.32s\n"\
"product_type_id  			==>	%.32s\n"\
"alg_id  				==>	%.32s\n"\
"nlooks_az  				==>	%.16s\n"\
"neff_looks_range  			==>	%.16s\n"\
"bandwidth_look_az			==>	%.16s\n"\
"bandwidth_look_range  			==>	%.16s\n"\
"total_look_bandwidth_az  		==>	%.16s\n"\
"total_look_bandwidth_range  		==>	%.16s\n"\
"w_func_designator_az  			==>	%.32s\n"\
"w_func_designator_range  		==>	%.32s\n"\
"data_input_source  			==>	%.16s\n"\
"nom_res_3db_range  			==>	%.16s\n"\
"nom_res_az				==>	%.16s\n"\
"reserved8				==>	%.32s\n"\
"a_track_dop_freq_const_early_image  	==>	%.16s\n"\
"a_track_dop_freq_lin_early_image  	==>	%.16s\n"\
"a_track_dop_freq_quad_early_image  	==>	%.16s\n"\
"spare8					==>	%.16s\n"\
"c_track_dop_freq_const_early_image  	==>	%.16s\n"\
"c_track_dop_freq_lin_early_image  	==>	%.16s\n"\
"c_track_dop_freq_quad_early_image  	==>	%.16s\n"\
"time_direction_along_pixel  		==>	%.8s\n"\
"time_direction_along_line  		==>	%.8s\n"\
"a_track_dop_freq_rate_const_early_image	==>	%.16s\n"\
"a_track_dop_freq_rate_lin_early_image  	==>	%.16s\n"\
"a_track_dop_freq_rate_quad_early_image 	==>	%.16s\n"\
"spare9					==>	%.16s\n"\
"c_track_dop_freq_rate_const_early_image	==>	%.16s\n"\
"c_track_dop_freq_rate_lin_early_image	==>	%.16s\n"\
"c_track_dop_freq_rate_quad_early_image	==>	%.16s\n"\
"spare10					==>	%.16s\n"\
"line_content_indicator  		==>	%.8s\n"\
"clut_lock_flag  			==>	%.4s\n"\
"autofocussing_flag			==>	%.4s\n"\
"line_spacing  				==>	%.16s\n"\
"pixel_spacing_range  			==>	%.16s\n"\
"range_compression_designator  		==>	%.16s\n"\
"spare11  				==>	%.16s\n"\
"spare12  				==>	%.16s\n"\
"calibration_data_indicator		==> 	%.4s\n"\
"start_line_upper_image			==>	%.8s\n"\
"stop_line_upper_image			==>	%.8s\n"\
"start_line_bottom_image		==>	%.8s\n"\
"stop_line_bottom_image			==>	%.8s\n"\
"PRF_switch				==>	%.4s\n"\
"PRF_switch_line			==>	%.8s\n"\
"spare13				==>	%.16s\n"\
"yaw_steering_mode			==>	%.4s\n"\
"parameter_table			==>	%.4s\n"\
"nom_offnadir_angle			==>	%.16s\n"\
"antenna_beam_number			==>	%.4s\n"\
"spare14				==>	%.28s\n"\
"spare15				==>	%.120s\n"\
"num_anno_points			==>	%.8s\n"\
"spare16				==>	%.8s\n"\
"image_annotation			==>	%.2048s\n"\
"spare17				==>	%.26s\n"

/* provides structures to read SAR tapes*/
/* modified from the rceos programs by
 C. Tomassini & F. Lorenna */

/*
also from:
 from CERS (RAW) CCT format specifications STD-TM#92-767F
   Canada Centre for Remote Sensing (CCRS)
   Surveys, Mapping and Remote Sensing Sector
   Energy, Mines and Resources Canada

	R. J. Mellors 
	July 1997, IGPP-SIO
*/

#define PLATFORM_RCS_ALOS "%32c%16c%16c%16c%16c%16c%16c%4c%4c%4c%4c%4c%22c%22c%64c%22c%16c%16c%16c%16c%16c%16c"
#define PLATFORM_RVL_ALOS(SP)\
(SP)->orbital_elements,\
(SP)->orbital_element_1,\
(SP)->orbital_element_2,\
(SP)->orbital_element_3,\
(SP)->orbital_element_4,\
(SP)->orbital_element_5,\
(SP)->orbital_element_6,\
(SP)->num_data_points,\
(SP)->year_of_data_points,\
(SP)->month_of_data_points,\
(SP)->day_of_data_points,\
(SP)->day_of_data_points_in_year,\
(SP)->sec_of_day_of_data,\
(SP)->data_points_time_gap,\
(SP)->ref_coord_sys,\
(SP)->greenwhich_mean_hour_angle,\
(SP)->a_track_pos_err,\
(SP)->c_track_pos_err,\
(SP)->radial_pos_err,\
(SP)->a_track_vel_err,\
(SP)->c_track_vel_err,\
(SP)->radial_vel_err

/* ALOS stuff added by RJM June 2007	*/

struct platform_ALOS {
char    orbital_elements[32];
char    orbital_element_1[16];
char    orbital_element_2[16];
char    orbital_element_3[16];
char    orbital_element_4[16];
char    orbital_element_5[16];
char    orbital_element_6[16];
char    num_data_points[4];
char    year_of_data_points[4];
char    month_of_data_points[4];
char    day_of_data_points[4];
char    day_of_data_points_in_year[4];
char    sec_of_day_of_data[22];
char    data_points_time_gap[22];
char    ref_coord_sys[64];
char    greenwhich_mean_hour_angle[22];
char    a_track_pos_err[16];
char    c_track_pos_err[16];
char    radial_pos_err[16];
char    a_track_vel_err[16];
char    c_track_vel_err[16];
char    radial_vel_err[16];
};

#define POSITION_VECTOR_RCS_ALOS "%22c%22c%22c%22c%22c%22c"

#define POSITION_VECTOR_RVL_ALOS(SP)\
(SP)->pos_x,\
(SP)->pos_y,\
(SP)->pos_z,\
(SP)->vel_x,\
(SP)->vel_y,\
(SP)->vel_z

struct position_vector_ALOS {
char pos_x[22] ;
char pos_y[22] ;
char pos_z[22] ;
char vel_x[22] ;
char vel_y[22] ;
char vel_z[22] ;
};

#define PLATFORM_WCS_ALOS "*********** PLATFORM POSITION VECTOR **********\n"\
"orbital_elements		==>	  |%.32s|\n"\
"orbital_element_1		==>	  |%.16s|\n"\
"orbital_element_2		==>	  |%.16s|\n"\
"orbital_element_3		==>	  |%.16s|\n"\
"orbital_element_4		==>	  |%.16s|\n"\
"orbital_element_5		==>	  |%.16s|\n"\
"orbital_element_6		==>	  |%.16s|\n"\
"num_data_points		==>	  |%.4s|\n"\
"year_of_data_points		==>	  |%.4s|\n"\
"month_of_data_points		==>	  |%.4s|\n"\
"day_of_data_points		==>	  |%.4s|\n"\
"day_of_data_points_in_year	==>	  |%.4s|\n"\
"sec_of_day_of_data		==>	  |%.22s|\n"\
"data_points_time_gap		==>	  |%.22s|\n"\
"ref_coord_sys			==>	  |%.64s|\n"\
"greenwhich_mean_hour_angle	==>	  |%.22s|\n"\
"a_track_pos_err		==>	  |%.16s|\n"\
"c_track_pos_err		==>	  |%.16s|\n"\
"radial_pos_err			==>	  |%.16s|\n"\
"a_track_vel_err		==>	  |%.16s|\n"\
"c_track_vel_err		==>	  |%.16s|\n"\
"radial_vel_err			==>	  |%.16s|\n"

#define POSITION_VECTOR_WCS_ALOS "*********** PLATFORM VECTOR **********\n"\
"pos_x	==>        %.22s\n"\
"pos_y	==>        %.22s\n"\
"pos_z	==>        %.22s\n"\
"vel_x	==>        %.22s\n"\
"vel_y	==>        %.22s\n"\
"vel_z	==>        %.22s\n\n" 

struct attitude_info_ALOS {
	char	num_att_data_points[4];
};

#define ATTITUDE_INFO_RCS_ALOS "%4c"

#define ATTITUDE_INFO_WCS_ALOS "*********** ATTITUDE INFO **********\n"\
"num_att_data_points		==>	|%.4s|\n"

#define ATTITUDE_INFO_RVL_ALOS(SP)\
(SP)->num_att_data_points

#define ATTITUDE_DATA_WCS_ALOS "*********** ATTITUDE DATA **********\n"\
"day_of_year		==>		|%.4s|\n"\
"millisecond_day		==>		|%.8s|\n"\
"pitch_data_quality		==>		|%.4s|\n"\
"roll_data_quality		==>		|%.4s|\n"\
"yaw_data_quality		==>		|%.4s|\n"\
"pitch		==>		|%.14s|\n"\
"roll		==>		|%.14s|\n"\
"yaw		==>		|%.14s|\n"\
"pitch_rate_data_quality		==>		|%.4s|\n"\
"roll_rate_data_quality		==>		|%.4s|\n"\
"yaw_rate_data_quality		==>		|%.4s|\n"\
"pitch_rate		==>		|%.14s|\n"\
"roll_rate		==>		|%.14s|\n"\
"yaw_rate		==>		|%.14s|\n"

#define ATTITUDE_DATA_RCS_ALOS "%4c%8c%4c%4c%4c%14c%14c%14c%4c%4c%4c%14c%14c%14c"

#define ATTITUDE_DATA_RVL_ALOS(SP)\
(SP)->day_of_year,\
(SP)->millisecond_day,\
(SP)->pitch_data_quality,\
(SP)->roll_data_quality,\
(SP)->yaw_data_quality,\
(SP)->pitch,\
(SP)->roll,\
(SP)->yaw,\
(SP)->pitch_rate_data_quality,\
(SP)->roll_rate_data_quality,\
(SP)->yaw_rate_data_quality,\
(SP)->pitch_rate,\
(SP)->roll_rate,\
(SP)->yaw_rate

struct attitude_data_ALOS {
	char day_of_year[4];
	char millisecond_day[8];
	char pitch_data_quality[4];
	char roll_data_quality[4];
	char yaw_data_quality[4];
	char pitch[14];
	char roll[14];
	char yaw[14];
	char pitch_rate_data_quality[4];
	char roll_rate_data_quality[4];
	char yaw_rate_data_quality[4];
	char pitch_rate[14];
	char roll_rate[14];
	char yaw_rate[14];
};

struct SAR_info {
	struct sarleader_fdr_fixseg 	*fixseg;
	struct sarleader_fdr_varseg 	*varseg;
	struct sarleader_dss_ALOS	*dss_ALOS; 
	struct platform_ALOS 		*platform_ALOS;
	struct position_vector_ALOS 	*position_ALOS;
	struct attitude_info_ALOS 	*attitude_info_ALOS;
	struct attitude_data_ALOS 	*attitude_ALOS;
	};
