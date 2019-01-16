/* Structure to read ALOSE signal data */
/*
Each structure has write control string (WCS) 
and pointers (RVL) to aid in input and output. 
RJM June 2007

  Dec. 2009 Modified for RESTEC format.    Jeff Bytof

  15-Apr-2010  Replace ALOS identifier with ALOSE  Jeff Bytof

*/
/*
struct ALOS_image {
	struct	sardata_record		*rec1;
	struct	sardata_descriptor	*dfd;	
	struct	sardata_record 		*rec2;
	struct	sardata_info		*sdr;
};
*/

/* beginning of short binary segment */
/*
struct sardata_record {
	int	record_seq_no;
	char	record_subtype_code1;
	char	record_type_code1;
	char	record_subtype_code2;
	char	record_subtype_code3;
	int	record_length;
};
*/

/*
#define SARDATA_RECORD_WCS "*********** SAR FDR BINARY **********\n"\
"record_seq_no		==>	%4x\n"\
"record_subtype_code1	==>	%1x\n"\
"record_type_code1	==>	%1x\n"\
"record_subtype_code2	==>	%1x\n"\
"record_subtype_code3	==>	%1x\n"\
"record_length		==>	%4x\n\n"

#define SARDATA_RECORD_RVL(SP)\
(SP)->record_seq_no,\
(SP)->record_subtype_code1,\
(SP)->record_type_code1,\
(SP)->record_subtype_code2,\
(SP)->record_subtype_code3,\
(SP)->record_length
*/

/* end of short binary segment */

/******* CONTINUATION OF RESTEC IMAGE OPTIONS FILE DESCRIPTOR RECORD ********/

struct sardata_descriptor_ALOSE {
	char	ascii_ebcdic_flag[2];
	char	blank_1[2];
	char	format_doc_ID[12];
	char	format_control_level[2];
	char	file_design_descriptor[2];
	char	facility_soft_release[12];
	char	file_number[4];
	char	file_name[16];
	char	record_seq_loc_type_flag_1[4];
	char	record_seq_loc_type_flag_2[8];
	char	sequence_number_loc[4];
	char	record_code_loc_flag[4];
	char	record_code_loc[8];
	char	record_code_field_length[4];
	char	record_length_loc_flag[4];
	char	record_length_loc[8];
	char	record_length_field_length[4];
	char	blank_2[68];
	char	number_sar_data_records[6];
	char	sar_data_record_length[6];
	char	blank_3[24];
	char	num_bits_sample[4];
	char	num_sample_data_group[4];
	char	num_bytes_data_group[4];
	char	just_order_samples[4];
	char	num_sar_channels[4];
	char	num_lines_data_set[8];
	char	num_left_border_pixels[4];
	char	total_num_data_groups[8];
	char	num_right_border_pixels[4];
	char	num_top_border_lines[4];
	char	num_bottom_border_lines[4];
	char	interleave_indicator[4];
	char	num_physical_records_line[2];
	char	num_physical_records_multi_chan[2];
	char	num_bytes_prefix[4];
	char	num_bytes_SAR_data[8];
	char	num_bytes_suffix[4];
	char	pref_fix_repeat_flag[4];
	char	sample_data_lin_no[8];
	char	SAR_chan_num_loc[8];
	char	time_SAR_data_line[8];
	char	left_fill_count[8];
	char	right_fill_count[8];
	char	pad_pixels[4];
	char	blank_4[28];
	char	sar_data_line_qual_loc[8];
	char	calib_info_field_loc[8];
	char	gain_values_field_loc[8];
	char	bias_values_field_loc[8];
	char	sar_data_format_code_1[28];
	char	sar_data_format_code_2[4];
	char	num_left_fill_bits_pixel[4];
	char	num_right_fill_bits_pixel[4];
	char	max_range_pixel[8];
/*	char	blank_5[272]; */  /* restec format change - bytof */
	char	blank_5[15804];     /* restec format change - bytof */
};

#define SARDATA_DESCRIPTOR_WCS_ALOSE "*********** SAR DATA DESCRIPTOR**********\n"\
"ascii_ebcdic_flag 		==>	%.2s\n"\
"blank_1 		==>	%.2s\n"\
"format_doc_ID 		==>	%.12s\n"\
"format_control_level 		==>	%.2s\n"\
"file_design_descriptor 		==>	%.2s\n"\
"facility_soft_release 		==>	%.12s\n"\
"file_number 		==>	%.4s\n"\
"file_name 		==>	%.16s\n"\
"record_seq_loc_type_flag_1 		==>	%.4s\n"\
"record_seq_loc_type_flag_2 		==>	%.8s\n"\
"sequence_number_loc 		==>	%.4s\n"\
"record_code_loc_flag 		==>	%.4s\n"\
"record_code_loc 		==>	%.8s\n"\
"record_code_field_length 		==>	%.4s\n"\
"record_length_loc_flag 		==>	%.4s\n"\
"record_length_loc 		==>	%.8s\n"\
"record_length_field_length 		==>	%.4s\n"\
"blank_2 		==>	%.68s\n"\
"number_sar_data_records 		==>	%.6s\n"\
"sar_data_record_length 		==>	%.6s\n"\
"blank_3 		==>	%.24s\n"\
"num_bits_sample 		==>	%.4s\n"\
"num_sample_data_group 		==>	%.4s\n"\
"num_bytes_data_group 		==>	%.4s\n"\
"just_order_samples 		==>	%.4s\n"\
"num_sar_channels 		==>	%.4s\n"\
"num_lines_data_set 		==>	%.8s\n"\
"num_left_border_pixels 		==>	%.4s\n"\
"total_num_data_groups 		==>	%.8s\n"\
"num_right_border_pixels 		==>	%.4s\n"\
"num_top_border_lines 		==>	%.4s\n"\
"num_bottom_border_lines 		==>	%.4s\n"\
"interleave_indicator 		==>	%.4s\n"\
"num_physical_records_line 		==>	%.2s\n"\
"num_physical_records_multi_chan 		==>	%.2s\n"\
"num_bytes_prefix 		==>	%.4s\n"\
"num_bytes_SAR_data 		==>	%.8s\n"\
"num_bytes_suffix 		==>	%.4s\n"\
"pref_fix_repeat_flag 		==>	%.4s\n"\
"sample_data_lin_no 		==>	%.8s\n"\
"SAR_chan_num_loc 		==>	%.8s\n"\
"time_SAR_data_line 		==>	%.8s\n"\
"left_fill_count 		==>	%.8s\n"\
"right_fill_count 		==>	%.8s\n"\
"pad_pixels 		==>	%.4s\n"\
"blank_4 		==>	%.28s\n"\
"sar_data_line_qual_loc 		==>	%.8s\n"\
"calib_info_field_loc 		==>	%.8s\n"\
"gain_values_field_loc 		==>	%.8s\n"\
"bias_values_field_loc 		==>	%.8s\n"\
"sar_data_format_code_1 		==>	%.28s\n"\
"sar_data_format_code_2 		==>	%.4s\n"\
"num_left_fill_bits_pixel 		==>	%.4s\n"\
"num_right_fill_bits_pixel 		==>	%.4s\n"\
"max_range_pixel 		==>	%.8s\n"\
"blank_5 		==>	%.15804s\n"

#define SARDATA_DESCRIPTOR_RVL_ALOSE(SP)\
(SP)->ascii_ebcdic_flag,\
(SP)->blank_1,\
(SP)->format_doc_ID,\
(SP)->format_control_level,\
(SP)->file_design_descriptor,\
(SP)->facility_soft_release,\
(SP)->file_number,\
(SP)->file_name,\
(SP)->record_seq_loc_type_flag_1,\
(SP)->record_seq_loc_type_flag_2,\
(SP)->sequence_number_loc,\
(SP)->record_code_loc_flag,\
(SP)->record_code_loc,\
(SP)->record_code_field_length,\
(SP)->record_length_loc_flag,\
(SP)->record_length_loc,\
(SP)->record_length_field_length,\
(SP)->blank_2,\
(SP)->number_sar_data_records,\
(SP)->sar_data_record_length,\
(SP)->blank_3,\
(SP)->num_bits_sample,\
(SP)->num_sample_data_group,\
(SP)->num_bytes_data_group,\
(SP)->just_order_samples,\
(SP)->num_sar_channels,\
(SP)->num_lines_data_set,\
(SP)->num_left_border_pixels,\
(SP)->total_num_data_groups,\
(SP)->num_right_border_pixels,\
(SP)->num_top_border_lines,\
(SP)->num_bottom_border_lines,\
(SP)->interleave_indicator,\
(SP)->num_physical_records_line,\
(SP)->num_physical_records_multi_chan,\
(SP)->num_bytes_prefix,\
(SP)->num_bytes_SAR_data,\
(SP)->num_bytes_suffix,\
(SP)->pref_fix_repeat_flag,\
(SP)->sample_data_lin_no,\
(SP)->SAR_chan_num_loc,\
(SP)->time_SAR_data_line,\
(SP)->left_fill_count,\
(SP)->right_fill_count,\
(SP)->pad_pixels,\
(SP)->blank_4,\
(SP)->sar_data_line_qual_loc,\
(SP)->calib_info_field_loc,\
(SP)->gain_values_field_loc,\
(SP)->bias_values_field_loc,\
(SP)->sar_data_format_code_1,\
(SP)->sar_data_format_code_2,\
(SP)->num_left_fill_bits_pixel,\
(SP)->num_right_fill_bits_pixel,\
(SP)->max_range_pixel,\
(SP)->blank_5

struct sardata_info_ALOSE {
	int 	sequence_number;
	char  		subtype[4];
	int 	record_length;
	int 	data_line_number;
	int	data_record_index;
	int	n_left_fill_pixels;	
	int	n_data_pixels;
	int	n_right_fill_pixels;
	int	sensor_update_flag;
	int	sensor_acquisition_year;
	int	sensor_acquisition_DOY;
	int	sensor_acquisition_msecs_day;
	short		channel_indicator;
	short		channel_code;
	short		transmit_polarization;
	short		receive_polarization;
	int	PRF;
	int	scan_ID;
	short		onboard_range_compress;
	short		chirp_type;
	int	chirp_length;
	int 	chirp_constant_coeff;
	int	chirp_linear_coeff;
	int	chirp_quad_coeff;
	char		spare1[4];
	char		spare2[4];
	int	receiver_gain;
	int	nought_line_flag;
	int	elec_antenna_elevation_angle;
	int	mech_antenna_elevation_angle;
	int	elec_antenna_squint_angle;
	int	mech_antenna_squint_angle;
	int	slant_range;
	int	data_record_window_position;
	char		spare3[4];
	short		platform_update_flag;
	int	platform_latitude;
	int	platform_longitude;
	int	platform_altitude;
	int	platform_ground_speed;
	int	platform_velocity_x;
	int	platform_velocity_y;
	int	platform_velocity_z;
	int	platform_acc_x;
	int	platform_acc_y;
	int	platform_acc_z;
	int	platform_track_angle_1;
	int	platform_track_angle_2;
	int	platform_pitch_angle;
	int	platform_roll_angle;
	int	platform_yaw_angle;

/*	char		blank1[92];      */   /* restec format change - bytof */
/*	int	frame_counter;           */   /* restec format change - bytof */

	char		PALSAR_aux_data[100];

/*	char		blank2[24];      */   /* restec format change - bytof */

};

#define SARDATA__WCS_ALOSE "*********** SAR DATA DESCRIPTOR**********\n"\
"sequence_number	==>	%d\n"\
"subtype	==>	%.4s\n"\
"record_length	==>	%d\n"\
"data_line_number	==>	%d\n"\
"data_record_index	==>	%d\n"\
"n_left_fill_pixels	==>	%d\n"\
"n_data_pixels	==>	%d\n"\
"n_right_fill_pixels	==>	%d\n"\
"sensor_update_flag	==>	%d\n"\
"sensor_acquisition_year	==>	%d\n"\
"sensor_acquisition_DOY	==>	%d\n"\
"sensor_acquisition_msecs_day	==>	%d\n"\
"channel_indicator	==>	%d\n"\
"channel_code	==>	%d\n"\
"transmit_polarization	==>	%d\n"\
"receive_polarization	==>	%d\n"\
"PRF	==>	%d\n"\
"scan_ID	==>	%d\n"\
"onboard_range_compress	==>	%d\n"\
"chirp_type	==>	%d\n"\
"chirp_length	==>	%d\n"\
"chirp_constant_coeff	==>	%d\n"\
"chirp_linear_coeff	==>	%d\n"\
"chirp_quad_coeff	==>	%d\n"\
"receiver_gain	==>	%d\n"\
"nought_line_flag	==>	%d\n"\
"elec_antenna_elevation_angle	==>	%d\n"\
"mech_antenna_elevation_angle	==>	%d\n"\
"elec_antenna_squint_angle	==>	%d\n"\
"mech_antenna_squint_angle	==>	%d\n"\
"slant_range	==>	%d\n"\
"data_record_window_position	==>	%d\n"\
"platform_update_flag	==>	%d\n"\
"platform_latitude	==>	%d\n"\
"platform_longitude	==>	%d\n"\
"platform_altitude	==>	%d\n"\
"platform_ground_speed	==>	%d\n"\
"platform_velocity_x	==>	%d\n"\
"platform_velocity_y	==>	%d\n"\
"platform_velocity_z	==>	%d\n"\
"platform_acc_x	==>	%d\n"\
"platform_acc_y	==>	%d\n"\
"platform_acc_z	==>	%d\n"\
"platform_track_angle_1	==>	%d\n"\
"platform_track_angle_2	==>	%d\n"\
"platform_pitch_angle	==>	%d\n"\
"platform_roll_angle	==>	%d\n"\
"platform_yaw_angle	==>	%d\n"    /* restec format change - bytof */
/* "frame_counter	==>	%d\n"    */  /* restec format change - bytof */

#define SARDATA_RVL_ALOSE(SP)\
(SP).sequence_number,\
(SP).subtype,\
(SP).record_length,\
(SP).data_line_number,\
(SP).data_record_index,\
(SP).n_left_fill_pixels,\
(SP).n_data_pixels,\
(SP).n_right_fill_pixels,\
(SP).sensor_update_flag,\
(SP).sensor_acquisition_year,\
(SP).sensor_acquisition_DOY,\
(SP).sensor_acquisition_msecs_day,\
(SP).channel_indicator,\
(SP).channel_code,\
(SP).transmit_polarization,\
(SP).receive_polarization,\
(SP).PRF,\
(SP).scan_ID,\
(SP).onboard_range_compress,\
(SP).chirp_type,\
(SP).chirp_length,\
(SP).chirp_constant_coeff,\
(SP).chirp_linear_coeff,\
(SP).chirp_quad_coeff,\
(SP).receiver_gain,\
(SP).nought_line_flag,\
(SP).elec_antenna_elevation_angle,\
(SP).mech_antenna_elevation_angle,\
(SP).elec_antenna_squint_angle,\
(SP).mech_antenna_squint_angle,\
(SP).slant_range,\
(SP).data_record_window_position,\
(SP).platform_update_flag,\
(SP).platform_latitude,\
(SP).platform_longitude,\
(SP).platform_altitude,\
(SP).platform_ground_speed,\
(SP).platform_velocity_x,\
(SP).platform_velocity_y,\
(SP).platform_velocity_z,\
(SP).platform_acc_x,\
(SP).platform_acc_y,\
(SP).platform_acc_z,\
(SP).platform_track_angle_1,\
(SP).platform_track_angle_2,\
(SP).platform_pitch_angle,\
(SP).platform_roll_angle,\
(SP).platform_yaw_angle     /* restec format change - bytof */
/* (SP).frame_counter   */   /* restec format change - bytof */
