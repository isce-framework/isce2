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

#define SARLEADER_FDR_BINARY_WCS "*********** SAR FDR BINARY **********\n"\
"record_seq_no		==>	%1d\n"\
"record_subtype_code1	==>	%1x\n"\
"record_type_code1	==>	%1x\n"\
"record_subtype_code2	==>	%1x\n"\
"record_subtype_code3	==>	%1x\n"\
"record_length		==>	%1d\n\n"

#define SARLEADER_FDR_BINARY_RVL(SP)\
(SP)->record_seq_no,\
(SP)->record_subtype_code1,\
(SP)->record_type_code1,\
(SP)->record_subtype_code2,\
(SP)->record_subtype_code3,\
(SP)->record_length

struct sarleader_binary {
	int	record_seq_no;
	char	record_subtype_code1;
	char	record_type_code1;
	char	record_subtype_code2;
	char	record_subtype_code3;
	int	record_length;
};

#define SARLEADER_FDR_FIXSEG_RCS "%2c%2c%12c%2c%2c%12c%4c%16c%4c%8c%4c%4c%8c%4c%4c%8c%4c%4c%64c"

#define SARLEADER_FDR_FIXSEG_RVL(SP)\
(SP)->A_E_flag,\
(SP)->blank_2,\
(SP)->for_con_doc,\
(SP)->for_con_doc_rev_level,\
(SP)->file_des_rev_level,\
(SP)->softw_rel,\
(SP)->file_number,\
(SP)->file_name,\
(SP)->rec_seq_loc_type_flag,\
(SP)->seq_number_loc,\
(SP)->seq_number_field_length,\
(SP)->rec_code_loc_type_flag,\
(SP)->rec_code_loc,\
(SP)->rec_code_field_length,\
(SP)->rec_len_loc_type_flag,\
(SP)->rec_len_loc,\
(SP)->rec_len_field_length,\
(SP)->reserved_4,\
(SP)->reserved_segment      


struct sarleader_fdr_fixseg {
	char 	A_E_flag[2];               		/*   13         */
	char 	blank_2[2];                		/*   15         */
	char 	for_con_doc[12];           		/*   17         */
	char 	for_con_doc_rev_level[2];  		/*   29         */
	char 	file_des_rev_level[2];     		/*   31           */
	char 	softw_rel[12];             		/*   33           */
	char 	file_number[4];            		/*   45           */
	char 	file_name[16];             		/*   49           */
	char 	rec_seq_loc_type_flag[4];  		/*   65           */
	char 	seq_number_loc[8];         		/*   69           */
	char 	seq_number_field_length[4];		/*   77           */
	char 	rec_code_loc_type_flag[4]; 		/*   81           */
	char 	rec_code_loc[8];           		/*   85           */
	char 	rec_code_field_length[4];  		/*   93           */
	char 	rec_len_loc_type_flag[4];  		/*   97           */
	char 	rec_len_loc[8];            		/*   101           */
	char 	rec_len_field_length[4];   		/*   109           */
	char	reserved_4[4];             		/*   113           */
	char	reserved_segment[64];      		/*   117           */
};

#define SARLEADER_FDR_VARSEG_RCS "%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%6c%60c%6c%6c%288c"

#define SARLEADER_FDR_VARSEG_RVL(SP)\
(SP)->n_data_set_summ_rec,\
(SP)->data_set_summ_rec_len,\
(SP)->n_map_projec_rec,\
(SP)->map_projec_rec_len,\
(SP)->n_plat_pos_data_rec,\
(SP)->plat_pos_data_rec_len,\
(SP)->n_att_data_rec,\
(SP)->att_data_rec_len,\
(SP)->n_rad_data_rec,\
(SP)->rad_data_rec_len,\
(SP)->n_rad_comp_rec,\
(SP)->rad_comp_rec_len,\
(SP)->n_data_qua_summ_rec,\
(SP)->data_qua_summ_rec_len,\
(SP)->n_data_hist_rec,\
(SP)->data_hist_rec_len,\
(SP)->n_range_spectra_rec,\
(SP)->range_spectra_rec_len,\
(SP)->n_DEM_des_rec,\
(SP)->DEM_des_rec_len,\
(SP)->n_radar_par_update_rec,\
(SP)->radar_par_update_rec_len,\
(SP)->n_annotation_data_rec,\
(SP)->annotation_data_rec_len,\
(SP)->n_detailed_proc_rec,\
(SP)->detailed_proc_rec_len,\
(SP)->n_cal_rec,\
(SP)->cal_rec_len,\
(SP)->n_GCP_rec,\
(SP)->GCP_rec_len,\
(SP)->spare_60,\
(SP)->n_facility_data_rec,\
(SP)->facility_data_rec_len,\
(SP)->blanks_288

struct sarleader_fdr_varseg {
	char  n_data_set_summ_rec[6];                  	/* 181-186  I6*/
	char  data_set_summ_rec_len[6];                	/* 187-192  I6*/
	char  n_map_projec_rec[6];                     	/* 193-198  I6*/
	char  map_projec_rec_len[6];                   	/* 199-204  I6*/
	char  n_plat_pos_data_rec[6];                  	/* 205-210  I6*/
	char  plat_pos_data_rec_len[6];                	/* 211-216  I6*/
	char  n_att_data_rec[6];                       	/* 217-222  I6*/
	char  att_data_rec_len[6];                     	/* 223-228  I6*/
	char  n_rad_data_rec[6];   			/* 229-234  I6*/
	char  rad_data_rec_len[6];             		/* 235-240  I6*/
	char  n_rad_comp_rec[6];      			/* 241-246  I6*/
	char  rad_comp_rec_len[6];    			/* 247-252  I6*/
	char  n_data_qua_summ_rec[6];  			/* 253-258  I6*/
	char  data_qua_summ_rec_len[6];		   	/* 259-264  I6*/
	char  n_data_hist_rec[6];   			/* 265-270  I6*/
	char  data_hist_rec_len[6];   			/* 271-276  I6*/
	char  n_range_spectra_rec[6];   		/* 277-282  I6*/
	char  range_spectra_rec_len[6];  		/* 283-288  I6*/
	char  n_DEM_des_rec[6];   			/* 289-294  I6*/
	char  DEM_des_rec_len[6]; 			/* 295-300  I6*/
	char  n_radar_par_update_rec[6]; 		/* 301-306  I6*/
	char  radar_par_update_rec_len[6]; 		/* 307-312  I6*/
	char  n_annotation_data_rec[6]; 		/* 313-318  I6*/
	char  annotation_data_rec_len[6];  		/* 319-324  I6*/
	char  n_detailed_proc_rec[6]; 			/* 325-330  I6*/
	char  detailed_proc_rec_len[6];  		/* 331-336  I6*/
	char  n_cal_rec[6];       			/* 337-342  I6*/
	char  cal_rec_len[6];         			/* 343-348  I6*/
	char  n_GCP_rec[6];  				/* 349-354  I6*/
	char  GCP_rec_len[6];  				/* 355-360  I6*/
	char  spare_60[60]; 				/* 361-420  I6*/
	char  n_facility_data_rec[6];  			/* 421-426  I6*/
	char  facility_data_rec_len[6]; 		/* 427-432  I6*/
	char  blanks_288[288];              		/* 433-720  A80*/
};

#define SARLEADER_FDR_FIXSEG_WCS "*********** SAR FDR FIXED SEGMENT ***********\n"\
"A_E_flag  		==>	%.2s\n"\
"blank_2  		==>	%.2s\n"\
"for_con_doc  		==>	%.12s\n"\
"for_con_doc_rev_level	==>	%.2s\n"\
"file_des_rev_level  	==>	%.2s\n"\
"softw_rel  		==>	%.12s\n"\
"file_number  		==>	%.4s\n"\
"file_name  		==>	%.16s\n"\
"rec_seq_loc_type_flag  	==>	%.4s\n"\
"seq_number_loc		==>	%.8s\n"\
"seq_number_field_length	==>	%.4s\n"\
"rec_code_loc_type_flag 	==>	%.4s\n"\
"rec_code_loc		==>	%.8s\n"\
"rec_code_field_length  	==>	%.4s\n"\
"rec_len_loc_type_flag  	==>	%.4s\n"\
"rec_len_loc  		==>	%.8s\n"\
"rec_len_field_length  	==>	%.4s\n"\
"reserved_4  		==>	%.4s\n"\
"reserved_segment  	==>	%.64s\n\n"

#define SARLEADER_FDR_VARSEG_WCS "*********** SAR FDR VARIABLE SEG ***********\n"\
"n_data_set_summ_rec		==>	%.6s\n"\
"data_set_summ_rec_len		==>	%.6s\n"\
"n_map_projec_rec		==>	%.6s\n"\
"map_projec_rec_len		==>	%.6s\n"\
"n_plat_pos_data_rec		==>	%.6s\n"\
"plat_pos_data_rec_len		==>	%.6s\n"\
"n_att_data_rec			==>	%.6s\n"\
"att_data_rec_len		==>	%.6s\n"\
"n_rad_data_rec			==>	%.6s\n"\
"rad_data_rec_len		==>	%.6s\n"\
"n_rad_comp_rec			==>	%.6s\n"\
"rad_comp_rec_len		==>	%.6s\n"\
"n_data_qua_summ_rec		==>	%.6s\n"\
"data_qua_summ_rec_len		==>	%.6s\n"\
"n_data_hist_rec			==>	%.6s\n"\
"data_hist_rec_len		==>	%.6s\n"\
"n_range_spectra_rec		==>	%.6s\n"\
"range_spectra_rec_len		==>	%.6s\n"\
"n_DEM_des_rec			==>	%.6s\n"\
"DEM_des_rec_len			==>	%.6s\n"\
"n_radar_par_update_rec		==>	%.6s\n"\
"radar_par_update_rec_len	==>	%.6s\n"\
"n_annotation_data_rec		==>	%.6s\n"\
"annotation_data_rec_len		==>	%.6s\n"\
"n_detailed_proc_rec		==>	%.6s\n"\
"detailed_proc_rec_len		==>	%.6s\n"\
"n_cal_rec			==>	%.6s\n"\
"cal_rec_len			==>	%.6s\n"\
"n_GCP_rec			==>	%.6s\n"\
"GCP_rec_len			==>	%.6s\n"\
"spare_60			==>	%.60s\n"\
"n_facility_data_rec		==>	%.6s\n"\
"facility_data_rec_len		==>	%.6s\n"\
"blanks_288			==>	%.288s\n\n"
