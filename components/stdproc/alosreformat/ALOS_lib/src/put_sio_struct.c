#include "image_sio.h"
#include "lib_functions.h"

#define OUTFILE stdout

/***************************************************************************/
void put_sio_struct(struct PRM prm)
{

	/* set by set_ALOS_defaults */
	if (prm.num_valid_az != NULL_INT) fprintf(OUTFILE,"num_valid_az   	= %d \n",prm.num_valid_az);
	if (prm.nrows != NULL_INT) 	fprintf(OUTFILE,"nrows   		= %d \n",prm.nrows);
	if (prm.first_line != NULL_INT) fprintf(OUTFILE,"first_line   		= %d \n",prm.first_line);
	if (strncmp(prm.deskew,NULL_CHAR,8) != 0) fprintf(OUTFILE,"deskew   		= %s \n",prm.deskew);
	if (prm.caltone != NULL_DOUBLE) fprintf(OUTFILE,"caltone   		= %lf \n",prm.caltone);
	if (prm.st_rng_bin != NULL_INT) fprintf(OUTFILE,"st_rng_bin   		= %d \n",prm.st_rng_bin);
	if (strncmp(prm.iqflip,NULL_CHAR,8) != 0) fprintf(OUTFILE,"Flip_iq   		= %s \n",prm.iqflip);
	if (strncmp(prm.offset_video,NULL_CHAR,8) != 0) fprintf(OUTFILE,"offset_video   	= %s \n",prm.offset_video);
	if (prm.az_res != NULL_DOUBLE) fprintf(OUTFILE,"az_res   		= %lf \n",prm.az_res);
	if (prm.nlooks != NULL_INT) fprintf(OUTFILE,"nlooks   		= %d \n",prm.nlooks);
	if (prm.chirp_ext != NULL_INT) fprintf(OUTFILE,"chirp_ext   		= %d \n",prm.chirp_ext);
	if (strncmp(prm.srm,NULL_CHAR,8) != 0) fprintf(OUTFILE,"scnd_rng_mig   	= %s \n",prm.srm);
	if (prm.rhww != NULL_DOUBLE) fprintf(OUTFILE,"rng_spec_wgt   	= %lf \n",prm.rhww);
	if (prm.pctbw != NULL_DOUBLE) fprintf(OUTFILE,"rm_rng_band   		= %lf \n",prm.pctbw);
	if (prm.pctbwaz != NULL_DOUBLE) fprintf(OUTFILE,"rm_az_band   		= %lf \n",prm.pctbwaz);
	if (prm.rshift != NULL_INT) fprintf(OUTFILE,"rshift  		= %d \n",prm.rshift);
	if (prm.ashift != NULL_INT) fprintf(OUTFILE,"ashift  	 	= %d \n",prm.ashift);
	if (prm.stretch_a != NULL_DOUBLE) fprintf(OUTFILE,"stretch_r   		= %lf \n",prm.stretch_r);
	if (prm.stretch_a != NULL_DOUBLE) fprintf(OUTFILE,"stretch_a   		= %lf \n",prm.stretch_a);
	if (prm.a_stretch_r != NULL_DOUBLE) fprintf(OUTFILE,"a_stretch_r   		= %lf \n",prm.a_stretch_r);
	if (prm.a_stretch_a != NULL_DOUBLE) fprintf(OUTFILE,"a_stretch_a   		= %lf \n",prm.a_stretch_a);
	if (prm.first_sample != NULL_INT) fprintf(OUTFILE,"first_sample   	= %d \n",prm.first_sample);
	if (prm.SC_identity != NULL_INT) fprintf(OUTFILE,"SC_identity   		= %d \n",prm.SC_identity);
	if (prm.fs != NULL_DOUBLE) fprintf(OUTFILE,"rng_samp_rate   	= %lf \n",prm.fs);

	/* from read_ALOS_data */
	if (strncmp(prm.input_file,NULL_CHAR,8) != 0) fprintf(OUTFILE,"input_file		= %s \n",prm.input_file);
	if (prm.num_rng_bins != NULL_INT) fprintf(OUTFILE,"num_rng_bins		= %d \n",prm.num_rng_bins);
	if (prm.bytes_per_line != NULL_INT) fprintf(OUTFILE,"bytes_per_line		= %d \n",prm.bytes_per_line);
	if (prm.good_bytes != NULL_INT) fprintf(OUTFILE,"good_bytes_per_line	= %d \n",prm.good_bytes);
	if (prm.prf != NULL_DOUBLE) fprintf(OUTFILE,"PRF			= %lf \n",prm.prf);
	if (prm.pulsedur != NULL_DOUBLE) fprintf(OUTFILE,"pulse_dur		= %e \n",prm.pulsedur);
	if (prm.near_range != NULL_DOUBLE) fprintf(OUTFILE,"near_range		= %lf \n",prm.near_range);
	if (prm.num_lines != NULL_INT) fprintf(OUTFILE,"num_lines		= %d \n",prm.num_lines);
	if (prm.num_patches != NULL_INT) fprintf(OUTFILE,"num_patches		= %d \n",prm.num_patches);
       	if (prm.SC_clock_start != NULL_DOUBLE) fprintf(OUTFILE,"SC_clock_start		= %16.10lf \n",prm.SC_clock_start);
       	if (prm.SC_clock_stop != NULL_DOUBLE) fprintf(OUTFILE,"SC_clock_stop		= %16.10lf \n",prm.SC_clock_stop);
	if (strncmp(prm.led_file,NULL_CHAR,8) != 0) fprintf(OUTFILE,"led_file		= %s \n",prm.led_file);

	/* from read_ALOS_ldrfile */
       	if (strncmp(prm.orbdir,NULL_CHAR,8) != 0) fprintf(OUTFILE,"orbdir			= %.1s \n",prm.orbdir);
       	if (prm.lambda != NULL_DOUBLE) fprintf(OUTFILE,"radar_wavelength	= %lg \n",prm.lambda);
       	if (prm.chirp_slope != NULL_DOUBLE) fprintf(OUTFILE,"chirp_slope		= %lg \n",prm.chirp_slope);
       	if (prm.fs != NULL_DOUBLE) fprintf(OUTFILE,"rng_samp_rate		= %lg \n",prm.fs);
       	if (prm.xmi != NULL_DOUBLE) fprintf(OUTFILE,"I_mean			= %lg \n",prm.xmi);
       	if (prm.xmq != NULL_DOUBLE) fprintf(OUTFILE,"Q_mean			= %lg \n",prm.xmq);
       	if (prm.vel != NULL_DOUBLE) fprintf(OUTFILE,"SC_vel			= %lf \n",prm.vel);
       	if (prm.RE != NULL_DOUBLE) fprintf(OUTFILE,"earth_radius		= %lf \n",prm.RE);
       	if (prm.ra != NULL_DOUBLE) fprintf(OUTFILE,"equatorial_radius	= %lf \n",prm.ra);
       	if (prm.rc != NULL_DOUBLE) fprintf(OUTFILE,"polar_radius		= %lf \n",prm.rc);
       	if (prm.ht != NULL_DOUBLE) fprintf(OUTFILE,"SC_height		= %lf \n",prm.ht);
       	if (prm.fd1 != NULL_DOUBLE) fprintf(OUTFILE,"fd1			= %lf \n",prm.fd1);
       	if (prm.fdd1 != NULL_DOUBLE) fprintf(OUTFILE,"fdd1			= %lf \n",prm.fdd1);
       	if (prm.fddd1 != NULL_DOUBLE) fprintf(OUTFILE,"fddd1			= %lf \n",prm.fddd1);

	/* from calc_baseline */
	if (prm.rshift != NULL_INT) printf("rshift                  = %d \n",prm.rshift);
	if (prm.sub_int_r != NULL_DOUBLE) printf("sub_int_r               = %f \n",prm.sub_int_r);
	if (prm.ashift != NULL_INT) printf("ashift                  = %d\n",prm.ashift);
	if (prm.sub_int_a != NULL_DOUBLE) printf("sub_int_a               = %f \n",prm.sub_int_a);
	if (prm.bpara != NULL_DOUBLE) printf("B_parallel              = %f \n",prm.bpara);
	if (prm.bperp != NULL_DOUBLE) printf("B_perpendicular         = %f \n",prm.bperp);
	if (prm.baseline_start != NULL_DOUBLE) printf("baseline_start          = %f \n",prm.baseline_start);
	if (prm.alpha_start != NULL_DOUBLE) printf("alpha_start             = %f \n",prm.alpha_start);
	if (prm.baseline_end != NULL_DOUBLE) printf("baseline_end            = %f \n",prm.baseline_end);
	if (prm.alpha_end != NULL_DOUBLE) printf("alpha_end               = %f \n",prm.alpha_end);

	/* from sarp */
	if (strncmp(prm.SLC_file,NULL_CHAR,8) !=0)  printf("SLC_file               = %s \n",prm.SLC_file);

}
/***************************************************************************/
