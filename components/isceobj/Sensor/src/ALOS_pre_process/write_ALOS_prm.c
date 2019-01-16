#include "image_sio.h"
#include "lib_functions.h"
/***************************************************************************/
void write_ALOS_prm(FILE *prmfile, char *filename, struct PRM *prm)
{

	fprintf(stderr,".... writing PRM file %s\n", filename);

	/* set by set_ALOS_defaults */
	fprintf(prmfile,"num_valid_az   	= %d \n",prm->num_valid_az);
	fprintf(prmfile,"nrows   		= %d \n",prm->nrows);
	fprintf(prmfile,"first_line   		= %d \n",prm->first_line);
	fprintf(prmfile,"deskew   		= %s \n",prm->deskew);
	fprintf(prmfile,"caltone   		= %lf \n",prm->caltone);
	fprintf(prmfile,"st_rng_bin   		= %d \n",prm->st_rng_bin);
	fprintf(prmfile,"Flip_iq   		= %s \n",prm->iqflip);
	fprintf(prmfile,"offset_video   	= %s \n",prm->offset_video);
	fprintf(prmfile,"az_res   		= %lf \n",prm->az_res);
	fprintf(prmfile,"nlooks   		= %d \n",prm->nlooks);
	fprintf(prmfile,"chirp_ext   		= %d \n",prm->chirp_ext);
	fprintf(prmfile,"scnd_rng_mig   	= %s \n",prm->srm);
	fprintf(prmfile,"rng_spec_wgt   	= %lf \n",prm->rhww);
	fprintf(prmfile,"rm_rng_band   		= %lf \n",prm->pctbw);
	fprintf(prmfile,"rm_az_band   		= %lf \n",prm->pctbwaz);
	fprintf(prmfile,"rshift  		= %d \n",prm->rshift);
	fprintf(prmfile,"ashift  	 	= %d \n",prm->ashift);
	fprintf(prmfile,"stretch_r   		= %lf \n",prm->stretch_r);
	fprintf(prmfile,"stretch_a   		= %lf \n",prm->stretch_a);
	fprintf(prmfile,"a_stretch_r   		= %lf \n",prm->a_stretch_r);
	fprintf(prmfile,"a_stretch_a   		= %lf \n",prm->a_stretch_a);
	fprintf(prmfile,"first_sample   	= %d \n",prm->first_sample);
	fprintf(prmfile,"SC_identity   		= %d \n",prm->SC_identity);
	fprintf(prmfile,"rng_samp_rate   	= %lf \n",prm->fs);

	/* from read_ALOS_data */
	fprintf(prmfile,"input_file		= %s \n",prm->input_file);
	fprintf(prmfile,"num_rng_bins		= %d \n",prm->num_rng_bins);
	fprintf(prmfile,"bytes_per_line		= %d \n",prm->bytes_per_line);
	fprintf(prmfile,"good_bytes_per_line	= %d \n",prm->good_bytes);
	fprintf(prmfile,"PRF			= %lf \n",prm->prf);
	fprintf(prmfile,"pulse_dur		= %e \n",prm->pulsedur);
	fprintf(prmfile,"near_range		= %lf \n",prm->near_range);
	fprintf(prmfile,"num_lines		= %d \n",prm->num_lines);
	fprintf(prmfile,"num_patches		= %d \n",prm->num_patches);
       	fprintf(prmfile,"SC_clock_start		= %16.10lf \n",prm->SC_clock_start);
       	fprintf(prmfile,"SC_clock_stop		= %16.10lf \n",prm->SC_clock_stop);
	fprintf(prmfile,"led_file		= %s \n",prm->led_file);

	/* from read_ALOS_ldrfile */
       	fprintf(prmfile,"date			= %.6s \n",prm->date);
       	fprintf(prmfile,"orbdir			= %.1s \n",prm->orbdir);
       	fprintf(prmfile,"radar_wavelength	= %lg \n",prm->lambda);
       	fprintf(prmfile,"chirp_slope		= %lg \n",prm->chirp_slope);
       	fprintf(prmfile,"rng_samp_rate		= %lg \n",prm->fs);
       	fprintf(prmfile,"I_mean			= %lg \n",prm->xmi);
       	fprintf(prmfile,"Q_mean			= %lg \n",prm->xmq);
       	fprintf(prmfile,"SC_vel			= %lf \n",prm->vel);
       	fprintf(prmfile,"earth_radius		= %lf \n",prm->RE);
       	fprintf(prmfile,"equatorial_radius	= %lf \n",prm->ra);
       	fprintf(prmfile,"polar_radius		= %lf \n",prm->rc);
       	fprintf(prmfile,"SC_height		= %lf \n",prm->ht);
       	fprintf(prmfile,"SC_height_start	= %lf \n",prm->ht_start);
       	fprintf(prmfile,"SC_height_end		= %lf \n",prm->ht_end);
       	fprintf(prmfile,"fd1			= %lf \n",prm->fd1);
       	fprintf(prmfile,"fdd1			= %lf \n",prm->fdd1);
       	fprintf(prmfile,"fddd1			= %lf \n",prm->fddd1);

	fclose(prmfile);
}
/***************************************************************************/
/*
difference between variable names (in prm file)
and variables in code.

changed:
offset_video 		off_vid		
chirp_ext 		nextend

prm			sio.h
----------------------------------
Flip_iq 		iqflip
scnd_rng_mig 		srm
rng_spec_wgt 		rhww
rm_rng_band 		pctbw
rm_az_band 		pctbwaz
rng_samp_rate		fs
*/
