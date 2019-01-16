/*--------------------------------------------------------------------*/
/*
	Read parameters into PRM structure from PRM file
	Based on get_params by Evelyn J. Price 
	Modified by RJM
*/
/*--------------------------------------------------------------------*/

#include "image_sio.h"
#include "lib_functions.h"

/*
void get_sio_struct(FILE *, struct PRM *);
void get_string(char *, char *, char *, char *);
void get_int(char *, char *, char *, int *);
void get_double(char *, char *, char *, double *);
*/

void get_sio_struct(FILE *fh, struct PRM *s)
{
char 	name[128], value[128];

if (debug) {
	fprintf(stderr,"get_sio_struct:\n");
	fprintf(stderr,"PRMname   (PRM value)     interpreted value\n");
	}

while(fscanf(fh,"%s = %s \n",name,value) != EOF){

	/* strings */
	if (strcmp(name,"input_file") == 0) get_string(name, "input_file", value, s->input_file);
	if (strcmp(name,"led_file") == 0) get_string(name, "led_file", value, s->led_file);
	if (strcmp(name,"out_amp_file") == 0) get_string(name, "out_amp_file", value, s->out_amp_file);
	if (strcmp(name,"out_data_file") == 0) get_string(name, "out_data_file", value, s->out_data_file);
	if (strcmp(name,"scnd_rng_mig") == 0) get_string(name, "scnd_rng_mig", value, s->srm);
	if (strcmp(name,"deskew") == 0) get_string(name, "deskew", value, s->deskew);
	if (strcmp(name,"Flip_iq") == 0) get_string(name, "Flip_iq", value, s->iqflip);
	if (strcmp(name,"offset_video") == 0) get_string(name, "offset_video", value, s->offset_video);
	if (strcmp(name,"ref_file") == 0)  get_string(name, "ref_file", value, s->ref_file);
	if (strcmp(name,"SLC_file") == 0)  get_string(name, "SLC_file", value, s->SLC_file);
	if (strcmp(name,"orbdir") == 0)  get_string(name, "orbdir", value, s->orbdir);

	/* integers */
	if (strcmp(name,"nrows") == 0) get_int(name, "nrows", value, &s->nrows);
	if (strcmp(name,"num_lines") == 0) get_int(name, "num_lines", value, &s->num_lines);
	if (strcmp(name,"bytes_per_line") == 0) get_int(name, "bytes_per_line", value, &s->bytes_per_line);
	if (strcmp(name,"good_bytes_per_line") == 0) get_int(name, "good_bytes_per_line", value, &s->good_bytes);
	if (strcmp(name,"first_line") == 0) get_int(name, "first_line", value, &s->first_line);
	if (strcmp(name,"num_patches") == 0) get_int(name, "num_patches", value, &s->num_patches);
	if (strcmp(name,"first_sample") == 0) get_int(name, "first_sample", value, &s->first_sample);
	if (strcmp(name,"num_valid_az") == 0) get_int(name, "num_valid_az", value, &s->num_valid_az);
	if (strcmp(name,"SC_identity") == 0) get_int(name, "SC_identity", value, &s->SC_identity);
	if (strcmp(name,"chirp_ext") == 0) get_int(name, "chirp_ext", value, &s->chirp_ext);
	if (strcmp(name,"st_rng_bin") == 0) get_int(name, "st_rng_bin", value, &s->st_rng_bin);
	if (strcmp(name,"num_rng_bins") == 0) get_int(name, "num_rng_bins", value, &s->num_rng_bins);
	if (strcmp(name,"ref_identity") == 0) get_int(name, "ref_identity", value, &s->ref_identity);
	if (strcmp(name,"nlooks") == 0) get_int(name, "nlooks", value, &s->nlooks);
	if (strcmp(name,"rshift") == 0) get_int(name, "rshift", value, &s->rshift);
	if (strcmp(name,"ashift") == 0) get_int(name, "ashift", value, &s->ashift);
	/* backwards compatibility for xshift/rshift yshift/ashift */
	if (strcmp(name,"xshift") == 0) get_int(name, "rshift", value, &s->rshift);
	if (strcmp(name,"yshift") == 0) get_int(name, "ashift", value, &s->ashift);
	if (strcmp(name,"SLC_format") == 0) get_int(name, "SLC_format", value, &s->SLC_format);

	/* doubles */
	if (strcmp(name, "SC_clock_start") == 0) get_double(name,"SC_clock_start",value,&s->SC_clock_start);
	if (strcmp(name, "SC_clock_stop") == 0) get_double(name,"SC_clock_stop", value, &s->SC_clock_stop);
	if (strcmp(name, "icu_start") == 0) get_double(name,"icu_start", value, &s->icu_start);
	if (strcmp(name, "ref_clock_start") == 0) get_double(name,"ref_clock_start", value, &s->ref_clock_start);
	if (strcmp(name, "ref_clock_stop") == 0) get_double(name,"ref_clock_stop", value, &s->ref_clock_stop);
	if (strcmp(name, "caltone") == 0) get_double(name,"caltone", value, &s->caltone);
	if (strcmp(name, "earth_radius") == 0) get_double(name,"earth_radius", value, &s->RE);
        if (strcmp(name, "equatorial_radius") == 0) get_double(name,"earth_radius", value, &s->ra);
        if (strcmp(name, "polar_radius") == 0) get_double(name,"earth_radius", value, &s->rc);
	if (strcmp(name, "SC_vel") == 0) get_double(name,"SC_vel", value, &s->vel);
	if (strcmp(name, "SC_height") == 0) get_double(name,"SC_height", value, &s->ht);
	if (strcmp(name, "near_range") == 0) get_double(name,"near_range", value, &s->near_range);
	if (strcmp(name, "PRF") == 0) get_double(name,"PRF", value, &s->prf);
	if (strcmp(name, "I_mean") == 0) get_double(name,"I_mean", value, &s->xmi);
	if (strcmp(name, "Q_mean") == 0) get_double(name,"Q_mean", value, &s->xmq);
	if (strcmp(name, "az_res") == 0) get_double(name,"az_res", value, &s->az_res);
	if (strcmp(name, "rng_samp_rate") == 0) get_double(name,"rng_samp_rate", value, &s->fs);
	if (strcmp(name, "chirp_slope") == 0) get_double(name,"chirp_slope", value, &s->chirp_slope);
	if (strcmp(name, "pulse_dur") == 0) get_double(name,"pulse_dur", value, &s->pulsedur);
	if (strcmp(name, "radar_wavelength") == 0) get_double(name,"radar_wavelength", value, &s->lambda);
	if (strcmp(name, "rng_spec_wgt") == 0) get_double(name,"rng_spec_wgt", value, &s->rhww);
	if (strcmp(name, "rm_rng_band") == 0) get_double(name,"rm_rng_band", value, &s->pctbw);
	if (strcmp(name, "rm_az_band") == 0) get_double(name,"rm_az_band", value, &s->pctbwaz);
	if (strcmp(name, "fd1") == 0) get_double(name,"fd1", value, &s->fd1);
	if (strcmp(name, "fdd1") == 0) get_double(name,"fdd1", value, &s->fdd1);
	if (strcmp(name, "fddd1") == 0) get_double(name,"fddd1", value, &s->fddd1);
	if (strcmp(name, "sub_int_r") == 0) get_double(name,"sub_int_r", value, &s->sub_int_r);
	if (strcmp(name, "sub_int_a") == 0) get_double(name,"sub_int_a", value, &s->sub_int_a);
	if (strcmp(name, "stretch_r") == 0) get_double(name,"stretch_r", value, &s->stretch_r);
	if (strcmp(name, "stretch_a") == 0) get_double(name,"stretch_a", value, &s->stretch_a);
	if (strcmp(name, "a_stretch_r") == 0) get_double(name,"a_stretch_r", value, &s->a_stretch_r);
	if (strcmp(name, "a_stretch_a") == 0) get_double(name,"a_stretch_a", value, &s->a_stretch_a);
	if (strcmp(name, "baseline_start") == 0) get_double(name,"baseline_start", value, &s->baseline_start);
	if (strcmp(name, "alpha_start") == 0) get_double(name,"alpha_start", value, &s->alpha_start);
	if (strcmp(name, "baseline_end") == 0) get_double(name,"baseline_end", value, &s->baseline_end);
	if (strcmp(name, "alpha_end") == 0) get_double(name,"alpha_end", value, &s->alpha_end);

	}
}
/*--------------------------------------------------------------------------------*/
void get_string(char *s1, char *name, char *value, char *s2)
{
	strcpy(s2,value);
	if (debug==1) fprintf(stderr," %s (%s) = %s\n",s1,name,value);
}
/*--------------------------------------------------------------------------------*/
void get_int(char *s1, char *name, char *value, int *iparam)
{
	*iparam = atoi(value);
	if (debug==1) fprintf(stderr," %s (%s) = %s (%d)\n",s1,name,value,*iparam);
}
/*--------------------------------------------------------------------------------*/
void get_double(char *s1, char *name, char *value, double *param)
{
	*param = atof(value);
	if (debug==1) fprintf(stderr," %s (%s) = %s (%lf)\n",s1,name,value,*param);
}
/*--------------------------------------------------------------------------------*/
