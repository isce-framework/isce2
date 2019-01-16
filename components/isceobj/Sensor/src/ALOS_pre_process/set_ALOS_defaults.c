#include"image_sio.h"
#include"lib_functions.h"

/*------------------------------------------------------*/
/* set some defaults					*/
/* replaces virgin.prm					*/
/*------------------------------------------------------*/
void set_ALOS_defaults(struct PRM *prm)
{
	strncpy(prm->input_file,"data.raw",8);	/* input to SAR processor */
	prm->input_file[8] = '\0';
	strncpy(prm->deskew,"n",1);	/* to deskew or not to deskew? */
	prm->deskew[1] = '\0';
	strncpy(prm->iqflip,"n",1);		/* Flip_iq */
	prm->iqflip[1] = '\0';
	strncpy(prm->offset_video,"n",1);		/* off_video */
	prm->offset_video[1] = '\0';
	strncpy(prm->srm,"n",1);			/* scnd_rng_mig */
	prm->srm[1] = '\0';

	prm->num_valid_az	= 9216;
	prm->nrows		= 16384;
	prm->first_line		= 1;
	prm->caltone		= 0.000000;
	prm->st_rng_bin		= 1;
	prm->az_res		= 5;
	prm->nlooks		= 1;
	prm->chirp_ext		= 1000;		/* nextend */
	prm->rhww		= 1.000000;	/* rng_spec_wgt */
	prm->pctbw		= 0.000000;	/* rm_rng_band */
	prm->pctbwaz		= 0.000000;	/* rm_az_band */
	prm->rshift		= 0;
	prm->ashift		= 0;
	prm->stretch_r		= 0.0;
	prm->stretch_a		= 0.0;
	prm->a_stretch_r	= 0.0;
	prm->a_stretch_a	= 0.0;
	prm->first_sample        = 206;
	prm->SC_identity	= 5;
	prm->fs       = 3.200000e+07;			/* rng_samp_rate */
	prm->lambda   = 0.236057;
	prm->near_range		= -1;		/* use -1 as default */
	prm->RE			= -1;		/* use -1 as default */
	prm->num_patches	= 1000;		/* use 1000 as default */
	prm->fd1		= 0.0;
	prm->fdd1		= 0.0;
	prm->fddd1		= 0.0;
	prm->sub_int_r          = 0.0;
	prm->sub_int_a          = 0.0;
} 
/*------------------------------------------------------*/
void print_ALOS_defaults(struct PRM *prm)
{
	fprintf(stderr," \n ALOS default settings *************\n\n");
	fprintf(stderr," led_file = %s \n",prm->led_file);
	fprintf(stderr," input_file = %s \n",prm->input_file);
	fprintf(stderr," num_valid_az = %d \n",prm->num_valid_az);
	fprintf(stderr," nrows = %d \n",prm->nrows);
	fprintf(stderr," first_line = %d \n",prm->first_line);
	fprintf(stderr," deskew = %s \n",prm->deskew);
	fprintf(stderr," caltone = %lf \n",prm->caltone);
	fprintf(stderr," st_rng_bin = %d \n",prm->st_rng_bin);
	fprintf(stderr," Flip_iq(iqflip) = %s \n",prm->iqflip);
	fprintf(stderr," offset_video(off_vid) = %s \n",prm->offset_video);
	fprintf(stderr," az_res = %lf \n",prm->az_res);
	fprintf(stderr," nlooks = %d \n",prm->nlooks);
	fprintf(stderr," chirp_ext(nextend) = %d \n",prm->chirp_ext);
	fprintf(stderr," scnd_rng_mig(srm) = %s \n",prm->srm);
	fprintf(stderr," rng_spec_wgt(rhww) = %lf \n",prm->rhww);
	fprintf(stderr," rm_rng_band(pctbw) = %lf \n",prm->pctbw);
	fprintf(stderr," rm_az_band(pctbwaz) = %lf \n",prm->pctbwaz);
	fprintf(stderr," rshift = %d \n",prm->rshift);
	fprintf(stderr," ashift = %d \n",prm->ashift);
	fprintf(stderr," stretch_r = %lf \n",prm->stretch_r);
	fprintf(stderr," stretch_a = %lf \n",prm->stretch_a);
	fprintf(stderr," a_stretch_r = %lf \n",prm->a_stretch_r);
	fprintf(stderr," a_stretch_a = %lf \n",prm->a_stretch_a);
	fprintf(stderr," first_sample = %d \n",prm->first_sample);
	fprintf(stderr," SC_identity = %d \n",prm->SC_identity);
	fprintf(stderr," rng_samp_rate(fs) = %lf \n",prm->fs);
	fprintf(stderr," near_range = %lf \n",prm->near_range);
} 
/*------------------------------------------------------*/

/* not all variables are called the same in sio.h 	
and the prm file 					

changed
offset_video 		off_video		
chirp_ext 		nextend

PRM			SOI.H
-------------------------------
Flip_iq 		iqflip
scnd_rng_mig 		srm
rng_spec_wgt 		rhww
rm_rng_band 		pctbw
rm_az_band 		pctbwaz
rng_samp_rate		fs
*/
