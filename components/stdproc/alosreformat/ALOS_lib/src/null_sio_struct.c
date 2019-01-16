#include "image_sio.h"
#include "lib_functions.h"

void null_sio_struct(struct PRM *p)
{

	/* characters */
	strncpy(p->input_file,NULL_CHAR,8);
	strncpy(p->SLC_file,NULL_CHAR,8);
	strncpy(p->out_amp_file,NULL_CHAR,8);
	strncpy(p->out_data_file,NULL_CHAR,8);
	strncpy(p->deskew,NULL_CHAR,8);
	strncpy(p->iqflip,NULL_CHAR,8);
	strncpy(p->offset_video,NULL_CHAR,8);
	strncpy(p->srm,NULL_CHAR,8);
	strncpy(p->ref_file,NULL_CHAR,8);
	strncpy(p->led_file,NULL_CHAR,8);
	strncpy(p->orbdir,NULL_CHAR,8);	
	strncpy(p->SLC_file,NULL_CHAR,8);	

	/* ints	*/
	p->debug_flag = NULL_INT;
	p->bytes_per_line = NULL_INT;
	p->good_bytes = NULL_INT;
	p->first_line = NULL_INT;
	p->num_patches = NULL_INT;
	p->first_sample = NULL_INT;
	p->num_valid_az = NULL_INT;
	p->st_rng_bin = NULL_INT;
	p->num_rng_bins = NULL_INT;
	p->chirp_ext = NULL_INT;
	p->nlooks = NULL_INT;
	p->rshift = NULL_INT;
	p->ashift = NULL_INT;
	p->fdc_ystrt = NULL_INT;
	p->fdc_strt = NULL_INT;
	p->rec_start = NULL_INT;
	p->rec_stop = NULL_INT;
	p->SC_identity = NULL_INT;	
	p->ref_identity = NULL_INT;	
	p->nrows = NULL_INT;
	p->num_lines = NULL_INT;
	p->SLC_format = NULL_INT;		

	/* doubles	*/
	p->SC_clock_start = NULL_DOUBLE;
	p->SC_clock_stop = NULL_DOUBLE;	
	p->icu_start = NULL_DOUBLE;	
	p->ref_clock_start = NULL_DOUBLE;
	p->ref_clock_stop = NULL_DOUBLE;
	p->caltone = NULL_DOUBLE;
	p->RE = NULL_DOUBLE;			
	p->rc = NULL_DOUBLE;			
	p->ra = NULL_DOUBLE;			
	p->vel = NULL_DOUBLE;			
	p->ht = NULL_DOUBLE;		
	p->near_range = NULL_DOUBLE;
	p->far_range = NULL_DOUBLE;
	p->prf = NULL_DOUBLE;
	p->xmi = NULL_DOUBLE;
	p->xmq = NULL_DOUBLE;
	p->az_res = NULL_DOUBLE;
	p->fs = NULL_DOUBLE;
	p->chirp_slope = NULL_DOUBLE;
	p->pulsedur = NULL_DOUBLE;
	p->lambda = NULL_DOUBLE;
	p->rhww = NULL_DOUBLE;
	p->pctbw = NULL_DOUBLE;
	p->pctbwaz = NULL_DOUBLE;
	p->fd1 = NULL_DOUBLE;
	p->fdd1 = NULL_DOUBLE;
	p->fddd1 = NULL_DOUBLE;
	p->delr = NULL_DOUBLE;	

	p->sub_int_r = NULL_DOUBLE;
	p->sub_int_a = NULL_DOUBLE;
	p->sub_double = NULL_DOUBLE;
	p->stretch_r = NULL_DOUBLE;
	p->stretch_a = NULL_DOUBLE;
	p->a_stretch_r = NULL_DOUBLE;
	p->a_stretch_a = NULL_DOUBLE;
	p->baseline_start = NULL_DOUBLE;
	p->baseline_end = NULL_DOUBLE;
	p->alpha_start = NULL_DOUBLE;
	p->alpha_end = NULL_DOUBLE;
	p->bpara = NULL_DOUBLE;			
	p->bperp = NULL_DOUBLE;		
};
