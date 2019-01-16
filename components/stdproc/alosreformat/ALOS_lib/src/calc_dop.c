/*******************************************************************************
 * Calculate Doppler centroid using the method of Masden 1989                  *  
 * Doppler variations with range are small for ALOS and not calculated         *
 *******************************************************************************/
/********************************************************************************
 * Creator:  Rob Mellors and David T. Sandwell                                  *
 *           (San Diego State University, Scripps Institution of Oceanography)  *
 * Date   :  10/03/2007                                                         *
 ********************************************************************************/
/********************************************************************************
 * Modification history:                                                        *
 * Date:                                                                        *
 * *****************************************************************************/

#include "image_sio.h"
#include "lib_functions.h"
#include "siocomplex.h"

void calc_dop(struct  PRM *prm)
{
	unsigned char *indata;
	int	i, j;
	long	n;
        float *xr, *ac, *sg; 
	double sumd;
	fcomplex	*ai, *bi, *ab;
	fcomplex ctmp;
	FILE	*fin;

	fprintf(stderr,".... calculating doppler for %s\n",prm->input_file);
	fin = fopen(prm->input_file,"r");
	if (fin == NULL) die("can't open",prm->input_file);

/* allocate memory */
	indata = (unsigned char *) malloc(prm->bytes_per_line*sizeof(unsigned char));

	n = prm->good_bytes/2 - prm->first_sample;

	xr = (float *) malloc(n*sizeof(float));
	ac = (float *) malloc(n*sizeof(float));
	sg = (float *) malloc(n*sizeof(float));

	ai = (fcomplex *) malloc(n*sizeof(fcomplex));
	bi = (fcomplex *) malloc(n*sizeof(fcomplex));
	ab = (fcomplex *) malloc(2*n*sizeof(fcomplex));

/* read a line of data from fin (input file, chars) to ai (complex floats) */
	fread(indata, sizeof(unsigned char), prm->bytes_per_line, fin);
	for (i=0; i<n; i++) read_data(ai, indata, i, prm);

/* read remaining lines and set ai = bi 		*/
/* inefficient; could put loops inside each other 	*/
	for (i=prm->first_line; i<prm->num_lines-1; i++){

		if (i/2000 == i/2000.0) fprintf(stderr," Working on line %d \n",i);

		fread(indata, sizeof(unsigned char), prm->bytes_per_line, fin);

		for (j=0; j<n; j++) {
			read_data(bi, indata, j, prm);

			ctmp = Cmul(Conjg(ai[j]), bi[j]);

			ab[j].r = ab[j].r + ctmp.r;
			ab[j].i = ab[j].i + ctmp.i;
			ai[j].r = bi[j].r;
			ai[j].i = bi[j].i;
			}
		}

/* compute the Doppler as a function of range */

        sumd = 0.0;
        for (j=0; j<n; j++){
           	ac[j] =  atan2f(ab[j].i, ab[j].r)/(2.0*M_PI);
           	sumd = sumd + ac[j];
		}

/* now either output the average Doppler or a linear trend fit */

	prm->fd1 = (sumd/(1.0*n))*prm->prf;
        prm->fdd1 = 0.0*prm->prf;
        prm->fddd1 = 0.0*prm->prf;

        fclose(fin);

	free(xr); free(ac); free(sg);
	free(ai); free(bi); free(ab);
	free(indata);
}
/*---------------------------------------------------*/
void read_data(fcomplex *data, unsigned char *indata, int i, struct PRM *prm)
{
int	ii ;

	ii = i + prm->first_sample ; 

	if ((((int)indata[2*ii]) != NULL_DATA) && 
		(((int) indata[2*ii+1]) != NULL_DATA)) {

		data[i].r = ((float) indata[2*ii]) - prm->xmi ;
		data[i].i = ((float) indata[2*ii+1]) - prm->xmq ;

	} else { data[i].r = 0.0 ; data[i].i = 0.0 ; }
}
/*---------------------------------------------------*/
