/***************************************************************************/
/* read_ALOS_data reads an ALOS IMG file containing raw signal data       */
/* and creates a raw-file and PRM-file suitable for our esarp processor.   */
/* The program skips the first 720 bytes of the IMG file but copies the    */
/* remaining data to the IMG.raw file after checking and fixing problems.  */
/* The first record is read to determine the linelength, starting PRF,     */
/* and near_range.  If the line length or PRF change then the program      */
/* halts.  If the near_range changes then the lines are shifted and        */
/* unconstrained values at the ends are set to NULL_DATA (15 or 16).       */
/* (random sequence of 15 and 16's)					   */
/* During this processing the available parameters are added to the        */
/* PRM-file.                                                               */
/***************************************************************************/

/***************************************************************************
 * Creator:  David T. Sandwell and Meng Wei                                *
 *           (Scripps Institution of Oceanography)                         *
 * Date   :  06/29/2006                                                     *
 ***************************************************************************/

/***************************************************************************
 * Modification history:                                                   *
 *                                                                         *
 * DATE                                                                    *
 *                                                                         *
 * 06/29/2006   added the near_range as an optional command-line argument  *
 * 02/19/2007   added the ability to remove duplicate lines		   *
 * 03/07/2007   more robust to bad data in file				   *
 * 03/26/2007   ability to swap bytes (read on PC) RJM			   *
 * 03/28/2007	part of subroutine		   RJM		           *
 * removed set n_azimuth to 9000 rather than default			   *
 * 07/17/08     creates new file when prf changes RJM			   *
 * 07/17/08     reformatted; added functions      RJM                      *
 ***************************************************************************/

/*
the data header information is read into the structure dfd
the line prefix information is read into sdr
Values read here (and or generated) are:

num_rng_bins bytes_per_line good_bytes_per_line 
PRF pulse_dur near_range
num_lines num_patches 
SC_clock_start SC_clock_stop
*/
/* fast random number generator */

#include "image_sio.h"
#include "lib_functions.h"
#define ZERO_VALUE (char)(63 + rand() % 2)
#define clip127(A) (((A) > 127) ? 127 : (((A) < 0) ? 0 : A))
#define znew   (int) (z=36969*(z&65535)+(z>>16))
typedef unsigned long UL;
 static UL z=362436069, t[256];

 void settable(UL i1)
 { int i; z=i1;
 for(i=0;i<256;i=i+1)  t[i]=znew;
 }

void swap_ALOS_data_info(struct sardata_info *sdr);
long read_sardata_info(FILE *, struct PRM *, int *, int *);
void print_params(struct PRM *prm);
int assign_sardata_params(struct PRM *, int, int *, int *);
int check_shift(struct PRM *, int *, int *, int *, int, int);
int set_file_position(FILE *, long *, int);
int reset_params(struct PRM *prm, long *, int *, int *);
int fill_shift_data(int, int, int, int, int, char *, char *, FILE *);
int handle_prf_change(struct PRM *, FILE *, long *, int); 
void change_dynamic_range(char *data, long length);

static struct sardata_record r1;
static struct sardata_descriptor dfd;
static struct sardata_info sdr;

long read_ALOS_data (FILE *imagefile, FILE *outfile, struct PRM *prm, long *byte_offset, struct resamp_info *rspi, int nPRF) {

	char 	*data, *shift_data;
        int 	record_length0;		/* length of record read at start of file */
	int	record_length1;		/* length of record read in file 	*/
	int start_sdr_rec_len = 0; /* sdr record length for fisrt record */
	int slant_range_old = 0;   /* slant range of previous record */
	int	line_suffix_size;	/* number of bytes after data 		*/
	int	data_length;		/* bytes of data			*/
        int 	n, m, ishift, shift, shift0, npatch_max;
	int	header_size, line_prefix_size;

        double 	get_clock();

	settable(12345);

	if (debug) fprintf(stderr,".... reading header \n");

    //here we still get sdr from the first data line no matter whether prf changes. 
    //this sdr is used to initialize record_length0 in assign_sardata_params, which 
    //is used at line 152 to check if record_length changed.
    //I think we should get sdr from first prf-change data line for the output of prf-change file.
    //Cunren Liang. 02-DEC-2019


	/* read header information */
	read_sardata_info(imagefile, prm, &header_size, &line_prefix_size);

	/* calculate parameters (data length, range bins, etc) */
	assign_sardata_params(prm, line_prefix_size, &line_suffix_size, &record_length0);

	//fprintf(stderr,"before allocate data\n");

	/* allocate data */
	if ((data = (char *) malloc(record_length0)) == NULL) die("couldn't allocate memory for input indata.\n","");
	//fprintf(stderr,"after allocate length0 data\n");

	if ((shift_data = (char *) malloc(record_length0)) == NULL) die("couldn't allocate memory for input indata.\n","");

	//fprintf(stderr,"after allocate data\n");

	/* if byte_offset < 0 this is the first time through 	*/
	/* if prf change has occurred, set file to byte_offset  */
	set_file_position(imagefile, byte_offset, header_size);

	if (verbose) fprintf(stderr,".... reading data (byte %ld) \n",ftell(imagefile));

	shift0 = 0;
	n = 1;
	m = 2;//first line sequence_number

	/* read the rest of the file */
	while ( (fread((void *) &sdr,sizeof(struct sardata_info), 1, imagefile)) == 1 ) {
        	n++;

		/* checks for little endian/ big endian */
		if (swap) swap_ALOS_data_info(&sdr);


        if (n == 2)
        	rspi->frame_counter_start[nPRF] = sdr.frame_counter;



		/* if this is partway through the file due to prf change, reset sequence,
		 * PRF, and near_range */
		if (n == 2)
			start_sdr_rec_len = sdr.record_length;

		if ((*byte_offset > 0) && (n == 2))
			reset_params(prm, byte_offset, &n, &m);

		if (sdr.record_length != start_sdr_rec_len) {
			printf(" ***** warning sdr.record_length error %d \n", sdr.record_length);
			sdr.record_length = start_sdr_rec_len;
			sdr.PRF = prm->prf;
			sdr.slant_range = slant_range_old;
		}
		if (sdr.sequence_number != n)
			printf(" missing line: n, seq# %d %d \n", n, sdr.sequence_number);


		/* check for changes in record_length and PRF */
          	record_length1 = sdr.record_length - line_prefix_size;
          	if (record_length0  != record_length1)  die("record_length changed",""); 

		/* if prf changes, close file and set byte_offset */
          	if ((sdr.PRF) != prm->prf) {
			handle_prf_change(prm, imagefile, byte_offset, n); 
			n-=1;
             		break;
          		}
          	rspi->frame_counter_end[nPRF] = sdr.frame_counter;

		/* check shift to see if it varies from beginning or from command line value */
		check_shift(prm, &shift, &ishift, &shift0, record_length1, 0);
		
		if ((verbose) && (n/2000.0 == n/2000)) {
			fprintf(stderr," Working on line %d prf %f record length %d slant_range %d \n"
				,sdr.sequence_number, 0.001*sdr.PRF, record_length1, sdr.slant_range);
			}

		/* read data (and trailing bytes) */
          	if ( fread ((char *) data, record_length1, (size_t) 1, imagefile) != 1 ) break;

		data_length = record_length1;
		slant_range_old = sdr.slant_range;

		/* write line header to output data  */
		//header is not written to output
        //fwrite((void *) &sdr, line_prefix_size, 1, outfile);

		/* write data */
	  	if (shift == 0) {
	  		change_dynamic_range(data, data_length);
			fwrite((char *) data, data_length, 1, outfile); 
			/* if data is shifted, fill in with data values of NULL_DATA at start or end*/
			} else if (shift != 0) {
			fill_shift_data(shift, ishift, data_length, line_suffix_size, record_length1, data, shift_data, outfile); 
			}
		}

    //we are not writing out line prefix data, need to correct these parameters
	//as they are used in doppler computation.
    prm->first_sample = 0;
    prm->bytes_per_line -= line_prefix_size;
    prm->good_bytes -= line_prefix_size;
      
	/* calculate end time and fix prf */
	prm->prf = 0.001*prm->prf;

    //this is the sdr of the first prf-change data line, should seek back to get last sdr to be used here.
	prm->SC_clock_stop =  get_clock(sdr, tbias);

	/* m is non-zero only in the event of a prf change */
	//not correct if PRF changes, so I updated it here.
	prm->num_lines = n - m + 1;

	/* calculate the maximum number of patches and use that if the default is set to 1000 */
        npatch_max = (int)((1.0*n)/(1.0*prm->num_valid_az));
	if(npatch_max < prm->num_patches) prm->num_patches = npatch_max;

	if (prm->num_lines == 0) prm->num_lines = 1;

    prm->xmi = 63.5;
    prm->xmq = 63.5;

	rspi->prf[nPRF] = prm->prf;
	rspi->SC_clock_start[nPRF] = prm->SC_clock_start;
	rspi->num_lines[nPRF] = prm->num_lines;
	rspi->num_bins[nPRF] = prm->bytes_per_line/(2*sizeof(char));


	if (verbose) print_params(prm); 

	free(data);
	free(shift_data);
	fclose (outfile);

	return(*byte_offset);
}
/***************************************************************************/
double get_clock(struct sardata_info sdr, double tbias)
{
double	nsd, time;

	nsd = 24.0*60.0*60.0;	/* seconds in a day */

	time = ((double) sdr.sensor_acquisition_year)*1000 +
		(double) sdr.sensor_acquisition_DOY +
		(double) sdr.sensor_acquisition_msecs_day/1000.0/86400.0 +
		tbias/86400.0;

	return(time);
}
/***************************************************************************/
void print_params(struct PRM *prm)
{
	fprintf(stdout,"input_file		= %s \n",prm->input_file);
	fprintf(stdout,"num_rng_bins		= %d \n",prm->num_rng_bins);
	fprintf(stdout,"bytes_per_line		= %d \n",prm->bytes_per_line);
	fprintf(stdout,"good_bytes_per_line	= %d \n",prm->good_bytes);
	fprintf(stdout,"first_sample		= %d \n",prm->first_sample);
	fprintf(stdout,"PRF			= %f \n",prm->prf);
	fprintf(stdout,"pulse_dur		= %e \n",prm->pulsedur);
	fprintf(stdout,"near_range		= %f \n",prm->near_range);
	fprintf(stdout,"num_lines		= %d \n",prm->num_lines);
	fprintf(stdout,"num_patches		= %d \n",prm->num_patches);
       	fprintf(stdout,"SC_clock_start		= %16.10lf \n",prm->SC_clock_start);
       	fprintf(stdout,"SC_clock_stop		= %16.10lf \n",prm->SC_clock_stop);
}
/***************************************************************************/
long read_sardata_info(FILE *imagefile, struct PRM *prm, int *header_size, int *line_prefix_size)
{
long nitems;

	*header_size = sizeof(struct sardata_record) + sizeof(struct sardata_descriptor);
	*line_prefix_size = sizeof(struct sardata_info);

	if (*header_size != 720) die("header size is not 720 bytes\n","");
	if (*line_prefix_size != 412) die("header size is not 720 bytes\n","");

	if (debug) fprintf(stderr," header_size %d line_prefix_size %d swap data %d\n", *header_size, *line_prefix_size, swap);

	/* make sure that we are at the beginning */
	/* re-read header even if resetting after a PRF change */
	 rewind(imagefile);

	if (verbose) fprintf(stderr,".... reading header (byte %ld) \n",ftell(imagefile));

	/* data processed before Sept 15, 2006 have a timing bias of 0.9 s */
	/* data processed after this data have a smaller bias 0.0 s */

	nitems = fread((void *) &r1, sizeof(struct sardata_record), 1, imagefile);
	if (debug) { 
		fprintf(stderr,SARDATA_RECORD_WCS,SARDATA_RECORD_RVL(&r1));
		fprintf(stderr," read %ld bytes at position %ld\n", (sizeof(struct sardata_record)), ftell(imagefile));
		}

	nitems = fread((void *) &dfd, sizeof(struct sardata_descriptor), 1, imagefile);
	if (debug) {
		fprintf(stderr,SARDATA_DESCRIPTOR_WCS,SARDATA_DESCRIPTOR_RVL(&dfd));
		fprintf(stderr," read %ld bytes at position %ld\n", (sizeof(struct sardata_descriptor)), ftell(imagefile));
		}

	nitems = fread((void *) &sdr, sizeof(struct sardata_info), 1, imagefile);
	if (debug) fprintf(stderr," read %ld bytes at position %ld\n", (sizeof(struct sardata_info)), ftell(imagefile));

	/* swap data little end/ big end if needed */
	if (swap) swap_ALOS_data_info(&sdr);

	if (debug) fprintf(stderr,SARDATA__WCS,SARDATA_RVL(sdr));

	return(nitems);
}
/***************************************************************************/
int assign_sardata_params(struct PRM *prm, int line_prefix_size, int *line_suffix_size, int *record_length0)
{
double get_clock();

	prm->prf = sdr.PRF;
	prm->pulsedur = (1e-9)*sdr.chirp_length;

	*record_length0 = sdr.record_length - line_prefix_size;

	prm->SC_clock_start =  get_clock(sdr, tbias);

	/* record_length is 21100 */
	/* beginning of line has a 412 byte prefix */
	/* end of line has a 80 byte (40 pixels) suffix (right-fill pixels)*/
	/* record_length0 (data length) is (20688 - 412) = 20276 */
	/* n_data_pixels  10304 */
	/* 2 bytes per pixel */
	/* 412 bytes + (2*10304) bytes + (40*2) bytes  = 21100 bytes*/

	prm->good_bytes = 2*sdr.n_data_pixels + line_prefix_size;
	prm->num_rng_bins = sdr.n_data_pixels + prm->chirp_ext;		/* chirp_ext formerly nextend */
	prm->bytes_per_line = sdr.record_length;
	
	*line_suffix_size = sdr.record_length - prm->good_bytes;

	if (prm->near_range < 0) prm->near_range = sdr.slant_range; 

	if (*record_length0 > 50000) {
		fprintf(stderr, "**** record_length is %d !\n", *record_length0);
		die("expect something like 21100 .... try -swap option?\n","exiting");
		}

	return(EXIT_SUCCESS);
}
/***************************************************************************/
int check_shift(struct PRM *prm, int *shift, int *ishift, int *shift0, int record_length1, int ALOS_format)
{
        *shift = 2*floor(0.5 + (sdr.slant_range - prm->near_range)/(0.5*SOL/prm->fs));
        *ishift = abs(*shift);

         if (*ishift > record_length1) { 
          	printf(" end: shift exceeds data window %d \n", *shift);
		die("exitting","");
          	}

          if(*shift != *shift0) {

	    	if(ALOS_format==0)
	    	printf(" near_range, shift = %d %d , at frame_counter: %d, line number: %d\n", sdr.slant_range, *shift, sdr.frame_counter, sdr.sequence_number-1);
            if(ALOS_format==1)
	    	printf(" near_range, shift = %d %d\n", sdr.slant_range, *shift);


            	*shift0 = *shift;
	  	}

	return(EXIT_SUCCESS);
}
/***************************************************************************/
int set_file_position(FILE *imagefile, long *byte_offset, int header_size)
{
	if (*byte_offset < 0) {
		*byte_offset = 0;
		rewind(imagefile);
		fseek(imagefile, header_size, SEEK_SET);
		} else {
		fseek(imagefile, *byte_offset, SEEK_SET);
		}

	return(EXIT_SUCCESS);
}
/***************************************************************************/
int reset_params(struct PRM *prm, long *byte_offset, int *n, int *m) {
	double get_clock();

	prm->SC_clock_start =  get_clock(sdr, tbias);
	prm->prf = sdr.PRF;
	//comment out so that all data files with different prfs can be aligned at the same starting range
	//prm->near_range = sdr.slant_range;
	*n = sdr.sequence_number;
	*m = *n;
	*byte_offset = 0;
	if (verbose) {
		fprintf(stderr, " new parameters: \n sequence number %d \n PRF  %f\n near_range  %lf\n", *n, 0.001 * prm->prf,
		        prm->near_range);
	}
	return (EXIT_SUCCESS);
}
/***************************************************************************/
int fill_shift_data(int shift, int ishift, int data_length, 
	int line_suffix_size, int record_length1, char *data, char *shift_data, FILE *outfile)
{
int	k;

	/* NULL_DATA = 15; znew randonly is 0 or 1			      */
       	if (shift > 0) {					
         	for (k = 0; k < ishift; k++) shift_data[k] = NULL_DATA+znew%2;
            	for (k = 0; k < data_length - ishift; k++) shift_data[k + ishift] = data[k];
		}

	/* if data is shifted, fill in with data vlues of NULL_DATA at end */
	  if ( shift < 0) {
            	for (k = 0; k < data_length - ishift - line_suffix_size; k++) shift_data[k] = data[k+ishift];
            	for (k = data_length - ishift - line_suffix_size; k < record_length1; k++ ) shift_data[k] = NULL_DATA+znew%2;
          	}

	/* write the shifted data out */
    change_dynamic_range(shift_data, data_length);
        fwrite((char *) shift_data, data_length, 1, outfile);

	return(EXIT_SUCCESS);
} 
/***************************************************************************/
int handle_prf_change(struct PRM *prm, FILE *imagefile, long *byte_offset, int n) 
{
	//prm->num_lines = n;

	/* skip back to beginning of the line */
	fseek(imagefile, -1*sizeof(struct sardata_info), SEEK_CUR);

	/* define byte_offset */
	*byte_offset = ftell(imagefile);

	/* tell the world */
	printf(" *** PRF changed from %lf to  %lf  at line %d (byte %ld)\n", (0.001*prm->prf),(0.001*sdr.PRF), n-1, *byte_offset);
    //    printf(" end: PRF changed from %lf to  %lf  at line %d \n", (0.001*prm->prf),(0.001*sdr.PRF), n);

	return(EXIT_SUCCESS);
}
/***************************************************************************/


void change_dynamic_range(char *data, long length){

  long i;
  
  for(i = 0; i < length; i++)
  	//THIS SHOULD NOT AFFECT DOPPLER COMPUTATION (SUCH AS IN calc_dop.c), BECAUSE
  	// 1. IQ BIAS IS REMOVED BEFORE COMPUTATION OF DOPPLER.
  	// 2. 2.0 WILL BE CANCELLED OUT IN atan2f().
  	// 3. actual computation results also verified this (even if there is a difference, it is about 0.* Hz)
  	//data[i] = (unsigned char)clip127(rintf(2. * (data[i] - 15.5) + 63.5));
    data[i] = (unsigned char)clip127(rintf(2.0 * (data[i] - 15.5)  + ZERO_VALUE));

}










