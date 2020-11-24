/***************************************************************************/
/* read_ALOSE_data reads an ERSDAC ALOS file containing raw signal data    */
/* and creates a raw-file and PRM-file suitable for our esarp processor.   */
/* The program skips the first 16252 bytes of the .raw file but copies the */
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
 * Creator:  David T. Sandwell, Meng Wei, Jeff Bytof                       *
 *           (Scripps Institution of Oceanography)                         *
 * 	     Rob Mellors, SDSU
 * Date   :  06/29/2006                                                    *
 * based on read_ALOS_data
 * 12/12/09     format changes for RESTEC files   Jeff Bytof               *
 * 15-Apr-2010  Replaced ALOS identifier with ALOSE  Jeff Bytof            * 
 **************************************************************************/

/********************************************************************************
This program has been upgraded to handle the ALOS-1 PRF change issue.
BUT HAS NOT BEEN TESTED YET!!!
*********************************************************************************/

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
/*
#define znew   (int) (z=36969*(z&65535)+(z>>16))
typedef unsigned long UL;
 static UL z=362436069, t[256];

 void settable(UL i1)
 { int i; z=i1;
 for(i=0;i<256;i=i+1)  t[i]=znew;
 }
*/

long read_sardata_info_ALOSE(FILE *, struct PRM *, int *, int *);
int assign_sardata_params_ALOSE(struct PRM *, int, int *, int *);

void swap_ALOS_data_info(struct sardata_info_ALOSE *sdr);
void settable(unsigned long);
void print_params(struct PRM *prm);
int check_shift(struct PRM *, int *, int *, int *, int, int);
int set_file_position(FILE *, long *, int);
int reset_params(struct PRM *prm, long *, int *, int *);
int fill_shift_data(int, int, int, int, int, char *, char *, FILE *);
int handle_prf_change_ALOSE(struct PRM *, FILE *, long *, int); 
void change_dynamic_range(char *data, long length);

static struct sardata_record r1;
static struct sardata_descriptor_ALOSE dfd;
static struct sardata_info_ALOSE sdr;

/*
differences in include file from ALOS AUIG 
struct sardata_descriptor_ALOSE 
SARDATA_DESCRIPTOR_WCS_ALOSE 
SARDATA_DESCRIPTOR_RVL_ALOSE(SP)

struct sardata_info_ALOSE 
SARDATA__WCS_ALOSE
SARDATA_RVL_ALOSE(SP)
*/
long read_ALOSE_data (FILE *imagefile, FILE *outfile, struct PRM *prm, long *byte_offset, struct resamp_info *rspi, int nPRF) {

	char 	*data_fbd, *data, *shift_data;
        int 	record_length0;		/* length of record read at start of file */
	int	record_length1;		/* length of record read in file 	*/
	int start_sdr_rec_len = 0; /* sdr record length for fisrt record */
	int slant_range_old = 0;   /* slant range of previous record */
	int	line_suffix_size;	/* number of bytes after data 		*/
	int	data_length;		/* bytes of data			*/
        int 	k, n, m, ishift, shift, shift0;
	int	header_size, line_prefix_size;
	double pri;

        double 	get_clock_ALOSE();

	settable(12345);

	if (verbose) fprintf(stderr,".... reading header \n");

	//here we still get sdr from the first data line no matter whether prf changes. 
	//this sdr is used to initialize record_length0 in assign_sardata_params, which 
	//is used at line 152 to check if record_length changed.
	//I think we should get sdr from first prf-change data line for the output of prf-change file.
	//Cunren Liang. 02-DEC-2019


	/* read header information */
	read_sardata_info_ALOSE(imagefile, prm, &header_size, &line_prefix_size);
	if (verbose) fprintf(stderr,".... reading header %d %d\n", header_size, line_prefix_size);

	/* calculate parameters (data length, range bins, etc) */
	assign_sardata_params_ALOSE(prm, line_prefix_size, &line_suffix_size, &record_length0);

	/* allocate data */

        if (verbose) printf( "record_length0 = %d \n", record_length0 );  /* bytof */

	if ((data = (char *) malloc(record_length0)) == NULL) die("couldn't allocate memory for input indata.\n","");
	if(sdr.receive_polarization == 2) if ((data_fbd = (char *) malloc(record_length0)) == NULL) die("couldn't allocate memory for input indata.\n","");

	if ((shift_data = (char *) malloc(record_length0)) == NULL) die("couldn't allocate memory for input indata.\n","");

	/* if byte_offset < 0 this is the first time through 	*/
	/* if prf change has occurred, set file to byte_offset  */
	set_file_position(imagefile, byte_offset, header_size);

	if (verbose) fprintf(stderr,".... reading data (byte %ld) \n",ftell(imagefile));

	shift0 = 0;
	n = 1;
	m = 2;//first line sequence_number

	/* read the rest of the file */
	while ( (fread((void *) &sdr,sizeof(struct sardata_info_ALOSE), 1, imagefile)) == 1 ) {
        	n++;

		/* checks for little endian/ big endian */
		if (swap) swap_ALOS_data_info(&sdr);


        if (n == 2)
        	//rspi->frame_counter_start[nPRF] = sdr.frame_counter;
            //unfortunately restec format does not have this info, so we are not able to adjust time
            rspi->frame_counter_start[nPRF] = 0;



		/* if this is partway through the file due to prf change, reset sequence, PRF, and near_range */
		if (n == 2)
			start_sdr_rec_len = sdr.record_length;

		if ((*byte_offset > 0)  && (n == 2)) reset_params(prm, byte_offset, &n, &m);

		if (sdr.record_length != start_sdr_rec_len) {
			printf(" ***** warning sdr.record_length error %d \n", sdr.record_length);
			sdr.record_length = start_sdr_rec_len;
			sdr.PRF = prm->prf;
			sdr.slant_range = slant_range_old;
		}
          	if (sdr.sequence_number != n) printf(" missing line: n, seq# %d %d \n", n, sdr.sequence_number);

		/* check for changes in record_length and PRF */
          	record_length1 = sdr.record_length - line_prefix_size;
          	if (record_length0  != record_length1)  die("record_length changed",""); 

		/* if prf changes, close file and set byte_offset */
          	if ((sdr.PRF) != prm->prf) {
			handle_prf_change_ALOSE(prm, imagefile, byte_offset, n); 
			n-=1;
             		break;
          		}
          	//rspi->frame_counter_end[nPRF] = sdr.frame_counter;
          	//unfortunately restec format does not have this info, so we are not able to adjust time
          	rspi->frame_counter_end[nPRF] = 0;

		/* check shift to see if it varies from beginning or from command line value */
		check_shift(prm, &shift, &ishift, &shift0, record_length1, 1);
		
		if ((verbose) && (n/2000.0 == n/2000)) {
			fprintf(stderr," Working on line %d prf %f record length %d slant_range %d \n"
				,sdr.sequence_number, 0.001*sdr.PRF, record_length1, sdr.slant_range);
			}

		/* read data (and trailing bytes) */
          	if ( fread ((char *) data, record_length1, (size_t) 1, imagefile) != 1 ) break;

		data_length = record_length1;
		slant_range_old = sdr.slant_range;

		/* write line header to output data  */
                /* PSA - turning off headers
          	fwrite((void *) &sdr, line_prefix_size, 1, outfile); */

		/* write either fbd or fbs */

		if(sdr.receive_polarization == 2) {
			for (k=0;k<data_length;k=k+4) {
				data_fbd[k/2]=data[k];
				data_fbd[k/2+1]=data[k+1];
			}
			/* write fbd data */
	  		if (shift == 0) {
	  			change_dynamic_range(data_fbd, data_length/2);
				fwrite((char *) data_fbd, data_length/2, 1, outfile); 
				} else if (shift != 0) {
				fill_shift_data(shift, ishift, data_length/2, line_suffix_size, record_length1, data_fbd, shift_data, outfile); 
			}	
		}
		else {
			/* write fbs data */
	  		if (shift == 0) {
	  			change_dynamic_range(data, data_length);
				fwrite((char *) data, data_length, 1, outfile); 
				} else if (shift != 0) {
				fill_shift_data(shift, ishift, data_length, line_suffix_size, record_length1, data, shift_data, outfile); 
			}	
		}
	}

    //we are not writing out line prefix data, need to correct these parameters
	//as they are used in doppler computation.
    prm->first_sample = 0;
    prm->bytes_per_line -= line_prefix_size;
    prm->good_bytes -= line_prefix_size;
 
	//this is the sdr of the first prf-change data line, should seek back to get last sdr to be used here.
	/* calculate end time */
	prm->SC_clock_stop =  get_clock_ALOSE(sdr, tbias);

	/* m is non-zero only in the event of a prf change */
	//not correct if PRF changes, so I updated it here.
	prm->num_lines = n - m + 1;
	prm->num_patches = (int)((1.0*n)/(1.0*prm->num_valid_az));
	if (prm->num_lines == 0) prm->num_lines = 1;

	/* compute the PRI and round to the nearest integer microsecond then the prf=1./pri */

	pri = (int) (1.e6*86400.*(prm->SC_clock_stop - prm->SC_clock_start)/(prm->num_lines-2.5)+.5);
	prm->prf = 1.e3/pri;
	

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
double get_clock_ALOSE(struct sardata_info_ALOSE sdr, double tbias)
{
double	nsd, time;

	nsd = 24.0*60.0*60.0;

	time = ((double) sdr.sensor_acquisition_year)*1000 +
		(double) sdr.sensor_acquisition_DOY +
		(double) sdr.sensor_acquisition_msecs_day/1000.0/86400.0 +
		tbias/86400.0;

        if (debug) printf( "get_clock: time = %f \n", time );

	return(time);
}
/***************************************************************************/
long read_sardata_info_ALOSE(FILE *imagefile, struct PRM *prm, int *header_size, int *line_prefix_size)
{
long nitems;

        if(debug) print_params( prm );   /* bytof */

	*header_size = sizeof(struct sardata_record) + sizeof(struct sardata_descriptor_ALOSE);

        if(debug) printf( "header_size = %d \n", *header_size );    /* bytof */

	*line_prefix_size = sizeof(struct sardata_info_ALOSE);

        if(debug) printf( "*line_prefix_size = %d \n", *line_prefix_size );  /* bytof */

	if (*header_size != 16252) die("header size is not 16252 bytes\n","");  /* restec format change - bytof */

	if (*line_prefix_size != 292) die("line_prefix_size is not 292 bytes\n",""); /* bytof */

	if (debug) fprintf(stderr," header_size %d line_prefix_size %d swap data %d\n", *header_size, *line_prefix_size, swap);

	/* make sure that we are at the beginning */
	/* re-read header even if resetting after a PRF change */
	 rewind(imagefile);

	if (verbose) fprintf(stderr,".... reading header (byte %ld) \n",ftell(imagefile));

	nitems = fread((void *) &r1, sizeof(struct sardata_record), 1, imagefile);

        if(debug) printf( "nitems = %ld \n", nitems );  /* bytof */

	if (debug) { 
		fprintf(stderr,SARDATA_RECORD_WCS,SARDATA_RECORD_RVL(&r1));
		fprintf(stderr," read %ld bytes at position %ld\n", (sizeof(struct sardata_record)), ftell(imagefile));
		}

	nitems = fread((void *) &dfd, sizeof(struct sardata_descriptor_ALOSE), 1, imagefile);
	if (debug) {
		fprintf(stderr,SARDATA_DESCRIPTOR_WCS_ALOSE,SARDATA_DESCRIPTOR_RVL_ALOSE(&dfd));
		fprintf(stderr," read %ld bytes at position %ld\n", (sizeof(struct sardata_descriptor_ALOSE)), ftell(imagefile));
		}

	nitems = fread((void *) &sdr, sizeof(struct sardata_info_ALOSE), 1, imagefile);
	if (debug) fprintf(stderr," read %ld bytes at position %ld\n", (sizeof(struct sardata_info_ALOSE)), ftell(imagefile));

	/* swap data little end/ big end if needed */
	if (swap) swap_ALOS_data_info(&sdr);

	if (debug) fprintf(stderr,SARDATA__WCS_ALOSE,SARDATA_RVL_ALOSE(sdr));

	return(nitems);
}
/***************************************************************************/
int assign_sardata_params_ALOSE(struct PRM *prm, int line_prefix_size, int *line_suffix_size, int *record_length0)
{
double get_clock();

	prm->prf = sdr.PRF;
	prm->pulsedur = (1e-9)*sdr.chirp_length;

	*record_length0 = sdr.record_length - line_prefix_size;

        if (verbose) printf( "sdr.record_length = %d \n", sdr.record_length ); /* bytof */
        if (verbose) printf( "line_prefix_size = %d \n", line_prefix_size ); /* bytof */
        if (verbose) printf( "sdr.record_length = %d \n", sdr.record_length ); /* bytof */
	if (verbose) printf( "sdr.transmit_polarization = %d \n",sdr.transmit_polarization);
	if (verbose) printf( "sdr.receive_polarization = %d \n",sdr.receive_polarization);

	prm->SC_clock_start =  get_clock_ALOSE(sdr, tbias);

/* restec format changes - bytof */

	/* record_length is 21100 */
	/* beginning of line has a 292 byte prefix */
	/* end of line has a 80 byte (40 pixels) suffix (right-fill pixels)*/
	/* record_length0 (data length) is (20688 - 412) = 20276 */
	/* n_data_pixels  10304 */
	/* 2 bytes per pixel */
	/* 412 bytes + (2*10304) bytes + (40*2) bytes  = 21100 bytes*/

	prm->good_bytes = 2*sdr.n_data_pixels + line_prefix_size;
	prm->num_rng_bins = sdr.n_data_pixels + prm->chirp_ext;		/* chirp_ext formerly nextend */

	prm->bytes_per_line = sdr.record_length;
	if(sdr.receive_polarization == 2) prm->bytes_per_line = line_prefix_size + (sdr.record_length - line_prefix_size)/2;
	
	*line_suffix_size = prm->bytes_per_line - prm->good_bytes;

	if (prm->near_range < 0) prm->near_range = sdr.slant_range; 

        if(debug) printf( "assign_sardata_params: \n" );  /* bytof */
        if(debug) print_params( prm );   /* bytof */

	if (*record_length0 > 50000) {
		fprintf(stderr, "**** record_length is %d !\n", *record_length0);
		die("expect something like 21100 .... try -swap option?\n","exiting");
		}

	return(EXIT_SUCCESS);
}
/***************************************************************************/
int handle_prf_change_ALOSE(struct PRM *prm, FILE *imagefile, long *byte_offset, int n) 
{
	//prm->num_lines = n;

	fseek(imagefile, -1*sizeof(struct sardata_info_ALOSE), SEEK_CUR);

	*byte_offset = ftell(imagefile);

	printf(" *** PRF changed from %lf to  %lf  at line %d (byte %ld)\n", (0.001*prm->prf),(0.001*sdr.PRF), n-1, *byte_offset);
    //    printf(" end: PRF changed from %lf to  %lf  at line %d \n", (0.001*prm->prf),(0.001*sdr.PRF), n);

	return(EXIT_SUCCESS);
}
/***************************************************************************/
