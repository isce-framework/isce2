/********************************************************************************
 * Program tp prepare ALOS L1.0 data for InSAR processing.  REad in data and    *
 * leader file and write out prm and raw file for siosar                        *
********************************************************************************
 * Creator:  Rob Mellors and David T. Sandwell                                  *
 *           (San Diego State University, Scripps Institution of Oceanography)  *
 * Date   :  10/03/2007                                                         *
 ********************************************************************************/
/********************************************************************************
 * Modification history:                                                        *
 * Date:                                                                        *
 * 4/23/07 added code to check endianess RJM                                    *
 * 07/15/08   added a command-line option to force earth radius DTS             *
 * 04/26/10   added a command-line option to force the number of patches       *
 * merged in ALOSE code by Jeff B and David S					*
 * - added options -ALOSE -ALOS
 * added write_roi
 * *****************************************************************************/


#include "alosglobals.h"
#include"image_sio.h"
#include"lib_functions.h"
#include "resamp.h"

//ALOS I or Q mean = 15.5, so get 15 or 16 randomly here
//#define ZERO_VALUE (char)(15 + rand() % 2)
//I changed the dynamic range when reading data
//ALOS I or Q mean = 63.5, so get 63 or 64 randomly here
#define ZERO_VALUE (char)(63 + rand() % 2)


char    *USAGE = "\n\nUsage: ALOS_pre_process imagefile LEDfile [-near near_range] [-radius RE] [-swap] [-V] [-debug] [-quiet] \n"
"\ncreates data.raw and writes out parameters (PRM format) to stdout\n"
"\nimagefile 	ALOS Level 1.0 complex file (CEOS format):\n"
"LEDfile 	ALOS Level 1.0 LED file (CEOS leaderfile format):\n"
"\n options: \n"
"-near near_range  	specify the near_range (m) \n"
"-radius RE 		specify the local earth radius (m) \n"
"-swap 			do byte-swap (should be automatic) \n"
"-nodopp 		does not calculate doppler (sets fd1 to zero!) \n"
"-npatch                set the number of patches \n"
"-fd1 [DOPP] 		sets doppler centroid [fd1] to DOPP\n"
"-quad                  adjust parameters for quad pol mod (PRF/2)\n"
"-ALOSE                 use ERSDAC format \n"
"-ALOS                  use AUIG format (default) \n"
"-roi                   write roi_pac format output\n"
"-V 			verbose write information) \n"
"-debug                 write even more information \n"
"-quiet                 don't write any information \n"
"-force_slope chirp_slope       force a value for the chirp slope\n"
"-chirp_ext chirp_ext           force a value for the chirp extension (integer)\n"
"-tbias     tbias               correct the clock bias (positive value means plus)\n"
"Example:\n"
"ALOS_pre_process  IMG-HH-ALPSRP050420840-H1.0__A LED-ALPSRP050420840-H1.0__A \n";
long read_ALOS_data (FILE *, FILE *, struct PRM *, long *, struct resamp_info *, int);
long read_ALOSE_data (FILE *, FILE *, struct PRM *, long *, struct resamp_info *, int);
void parse_ALOS_commands(int, char **, char *, struct PRM *);
void set_ALOS_defaults(struct PRM *);
void print_ALOS_defaults(struct PRM *);
void swap_ALOS_data_info(struct sardata_info *);
void get_files(struct PRM *, FILE **, FILE **, char *, char *, int);
// roi_pac stuff
int write_roi_orbit(struct ALOS_ORB, char *);
int write_roi(char *, FILE *, struct PRM, struct ALOS_ORB, char *);
// ISCE stuff
void init_from_PRM(struct PRM inPRM, struct PRM *prm);

int resamp_azimuth(char *slc2, char *rslc2, int nrg, int naz1, int naz2, double prf, double *dopcoeff, double *azcoef, int n, double beta);

int
ALOS_pre_process(struct PRM inputPRM, struct PRM *outputPRM,struct GLOBALS globals, int image_i) //image number starts with 0!!!
{
FILE	*imagefile, *ldrfile;
FILE	*rawfile[11], *prmfile[11];
char	prmfilename[128];
int	nPRF;
long	byte_offset;
struct 	PRM prm;
struct 	ALOS_ORB orb;
char   	date[8];


//////////////////////////////////////////////
    FILE	*resampinfofile;
    struct 	resamp_info rspi;
    struct 	resamp_info rspi_new;
    struct 	resamp_info rspi_pre[100];//maximum number of frames: 100
    int i, j, k;
    double SC_clock_start;
    double SC_clock_start_resamp;
    double d2s = 24.0 * 3600.0;
    double line_number_first;
    int num_lines_out;
    int gap_flag;

    double prf_all[200];//maximum number of prfs: 200
    int frame_counter_start_all[200];//maximum number of prfs: 200
    int nPRF_all;//maximum number of prfs: 200

    double dopcoeff[4];
    double azcoef[2];
    int num_lines_max, j_max;
    char outputfile[256];
    char *data;
    FILE *first_prf_fp;
    FILE *next_prf_fp;
    int num_lines_append;
    //int num_lines_gap;
    int ret;
//////////////////////////////////////////////



	//if (argc < 3) die (USAGE,"");
    printf("reading image: %d\n", image_i);

	/* set flags  */
	dopp = globals.dopp;
	roi = 0;
	quad_pol = globals.quad_pol;
	debug = 0;
	verbose = 0;
	swap = 0;
	quiet_flag = 0;

	nPRF = 0;
	ALOS_format = globals.ALOS_format;

	null_sio_struct(&prm);
	set_ALOS_defaults(&prm);

	/* read command line */
	init_from_PRM(inputPRM,&prm);
	//parse_ALOS_commands(argc, argv, USAGE, &prm);

	/* apply an additional timing bias based on corner reflector analysis */
	//tbias = tbias - 0.0020835;

	if (verbose) print_ALOS_defaults(&prm);
	if (is_big_endian_() == -1) {swap = 1;fprintf(stderr,".... swapping bytes\n");} else {swap = 0;} 

	/* IMG and LED files should exist already */
	if ((imagefile = fopen(globals.imagefilename, "r")) == NULL) die ("couldn't open Level 1.0 IMG file \n",globals.imagefilename);
	if ((ldrfile = fopen(inputPRM.led_file, "r")) == NULL) die ("couldn't open LED file \n",inputPRM.led_file);

	/* if it exists, copy to prm structure */
	strcpy(prm.led_file,inputPRM.led_file);

	/* name and open output files and header files for raw data (but input for later processing) */
	get_files(&prm, &rawfile[nPRF], &prmfile[nPRF], prmfilename, prm.input_file, nPRF);

	/* read sarleader; put info into prm; write log file if specified 		*/
	read_ALOS_sarleader(ldrfile, &prm, &orb);
	
	/* infer type of data from ldrfile						*/
	if ((SAR_mode == 2) && (quad_pol == 0)) {
		fprintf(stderr," SAR_mode = %d ; assuming quad_pol\n", SAR_mode);
		quad_pol = 1;
		}

	/* read Level 1.0 file;  put info into prm; convert to *.raw format 		*/
	/* if PRF changes halfway through, create new set of header and data files      */
	/* byte_offset is non-zero only if the prf changes				*/
	/* byte_offset gets set to point in file at prf change				*/

	byte_offset = -1;
	while (byte_offset != 0){

		/* if prf changes, create new prm and data files			*/
		if (nPRF > 0 ) {
			if (verbose) fprintf(stderr,"creating multiple files due to PRF change (*.%d) \n",nPRF+1);
			get_files(&prm, &rawfile[nPRF], &prmfile[nPRF], prmfilename, prm.input_file, nPRF);
			}

		/* set the chirp extension to 500 if FBD fs = 16000000 */
        	if (prm.fs < 17000000.) {
			prm.chirp_ext = 500;
			prm.chirp_slope =  -5.18519e+11;
		} else {
			prm.chirp_slope = -1.03704e+12;
		}
		if (ALOS_format == 1) prm.first_sample = 146;

		/* read_ALOS_data returns 0 if all data file is read;
		returns byte offset if the PRF changes  */
		/* calculate parameters from orbit */
		if (ALOS_format == 0) {
			byte_offset = read_ALOS_data(imagefile, rawfile[nPRF], &prm, &byte_offset, &rspi, nPRF);
			}

		/* ERSDAC  - use read_ALOSE_data */
		if (ALOS_format == 1) {
			byte_offset = read_ALOSE_data(imagefile, rawfile[nPRF], &prm, &byte_offset, &rspi, nPRF);
			}

		// should work for AUIG and ERSDAC
		ALOS_ldr_orbit(&orb, &prm);

		/* calculate doppler from raw file */
		dopp=1;//always compute doppler for doing prf resampling
		if (dopp == 1) calc_dop(&prm);
		//prf as a function of range in Hz
		rspi.fd1[nPRF] = prm.fd1;
		rspi.fdd1[nPRF] = prm.fdd1;
		rspi.fddd1[nPRF] = prm.fddd1;
		//rspi.input_file[nPRF] = prm.input_file;
		strcpy(rspi.input_file[nPRF], prm.input_file);

		/* divide prf in half for quad_pol 	*/
		/* fix chirp slope			*/
		if (quad_pol) {
			prm.prf = 0.5 *  prm.prf; 
			prm.chirp_slope = -871580000000.0;
			prm.chirp_ext = 500.0;
			fprintf(stderr," quad pol: fixing prf %f\n", prm.prf);
			fprintf(stderr," quad pol: fixing chirp_slope %g\n", prm.chirp_slope);
			fprintf(stderr," quad pol: fixing chirp_ext %d\n", prm.chirp_ext);
			}

		/* force chirp slope if asked to */
		if (force_slope == 1) prm.chirp_slope = forced_slope;

		/* write ascii output, SIO format */
		put_sio_struct(prm, prmfile[nPRF]);

		/* write roi_pac output */
		if (roi) {
			// first part of rsc file
			//write_roi(argv[1], ldrfile, prm, orb, date);
			// orbit file 
			//write_roi_orbit(orb, date);
			}

		nPRF++;
		}
        rspi.nPRF=nPRF;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    printf("\nPRF details of frame: %d\n", image_i);
    printf("+++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("number of PRF: %d\n", rspi.nPRF);
    for (i = 0; i < rspi.nPRF; i++){
		printf("PRF %d prf (Hz): %f\n", i+1, rspi.prf[i]);
		printf("PRF %d start time (days): %20.12f\n", i+1, rspi.SC_clock_start[i]);
	    printf("PRF %d frame_counter_start: %d\n", i+1, rspi.frame_counter_start[i]);
	    printf("PRF %d frame_counter_end: %d\n", i+1, rspi.frame_counter_end[i]);
	    printf("PRF %d number of lines: %d\n\n", i+1, rspi.frame_counter_end[i]-rspi.frame_counter_start[i]+1);
    }

    //open parameter file for doing time adjustment and interpolation
    if (image_i == 0){
        if((resampinfofile = fopen("resampinfo.bin", "wb")) == NULL)
        	die("couldn't open resampinfo file","resampinfo.bin");
    }
    else{
    	//open the file for reading and appending
	    if((resampinfofile = fopen("resampinfo.bin", "ab+")) == NULL)
	    	die("couldn't open resampinfo file","resampinfo.bin");
	    rewind(resampinfofile);
	    for(i=0; i < image_i; i++){
			if((fread((void *) &rspi_pre[i],sizeof(struct resamp_info), 1, resampinfofile)) != 1)
				die("couldn't read from file","resampinfo.bin");
		}
    }

    //get parameter from this image
    memcpy(&rspi_pre[image_i], &rspi, sizeof(struct resamp_info));

    //initialize rspi_new with resamp_info from reading the image, put the adjusted time in it
    memcpy(&rspi_new, &rspi, sizeof(struct resamp_info));

    //adjust start time
    //unified PRF of the full track: first prf of first image
    //start time of the full track: first line of first image
    //only adjust time when the format is not ERSDAC format, becasue ERSDAC format does not have sdr.frame_counter.
    printf("adjust start times\n");
    if(ALOS_format == 0){

    if(image_i==0){
		//adjust start time of prf file i, no need to adjust for first prf
		for(i = 1; i < rspi_pre[0].nPRF; i++){
			//time of the line just before the first line of first prf file
			SC_clock_start = rspi_pre[0].SC_clock_start[0] - (1.0/rspi_pre[0].prf[0]) / d2s;
			//time of the last line of each prf file
			for(j = 0; j < i; j++){
				if(rspi_pre[0].num_lines[j] != rspi_pre[0].frame_counter_end[j] - rspi_pre[0].frame_counter_start[j] + 1)
					fprintf(stderr, "\n\nWARNING: in image %d prf file %d, \
						number of lines in file: %d is not equal to that computed from frame_counter: %d\n\n", \
						0, j, rspi_pre[0].num_lines[j], rspi_pre[0].frame_counter_end[j] - rspi_pre[0].frame_counter_start[j] + 1);
			    SC_clock_start += (rspi_pre[0].frame_counter_end[j]-rspi_pre[0].frame_counter_start[j]+1) * (1.0/rspi_pre[0].prf[j]) / d2s;
			}
            //time of the first line of current prf file
            SC_clock_start += (1.0/rspi_pre[0].prf[i]) / d2s;

            printf("time adjustment result for image %d, prf %d:\n", image_i,  i);
            printf("+++++++++++++++++++++++++++++++++++++++++++++++\n");
			printf("original start time: %20.12f\n", rspi_pre[0].SC_clock_start[i]);
			printf("adjusted start time: %20.12f\n", SC_clock_start);
			printf("original - adjusted: %f (number of PRI)\n\n", (rspi_pre[0].SC_clock_start[i]-SC_clock_start)*d2s/(1.0/rspi_pre[0].prf[i]));
            //update
            rspi_new.SC_clock_start[i] = SC_clock_start;
		}
    }
    else{
        //1. check to see if there is gap between images
        gap_flag = 0;
        for(i = 0; i < image_i; i++){
			if (rspi_pre[i].frame_counter_end[rspi_pre[i].nPRF-1] - rspi_pre[i+1].frame_counter_start[0] <= -2){
				fprintf(stderr, "\n\nWARNING: there are gaps between image %d and image: %d\n", i, i+1);
				fprintf(stderr, "since we don't know the prf of these gap lines, we are not able to adjust starting time\n\n");
				gap_flag = 1;
			}
        }
        //2. adjust start time
        if(gap_flag == 0){
        	//2.1 count the number of prf chunks in the full track including this image
        	nPRF_all = 0;
	        for(i = 0; i < image_i+1; i++){
				for(j = 0; j < rspi_pre[i].nPRF; j++){
					if((i==0) && (j==0)){
						prf_all[nPRF_all] = rspi_pre[i].prf[j];
						frame_counter_start_all[nPRF_all] = rspi_pre[i].frame_counter_start[j];
						nPRF_all += 1;
					}
					else{
						if((rspi_pre[i].frame_counter_start[j]>frame_counter_start_all[nPRF_all-1]) && (rspi_pre[i].prf[j]!=prf_all[nPRF_all-1])){
							prf_all[nPRF_all] = rspi_pre[i].prf[j];
							frame_counter_start_all[nPRF_all] = rspi_pre[i].frame_counter_start[j];
							nPRF_all += 1;	
						}
					}
				}
	        }
            printf("number of prfs including this image: %d\n", nPRF_all);
            printf("list of prfs:\n");
            for(i = 0; i < nPRF_all; i++){
            	printf("frame_counter: %d, prf: %f\n", frame_counter_start_all[i], prf_all[i]);
            }

            //2.2 adjust start time
            for(i = 0; i < rspi_pre[image_i].nPRF; i++){
    			//time of the line just before the first line of first prf file
    			//because the unite is day, the errors caused can be 0.042529743164777756 lines, should remove the integer or year part of SC_clock_start, or
    			//use second as unit in the future
    			SC_clock_start = rspi_pre[0].SC_clock_start[0] - (1.0/rspi_pre[0].prf[0]) / d2s;
                //if there is only one PRF (no prf changes across all images)
                if(nPRF_all == 1){
                	SC_clock_start += (rspi_pre[image_i].frame_counter_start[0] - rspi_pre[0].frame_counter_start[0] + 1) * (1.0/rspi_pre[0].prf[0]) / d2s;
                }
                else{
                	//find its position among the prfs, start from the second prf
	            	for(j = 1; j < nPRF_all; j++){
	            		if(rspi_pre[image_i].frame_counter_start[i] < frame_counter_start_all[j]){
	                        //time of the last line of each prf chuck
	            			for(k = 1; k < j; k++)
	            				SC_clock_start += (frame_counter_start_all[k]-frame_counter_start_all[k-1]) * (1.0/prf_all[k-1]) / d2s;
	            			SC_clock_start += (rspi_pre[image_i].frame_counter_start[i] - frame_counter_start_all[j-1] + 1) * (1.0/prf_all[j-1]) / d2s;
	            			break;
	            		}
	            		else if(rspi_pre[image_i].frame_counter_start[i] == frame_counter_start_all[j]){
	                        //time of the last line of each prf chuck
	            			for(k = 1; k < j; k++)
	            				SC_clock_start += (frame_counter_start_all[k]-frame_counter_start_all[k-1]) * (1.0/prf_all[k-1]) / d2s;
	            			SC_clock_start += (rspi_pre[image_i].frame_counter_start[i] - frame_counter_start_all[j-1] + 1) * (1.0/prf_all[j-1]) / d2s;
	                        //extra pri of j-1 above, so remove it and add the pri of j
	                        SC_clock_start += (1.0/prf_all[j]) / d2s - (1.0/prf_all[j-1]) / d2s;
	            			break;
	            		}
	            		else{
	            			if(j == nPRF_all - 1){
		            			for(k = 1; k < j+1; k++)
		            				SC_clock_start += (frame_counter_start_all[k]-frame_counter_start_all[k-1]) * (1.0/prf_all[k-1]) / d2s;
		            			SC_clock_start += (rspi_pre[image_i].frame_counter_start[i] - frame_counter_start_all[j] + 1) * (1.0/prf_all[j]) / d2s;
	            				break;
	            			}
	            			else{
	            				continue;
	            			}
	            		}
	            	}
                }

                //time of the first line of current prf file
                printf("time adjustment result for image %d, prf %d:\n", image_i,  i);
                printf("+++++++++++++++++++++++++++++++++++++++++++++++\n");
    			printf("original start time: %20.12f\n", rspi_pre[image_i].SC_clock_start[i]);
    			printf("adjusted start time: %20.12f\n", SC_clock_start);
    			printf("original - adjusted: %f (number of PRI)\n\n", (rspi_pre[image_i].SC_clock_start[i]-SC_clock_start)*d2s/(1.0/rspi_pre[image_i].prf[i]));

	            //update
	            rspi_new.SC_clock_start[i] = SC_clock_start;
            }
        }
    }

    }


    // use parameters from rspi_pre[image_i], instead of rspi_new (to be updated)
    //except rspi_new.SC_clock_start[i], since it was updated (more accurate) above.
    printf("azimuth resampling\n");
    for(i = 0; i < rspi_pre[image_i].nPRF; i++){
    	if((image_i==0)&&(i==0))
    		continue;
	    //convention: line numbers start with zero
	    //line number of first line of first prf of first image: 0
	    //line number of first line of this prf file
	    line_number_first = (rspi_new.SC_clock_start[i] - rspi_pre[0].SC_clock_start[0]) * d2s / (1.0 / rspi_pre[0].prf[0]);
	    //unit: pri of first prf of first image
	    num_lines_out = (int)((rspi_pre[image_i].frame_counter_end[i] - rspi_pre[image_i].frame_counter_start[i] + 1) * (1.0/rspi_pre[image_i].prf[i]) / (1.0/rspi_pre[0].prf[0]));

        if((fabs(roundfi(line_number_first)-line_number_first)<0.1) && (rspi_pre[image_i].prf[i]==rspi_pre[0].prf[0]))
        	continue;

        //time of first line of the resampled image
        SC_clock_start_resamp = rspi_pre[0].SC_clock_start[0] + roundfi(line_number_first) * (1.0 / rspi_pre[0].prf[0]) / d2s;
        //compute offset parameters
        //azcoef[0] + azpos * azcoef[1]
        azcoef[0] = (SC_clock_start_resamp - rspi_new.SC_clock_start[i]) * d2s / (1.0/rspi_pre[image_i].prf[i]);
        azcoef[1] = (1.0/rspi_pre[0].prf[0]) / (1.0/rspi_pre[image_i].prf[i]) - 1.0;
        
		//use doppler centroid frequency estimated from prf with maximum number of lines in this image
		num_lines_max = -1;
		j_max = -1;
        for(j = 0; j < rspi_pre[image_i].nPRF; j++){
        	if(rspi_pre[image_i].num_lines[j] >= num_lines_max){
        		num_lines_max = rspi_pre[image_i].num_lines[j];
        		j_max = j;
        	}
        }
        dopcoeff[0] = rspi_pre[image_i].fd1[j_max]; //average prf for alos-1 is good enough (calc_dop.c).
        dopcoeff[1] = 0.0;
        dopcoeff[2] = 0.0;
        dopcoeff[3] = 0.0;

        //The filenames of all three files created for each prf, are from prm.input_file
        //PRM:                   prm.input_file.PRM + (.prfno_start_from_1, if not first prf)
        //data:                  prm.input_file     + (.prfno_start_from_1, if not first prf)
        //data after resampling: prm.input_file     + (.prfno_start_from_1, if not first prf) + .interp

        sprintf(outputfile,"%s.interp", rspi_pre[image_i].input_file[i]);
        //start interpolation
        resamp_azimuth(rspi_pre[image_i].input_file[i], outputfile, rspi_pre[image_i].num_bins[i], num_lines_out, rspi_pre[image_i].num_lines[i], rspi_pre[image_i].prf[i], dopcoeff, azcoef, 9, 5.0);

        //update parameters
        rspi_new.SC_clock_start[i] = SC_clock_start_resamp;
        rspi_new.num_lines[i] = num_lines_out;
        rspi_new.prf[i] = rspi_pre[0].prf[0];
        rspi_new.fd1[i] = dopcoeff[0];
        rspi_new.fdd1[i]= dopcoeff[1];
        rspi_new.fddd1[i]=dopcoeff[2];
        strcpy(rspi_new.input_file[i], outputfile);
    }


    //concatenate prfs: put all prfs to the first prf
    // use parameters from rspi_new (updated), instead of rspi_pre[image_i]
    if(rspi_new.nPRF > 1){

        //prepare for appending subsequent prfs to first prf: open files and allocate memory
        if((first_prf_fp = fopen(rspi_new.input_file[0], "ab")) == NULL)
        	die("can't open", rspi_new.input_file[0]);
        //number of range samples in each prf is asummed to be same
		if((data = (char *)malloc(2*sizeof(char)*rspi_new.num_bins[0])) == NULL)
			die("can't allocate memory for data", "");

	    //append prf i
	    for(i = 1; i < rspi_new.nPRF; i++){
            //number of lines to be appended between frames if there are gaps
            num_lines_append = roundfi((rspi_new.SC_clock_start[i] - rspi_new.SC_clock_start[0]) * d2s / (1.0/rspi_pre[0].prf[0])) - rspi_new.num_lines[0];
            if(num_lines_append >= 1){
            	for(j = 0; j < num_lines_append; j++){
            		for(k = 0; k < 2*rspi_new.num_bins[i]; k++)
            			data[k] = ZERO_VALUE;
			        if(fwrite((char *)data, 2*sizeof(char)*rspi_new.num_bins[i], 1, first_prf_fp) != 1)
			        	die("can't write data to", rspi_new.input_file[0]);
            	}
	            rspi_new.num_lines[0] += num_lines_append;
            }

            //append data from rspi_new.input_file[i]
	        if((next_prf_fp = fopen(rspi_new.input_file[i], "rb")) == NULL)
	        	die("can't open", rspi_new.input_file[i]);
            num_lines_append = 0;
            for(j = 0; j < rspi_new.num_lines[i]; j++){
                if(roundfi((rspi_new.SC_clock_start[i] + j * (1.0/rspi_pre[0].prf[0]) / d2s -  rspi_new.SC_clock_start[0]) * d2s / (1.0/rspi_pre[0].prf[0])) >= rspi_new.num_lines[0]){
			        if(fread((char *)data, 2*sizeof(char)*rspi_new.num_bins[i], 1, next_prf_fp) != 1)
			        	die("can't read data from", rspi_new.input_file[i]);
			        if(fwrite((char *)data, 2*sizeof(char)*rspi_new.num_bins[i], 1, first_prf_fp) != 1)
			        	die("can't write data to", rspi_new.input_file[0]);
                    num_lines_append += 1;
                }
                else{
                	fseek(next_prf_fp, 2*sizeof(char)*rspi_new.num_bins[i], SEEK_CUR);
                }
            }
            rspi_new.num_lines[0] += num_lines_append;
            fclose(next_prf_fp);
	    }
	    free(data);
	    fclose(first_prf_fp);
    }


    //tidy up intermediate files
    for(i = 0; i < rspi_pre[image_i].nPRF; i++){
    	//if Return value = 0 then it indicates str1 is equal to str2.
    	ret = strcmp(rspi_new.input_file[i], rspi_pre[image_i].input_file[i]);
    	if(i == 0){
    		if(ret != 0){
               //remove original
  			   if(remove(rspi_pre[image_i].input_file[i]) != 0)
				   die("can't delete file", rspi_pre[image_i].input_file[i]);
               //keep resampled and appended
			   if(rename(rspi_new.input_file[i], rspi_pre[image_i].input_file[i]) != 0)
				   die("can't rename file", rspi_new.input_file[i]);
    		}
    	}
        else{
        	//remove original
        	if(remove(rspi_pre[image_i].input_file[i]) != 0)
        		die("can't delete file", rspi_pre[image_i].input_file[i]);
            //remove resampled
            if(ret != 0){
	        	if(remove(rspi_new.input_file[i]) != 0)
	        		die("can't delete file", rspi_new.input_file[i]);
            }
        }
    }


    //update prm
    prm.prf = rspi_new.prf[0];
    prm.num_lines = rspi_new.num_lines[0];
    prm.SC_clock_start = rspi_new.SC_clock_start[0];
    prm.SC_clock_stop = prm.SC_clock_start + (prm.num_lines - 1) * (1.0/prm.prf) / d2s;
	prm.fd1 = rspi_pre[image_i].fd1[j_max]; //average prf for alos-1 is good enough (calc_dop.c).
	prm.fdd1 = 0.0;
	prm.fddd1 =0.0;

	prm.xmi = 63.5;
	prm.xmq = 63.5;

	//write to resampinfo.bin
	if((fwrite((void *)&rspi_pre[image_i], sizeof(struct resamp_info), 1, resampinfofile)) != 1 )
		die("couldn't write to file", "resampinfo.bin");
    fclose(resampinfofile);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	if (orb.points != NULL)
	{
		free(orb.points);
	}
	*outputPRM = prm;
	return(EXIT_SUCCESS);
}
/*------------------------------------------------------*/
void get_files(struct PRM *prm, FILE **rawfile, FILE **prmfile, char *prmfilename, char *name, int n)
{
	/* name and open output file for raw data (but input for later processing)      */
	/* if more than 1 set of output files, append an integer (beginning with 2)     */

	//if (n == 0) {
	//	sprintf(prm->input_file,"%s.raw", name);
	//	sprintf(prmfilename,"%s.PRM", name);
	//} else {
	//	sprintf(prm->input_file,"%s.raw.%d",name,n+1);
	//	sprintf(prmfilename,"%s.PRM.%d", name, n+1);
	//}
	if (n==0) {
		sprintf(prmfilename,"%s.PRM", name);
		sprintf(prm->input_file,"%s",name);
	} else {
		sprintf(prmfilename,"%s.PRM.%d", name, n+1);
		sprintf(prm->input_file,"%s.%d",name,n+1);
	}

	/* now open the files */
	if ((*rawfile = fopen(prm->input_file,"w")) == NULL) die("can't open ",prm->input_file);

	if ((*prmfile = fopen(prmfilename, "w")) == NULL) die ("couldn't open output PRM file \n",prmfilename);

}

