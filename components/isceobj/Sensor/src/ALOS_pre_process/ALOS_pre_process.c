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

long read_ALOS_data (FILE *, FILE *, struct PRM *, long *);
long read_ALOSE_data (FILE *, FILE *, struct PRM *, long *);
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

int
ALOS_pre_process(struct PRM inputPRM, struct PRM *outputPRM,struct GLOBALS globals)
{
FILE	*imagefile, *ldrfile;
FILE	*rawfile[11];//*prmfile[11];
//char	prmfilename[128];
int	nPRF;
long	byte_offset;
struct 	PRM prm;
struct 	ALOS_ORB orb;
char   	date[8];

	//if (argc < 3) die (USAGE,"");

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

	//if (verbose) print_ALOS_defaults(&prm);
	if (is_big_endian_() == -1) {swap = 1;fprintf(stderr,".... swapping bytes\n");} else {swap = 0;} 

	/* IMG and LED files should exist already */
	if ((rawfile[0] = fopen(prm.input_file,"w")) == NULL) die("can't open ",prm.input_file);
	if ((imagefile = fopen(globals.imagefilename, "r")) == NULL) die ("couldn't open Level 1.0 IMG file \n",globals.imagefilename);
	if ((ldrfile = fopen(inputPRM.led_file, "r")) == NULL) die ("couldn't open LED file \n",inputPRM.led_file);

	/* if it exists, copy to prm structure */
	//strcpy(prm.led_file,leaderFilename);

	/* name and open output files and header files for raw data (but input for later processing) */
	//get_files(&prm, &rawfile[nPRF], &prmfile[nPRF], prmfilename, argv[1], nPRF);

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
			//get_files(&prm, &rawfile[nPRF], &prmfile[nPRF], prmfilename, argv[1], nPRF);
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
			byte_offset = read_ALOS_data(imagefile, rawfile[nPRF], &prm, &byte_offset);
			}

		/* ERSDAC  - use read_ALOSE_data */
		if (ALOS_format == 1) {
			byte_offset = read_ALOSE_data(imagefile, rawfile[nPRF], &prm, &byte_offset);
			}

		//fclose(rawfile[nPRF]);

		// should work for AUIG and ERSDAC
		ALOS_ldr_orbit(&orb, &prm);

		/* calculate doppler from raw file */
		if (dopp == 1) calc_dop(&prm);

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
		//put_sio_struct(prm, prmfile[nPRF]);

		/* write roi_pac output */
		if (roi) {
			// first part of rsc file
			//write_roi(argv[1], ldrfile, prm, orb, date);
			// orbit file 
			//write_roi_orbit(orb, date);
			}

		nPRF++;
		}

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

	if (n == 0) {
		sprintf(prm->input_file,"%s.raw", name);
		sprintf(prmfilename,"%s.PRM", name);
	} else {
		sprintf(prm->input_file,"%s.raw.%d",name,n+1);
		sprintf(prmfilename,"%s.PRM.%d", name, n+1);
	}

	/* now open the files */
	if ((*rawfile = fopen(prm->input_file,"w")) == NULL) die("can't open ",prm->input_file);

	if ((*prmfile = fopen(prmfilename, "w")) == NULL) die ("couldn't open output PRM file \n",prmfilename);

}
/*------------------------------------------------------*/
