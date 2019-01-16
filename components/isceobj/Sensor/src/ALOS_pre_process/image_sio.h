/* taken from soi.h */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#define SOL 299792458.0
#define PI 3.1415926535897932
#define PI2 6.2831853071795864
#define I2MAX 32767.0
#define I2SCALE 4.e6
#define TRUE 1
#define FALSE 0
#define RW 0666
#define MULT_FACT 1000.0
#define sgn(A) ((A) >= 0.0 ? 1.0 : -1.0)
#define clipi2(A) ( ((A) > I2MAX) ? I2MAX : (((A) < -I2MAX) ? -I2MAX : A) )
#define nint(x) (int)rint(x)
#define ERS1 1
#define ERS2 2
#define RSAT 3
#define ENVS 4
#define ALOS 5

#define EXIT_FLAG 1
#define paka(p) {perror((p)); exit(EXIT_FLAG);}
#define MALLOC(p,s) if (((p) = malloc(s)) == NULL) {paka("error: malloc()  ");}

#define NULL_DATA 15
#define NULL_INT -99999
#define NULL_DOUBLE -99999.9999
#define NULL_CHAR  "XXXXXXXX"

typedef struct SCOMPLEX {short r,i;} scomplex;
typedef struct FCOMPLEX {float r,i;} fcomplex;
typedef struct DCOMPLEX {double r,i;} dcomplex;

struct PRM {
	char input_file[256];
	char SLC_file[256];
	char out_amp_file[256];
	char out_data_file[256];
	char deskew[8];
	char iqflip[8];
	char offset_video[8];
	char srm[8];
	char ref_file[128];
	char led_file[128];
	char orbdir[8];	/* orbit direction A or D (ASCEND or DESCEND) - added by RJM*/
	char dtype[8];  /* SLC data type a-SCOMPLEX integer complex, c-FCOMPLEX float complex */
	char date[16];	/* yymmdd format - skip first two digits of year - added by RJM*/

	int debug_flag;
	int bytes_per_line;
	int good_bytes;
	int first_line;
	int num_patches;
	int first_sample;
	int num_valid_az;
	int st_rng_bin;
	int num_rng_bins;
	int chirp_ext;
	int nlooks;
	int rshift;
	int ashift;
	int fdc_ystrt;
	int fdc_strt;
	int rec_start;
	int rec_stop;
	int SC_identity;	/* (1)-ERS1 (2)-ERS2 (3)-Radarsat (4)-Envisat (5)-ALOS */
	int ref_identity;	/* (1)-ERS1 (2)-ERS2 (3)-Radarsat (4)-Envisat (5)-ALOS */
	int nrows;
	int num_lines;
	int SLC_format;		/* 1 => complex ints (2 bytes)	2 => complex floats (4 bytes) */

	double SC_clock_start;	/* YYDDD.DDDD */
	double SC_clock_stop;	/* YYDDD.DDDD */
	double icu_start;	/* onboard clock counter */
	double ref_clock_start;
	double ref_clock_stop;
	double caltone;
	double RE;			/*local earth eadius */
	double rc;			/* polar radius */
	double ra;			/* equatorial radius */
	double vel;			/* Equivalent SC velocity */
	double ht;			/* (SC_radius - RE) center */
	double ht_start;		/* (SC_radius - RE) start */
	double ht_end;			/* (SC_radius - RE) end */
	double near_range;
	double far_range;
	double prf;
	double xmi;
	double xmq;
	double az_res;
	double fs;
	double chirp_slope;
	double pulsedur;
	double lambda;
	double rhww;
	double pctbw;
	double pctbwaz;
	double fd1;
	double fdd1;
	double fddd1;
	double delr;			/* added RJM */
	double yaw;                     /* added RJM 12/07*/

	double sub_int_r;
	double sub_int_a;
	double sub_double;
	double stretch_r;
	double stretch_a;
	double a_stretch_r;
	double a_stretch_a;
	double baseline_start;
	double baseline_end;
	double alpha_start;
	double alpha_end;
	double bpara;			/* parallel baseline - added by RJM */
	double bperp;			/* perpendicular baseline - added by RJM */
};
/*
offset_video 		off_vid		
chirp_ext 		nextend
-------------------------------
scnd_rng_mig 		srm
Flip_iq 		iqflip
reference_file		ref_file
rng_spec_wgt 		rhww
rm_rng_band 		pctbw
rm_az_band 		pctbwaz
rng_samp_rate		fs
good_bytes_per_line 	good_bytes
earth_radius		RE
SC_vel			vel
SC_height		ht
SC_height_start		ht_start
SC_height_end		ht_end
PRF			prf
I_mean			xmi
Q_mean			xmq
pulse_dur		pulsedur
radar_wavelength	lambda
rng_spec_wgt		rhww

*/
int	verbose;	/* controls minimal level of output 	*/ 
int	debug; 		/* more output 				*/
int	roi; 		/* more output 				*/
int	swap; 		/* whether to swap bytes 		*/
int	quad_pol; 	/* quad polarization data 		*/
int	ALOS_format; 	/* AUIG:  ALOS_format = 0  		*/
			/* ERSDAC:  ALOS_format = 1  		*/
int	force_slope;	/* whether to set the slope	 	*/
int	dopp;		/* whether to calculate doppler 	*/
int	quiet_flag;	/* reduce output			*/
int	SAR_mode;	/* 0 => high-res 			*/	
			/* 1 => wide obs 			*/
			/* 2 => polarimetry 			*/
			/* from ALOS Product Format 3-2		*/
double	forced_slope;	/* value to set chirp_slope to          */
double  tbias;          /* time bias for bad orbit data         */
