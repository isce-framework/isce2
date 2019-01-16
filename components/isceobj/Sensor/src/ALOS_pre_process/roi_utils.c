// write_roi
// attempts to create a rsc file for roi_pac
// adapted from make_raw_alos.pl
// rjm - sdsu 7/2010
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"image_sio.h"
#include"lib_functions.h"
#define FACTOR 1000000

int prm2roi(struct PRM, double *, double *, int *, double *, double *, int *, int *, int *);
long get_file_size(FILE *);
int get_utc(double, struct SAR_info, double *, double *, double *, double, int);
int write_roi_orbit(struct ALOS_ORB, char *);

/* writes out rsc file for roi_pac */
int write_roi(char *imagery, FILE *ldrfile, struct PRM prm, struct ALOS_ORB orb, char *date)
{
int 	nitems, clength;
int 	xmin, xmax, ymin, ymax;
int 	file_length, width, first_sample;
int 	yr, yr2, mo, da, mn, hr, sc, ms;
long 	size;

double 	C = 299792458.0;
double 	ANTENNA_SIDE = -1;
double 	ANTENNA_LENGTH = 8.9;
double 	PLANET_GM = 3.98600448073E+14;
double 	PLANET_SPINRATE = 7.29211573052E-05;

double 	first_line_utc, last_line_utc, center_utc;
double 	ae, flat, pri, start_time;
double 	range_pixel_size, range_bias, starting_range, chirp;
double	ibias, qbias, wavelength, pulsedur, range_sample_freq, prf;

int	orbit_num, first_frame;
char  	syr2[2],smo[2],sda[2];
char   	proc_sys[64], proc_ver[64], ctime[64], foutname[128];
char 	polar[32], swath[16];

FILE 	*rsc, *datafile;

struct	SAR_info	sar;
struct	sarleader_binary	sb;

first_frame = 000;
/* assign variables from prm to roi		*/
/* these are from data file 			*/
prm2roi(prm, &start_time, &starting_range, &first_sample, &prf, &chirp, &width, &xmin, &xmax);

/* just define it rather than  read from dc_bias_i,q */
ibias  = qbias = 15.5;
ae = 6378137;
flat = 1.0/298.257223563;
clength = 0;
range_bias = 0.0;
pri = 1.0 / prf;

/* find size of raw input file  - this is a pain */
if ((datafile = fopen(imagery,"r")) == NULL) die("error opening ",imagery);

/* find data file size */
size = get_file_size(datafile);
ymin = 0;
ymax = file_length = size / width;

// allocate memory for structures 
sar.fixseg = (struct sarleader_fdr_fixseg *) malloc(sizeof(struct sarleader_fdr_fixseg));
sar.varseg = (struct sarleader_fdr_varseg *) malloc(sizeof(struct sarleader_fdr_varseg));
sar.dss_ALOS = (struct sarleader_dss_ALOS *) malloc(sizeof(struct sarleader_dss_ALOS));
sar.platform_ALOS = (struct platform_ALOS *) malloc(sizeof(struct platform_ALOS));

// read in sar leader (again)
// the first ones are not used (sarleader_binary, sarleader_binary)
// but read for completeness and move ahead into file
rewind(ldrfile);
nitems = fread(&sb, sizeof(struct sarleader_binary), 1, ldrfile);
fscanf(ldrfile, SARLEADER_FDR_FIXSEG_RCS, SARLEADER_FDR_FIXSEG_RVL(sar.fixseg));
fscanf(ldrfile, SARLEADER_FDR_VARSEG_RCS, SARLEADER_FDR_VARSEG_RVL(sar.varseg));
nitems = fread(&sb, sizeof(struct sarleader_binary), 1, ldrfile);
// this has the useful information
fscanf(ldrfile, SARLEADER_DSS_RCS_ALOS, SARLEADER_DSS_RVL_ALOS(sar.dss_ALOS));

// get some parameters from leaderfile
// not all these were read in for the PRM struct
// so need to read leaderfile again 
wavelength = atof(sar.dss_ALOS->radar_wavelength);
pulsedur = (atof(sar.dss_ALOS->range_pulse_length)/FACTOR);
range_sample_freq = (atof(sar.dss_ALOS->sampling_rate));
range_pixel_size = C / range_sample_freq / 2.0;

/* handling strings in C - happy, happy, joy, joy */
sscanf(sar.dss_ALOS->processing_system_id, " %s", &proc_sys[0]);
proc_sys[10] = '\0';
sscanf(sar.dss_ALOS->processing_version_id, " %s", &proc_ver[0]);
proc_ver[4] = '\0';
sscanf(sar.dss_ALOS->antenna_mech_bor, " %s", &swath[0]);
swath[4] = '\0';

sscanf(sar.dss_ALOS->orbit_number, " %d", &orbit_num);
strncpy(&polar[0], &sar.dss_ALOS->sensor_id_and_mode[16], 2);
polar[2] = '\0';

/* use time from leaderfile */
strncpy(&ctime[0], (sar.dss_ALOS->input_scene_center_time), 30);
ctime[30] = '\0';
sscanf(&ctime[0]," %2d%2d%2d%2d%2d%2d%2d%4d",&yr,&yr2,&mo,&da,&hr,&mn,&sc,&ms);
sscanf(&ctime[0]," %4d", &yr);
sscanf(&ctime[2]," %2s%2s%2s", &syr2[0], &smo[0], &sda[0]);
sprintf(&date[0],"%2s%2s%2s",syr2,smo,sda);

// utc time
get_utc(start_time, sar, &first_line_utc, &last_line_utc, &center_utc, pri, file_length);

// open output file
sprintf(foutname,"tmp.%s.raw.rsc",date);
if ((rsc = fopen(foutname,"w")) == NULL) die("error opening tmp_raw.rsc","");

fprintf(rsc,"FIRST_FRAME                              %d\n", first_frame);
fprintf(rsc,"FIRST_FRAME_SCENE_CENTER_TIME            %s\n", ctime);
fprintf(rsc,"FIRST_FRAME_SCENE_CENTER_LINE            %d\n", clength);
fprintf(rsc,"DATE                                     %s\n", date);
fprintf(rsc,"FIRST_LINE_YEAR                          %d\n", yr);
fprintf(rsc,"FIRST_LINE_MONTH_OF_YEAR                 %02d\n", mo);
fprintf(rsc,"FIRST_LINE_DAY_OF_MONTH                  %02d\n", da);
fprintf(rsc,"FIRST_CENTER_HOUR_OF_DAY                 %02d\n", hr);
fprintf(rsc,"FIRST_CENTER_MN_OF_HOUR                  %02d\n", mn);
fprintf(rsc,"FIRST_CENTER_S_OF_MN                     %02d\n", sc);
fprintf(rsc,"FIRST_CENTER_MS_OF_S                     %d\n", ms);
fprintf(rsc,"PROCESSING_SYSTEM                        %s\n", proc_sys);
fprintf(rsc,"PROCESSING_VERSION                       %s\n", proc_ver);
fprintf(rsc,"WAVELENGTH                               %f\n", wavelength);		
fprintf(rsc,"PULSE_LENGTH                             %g\n", pulsedur);		
fprintf(rsc,"CHIRP_SLOPE                              %g\n", chirp);		
fprintf(rsc,"I_BIAS                                   %4.1lf\n", ibias);			
fprintf(rsc,"Q_BIAS                                   %4.1lf\n", qbias);			
fprintf(rsc,"PLATFORM                                 ALOS\n");
fprintf(rsc,"BEAM                                     %s\n", swath);
fprintf(rsc,"POLARIZATION                             %s\n", polar);
fprintf(rsc,"ORBIT_NUMBER                             %d\n", orbit_num);
fprintf(rsc,"RANGE_BIAS                               %lf\n", range_bias);
fprintf(rsc,"STARTING_RANGE                           %-20.0lf\n", starting_range);	
fprintf(rsc,"RANGE_PIXEL_SIZE                         %-15.10lf\n", range_pixel_size);
fprintf(rsc,"PRF                                      %lf\n", prf);			
fprintf(rsc,"ANTENNA_SIDE                             %lf \n", ANTENNA_SIDE);
fprintf(rsc,"ANTENNA_LENGTH                           %3.1lf \n", ANTENNA_LENGTH);
fprintf(rsc,"FILE_LENGTH                              %d\n", file_length);
fprintf(rsc,"XMIN                                     %d\n", xmin);
fprintf(rsc,"XMAX                                     %d\n", xmax);
fprintf(rsc,"WIDTH                                    %d\n", width);
fprintf(rsc,"YMIN                                     0\n");
fprintf(rsc,"YMAX                                     %d\n", ymax);
fprintf(rsc,"RANGE_SAMPLING_FREQUENCY                 %-20.0lf\n", range_sample_freq);
fprintf(rsc,"PLANET_GM                                %-20.0lf\n", PLANET_GM);
fprintf(rsc,"PLANET_SPINRATE                          %-15.11e\n", PLANET_SPINRATE);
fprintf(rsc,"FIRST_LINE_UTC                           %lf\n", first_line_utc);
fprintf(rsc,"CENTER_LINE_UTC                          %lf\n", center_utc);
fprintf(rsc,"LAST_LINE_UTC                            %lf\n", last_line_utc);

fprintf(rsc,"EQUATORIAL_RADIUS                        %f\n", prm.RE);		// equatorial radius

//HEIGHT_TOP
//HEIGHT
//HEIGHT_DT
//VELOCITY
//LATITUDE
//LONGITUDE
//HEADING
//EQUATORIAL_RADIUS
//ECCENTRICITY_SQUARED
//EARTH_EAST_RADIUS
//EARTH_NORTH_RADIUS
//EARTH_RADIUS
//ORBIT_DIRECTION

/*
fprintf(rsc," %d\n", prm.num_lines);			// length
fprintf(rsc," %f\n", prm.SC_clock_start);		// start_time
fprintf(rsc," %s\n", prm.orbdir);			// orbdir
fprintf(rsc," %f\n", prm.ht);				// height
fprintf(rsc," %f\n", prm.vel);				// vel
fprintf(rsc," %f\n", prm.fd1);				// fd1
*/

fclose(rsc);

return(EXIT_SUCCESS);
}
/*--------------------------------------------------------------------------------------------------------------*/
int prm2roi(struct PRM prm, double *start_time, double *starting_range, int *first_sample, double *prf, double *chirp, int *width, int *xmin, int *xmax)
{
	*prf = prm.prf;
	*start_time = prm.SC_clock_start;
	*starting_range = prm.near_range;
	*first_sample = prm.first_sample;
	*width = prm.bytes_per_line;
	*xmin = (2 * (*first_sample)) + 1;
	*xmax = prm.good_bytes;
	*chirp = prm.chirp_slope;

	return(EXIT_SUCCESS);
}
/*--------------------------------------------------------------------------------------------------------------*/
long get_file_size(FILE *datafile)
{
long size;

	fseek(datafile, 0, SEEK_END);
	size = ftell(datafile);
	fclose(datafile);

	return(size);
}
/*--------------------------------------------------------------------------------------------------------------*/
int get_utc(double start_time, struct SAR_info sar, double *first_line_utc, double *last_line_utc, double *center_utc, double pri, int file_length)
{
double  tday, hr, mn, sc, ms;

	tday = start_time - floor(start_time);
	tday = start_time - floor(start_time);

	hr = floor(tday*24.0);
	tday = tday - hr/24.0;
	mn = floor(tday*60.0*24.0);
	tday = tday - mn/60.0/24.0;
	sc = floor(tday*60.0*60.0*24.0);
	tday = tday - sc/60.0/60.0/24.0;
	ms = floor(tday*1000.0*60.0*60.0*24.0);

	*first_line_utc = (double) (3600 * hr  + 60 * mn + sc + ms/1000.0);
	*last_line_utc = *first_line_utc + pri * file_length;
	*center_utc = (*first_line_utc + *last_line_utc) / 2.0;

	return(EXIT_SUCCESS);
}
/*--------------------------------------------------------------------------------------------------------------*/
int write_roi_orbit(struct ALOS_ORB orb, char *date)
{
int i;
FILE *orbit_rsc;
char fname[128];

	sprintf(fname,"hdr_data_points_%s.rsc",date);
	if ((orbit_rsc = fopen(fname,"w")) == NULL) die("error opening ",fname);

	for (i=0; i<orb.nd; i++){
		fprintf(orbit_rsc,"%-6.0lf",   orb.sec + (i * orb.dsec));
		/* position */
		fprintf(orbit_rsc," %-18.15E",   orb.points[i].px);
		fprintf(orbit_rsc," %-18.15E",   orb.points[i].py);
		fprintf(orbit_rsc," %-18.15E",   orb.points[i].pz);

		/* velocity */
		fprintf(orbit_rsc," %-18.15E",   orb.points[i].vx);
		fprintf(orbit_rsc," %-18.15E",   orb.points[i].vy);
		fprintf(orbit_rsc," %-18.15E\n", orb.points[i].vz);
		}

	fclose(orbit_rsc);

	return(EXIT_SUCCESS);
}
/*--------------------------------------------------------------------------------------------------------------*/
