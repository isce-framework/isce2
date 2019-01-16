/* include files to define sarleader structure */
#include "data_ALOS.h"
#include "data_ALOSE.h"
#include "orbit_ALOS.h"
#include "sarleader_ALOS.h"
#include "sarleader_fdr.h"

/* function prototypes 				*/
void ALOS_ldr_orbit(struct ALOS_ORB *, struct PRM *);
void calc_height_velocity(struct ALOS_ORB *, struct PRM *, double, double, double *, double *, double *, double *, double *);
void calc_dop(struct  PRM *);
void cfft1d_(int *, fcomplex *, int *);
void read_data(fcomplex *, unsigned char *, int, struct PRM *);
void null_sio_struct(struct PRM *);
void get_sio_struct(FILE *, struct PRM *);
void put_sio_struct(struct PRM, FILE *);
void get_string(char *, char *, char *, char *);
void get_int(char *, char *, char *, int *);
void get_double(char *, char *, char *, double *);
void hermite_c(double *, double *, double *, int, int, double, double *, int *);
void interpolate_ALOS_orbit_slow(struct ALOS_ORB *, double, double *, double *, double *, int *);
void interpolate_ALOS_orbit(struct ALOS_ORB *, double *, double *, double *, double, double *, double *, double *, int *);
void get_orbit_info(struct ALOS_ORB *, struct SAR_info);
void get_attitude_info(struct ALOS_ATT *, int, struct SAR_info);
void print_binary_position(struct sarleader_binary *, int, FILE *, FILE *);
void read_ALOS_sarleader(FILE *, struct PRM *, struct ALOS_ORB *);
void set_ALOS_defaults(struct PRM *);
void ALOS_ldr_prm(struct SAR_info, struct PRM *);
int is_big_endian_(void);
int is_big_endian__(void);
void die (char *, char *);
void cross3_(double *, double *, double *);
void get_seconds(struct PRM, double *, double *);
void plh2xyz(double *, double *, double, double);
void xyz2plh(double *, double *, double, double);
void polyfit(double *, double *, double *, int *, int *);
void gauss_jordan(double **, double *, double *, int *);
