//////////////////////////////////////
// Cunren Liang, NASA JPL/Caltech
// Copyright 2017
//////////////////////////////////////


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#define NR_END 1
#define FREE_ARG char*
#define PI 3.1415926535897932384626433832795028841971693993751058

typedef struct {
  float re;
  float im;
} fcomplex;

typedef struct {
  double re;
  double im;
} dcomplex;

//allocate arrays
signed char *vector_char(long nl, long nh);
void free_vector_char(signed char *v, long nl, long nh);
unsigned char *vector_unchar(long nl, long nh);
void free_vector_unchar(unsigned char *v, long nl, long nh);
int *vector_int(long nl, long nh);
void free_vector_int(int *v, long nl, long nh);
float *vector_float(long nl, long nh);
void free_vector_float(float *v, long nl, long nh);
double *vector_double(long nl, long nh);
void free_vector_double(double *v, long nl, long nh);
fcomplex *vector_fcomplex(long nl, long nh);
void free_vector_fcomplex(fcomplex *v, long nl, long nh);
signed char **matrix_char(long nrl, long nrh, long ncl, long nch);
void free_matrix_char(signed char **m, long nrl, long nrh, long ncl, long nch);
unsigned char **matrix_unchar(long nrl, long nrh, long ncl, long nch);
void free_matrix_unchar(unsigned char **m, long nrl, long nrh, long ncl, long nch);
float **matrix_float(long nrl, long nrh, long ncl, long nch);
void free_matrix_float(float **m, long nrl, long nrh, long ncl, long nch);
double **matrix_double(long nrl, long nrh, long ncl, long nch);
void free_matrix_double(double **m, long nrl, long nrh, long ncl, long nch);


//allocate C-style arrays
FILE **array1d_FILE(long nc);
void free_array1d_FILE(FILE **fv);
signed char *array1d_char(long nc);
void free_array1d_char(signed char *fv);
unsigned char *array1d_unchar(long nc);
void free_array1d_unchar(unsigned char *fv);
int *array1d_int(long nc);
void free_array1d_int(int *fv);
float *array1d_float(long nc);
void free_array1d_float(float *fv);
double *array1d_double(long nc);
void free_array1d_double(double *fv);
fcomplex *array1d_fcomplex(long nc);
void free_array1d_fcomplex(fcomplex *fcv);
dcomplex *array1d_dcomplex(long nc);
void free_array1d_dcomplex(dcomplex *fcv);
signed char **array2d_char(long nl, long nc);
void free_array2d_char(signed char **m);
unsigned char **array2d_unchar(long nl, long nc);
void free_array2d_unchar(unsigned char **m);
float **array2d_float(long nl, long nc);
void free_array2d_float(float **m);
double **array2d_double(long nl, long nc);
void free_array2d_double(double **m);
fcomplex **array2d_fcomplex(long nl, long nc);
void free_array2d_fcomplex(fcomplex **m);

//handling error
void nrerror(char error_text[]);

//complex operations
fcomplex cmul(fcomplex a, fcomplex b);
fcomplex cconj(fcomplex z);
fcomplex cadd(fcomplex a, fcomplex b);
float xcabs(fcomplex z);
float cphs(fcomplex z);

//functions
long next_pow2(long a);
void circ_shift(fcomplex *in, int na, int nc);
void left_shift(fcomplex *in, int na);
void right_shift(fcomplex *in, int na);
int roundfi(float a);
void sinc(int n, int m, float *coef);
void kaiser(int n, int m, float *coef, float beta);
void kaiser2(float beta, int n, float *coef);
void bandpass_filter(float bw, float bc, int n, int nfft, int ncshift, float beta, fcomplex *filter);
float bessi0(float x);
void four1(float data[], unsigned long nn, int isign);

//file operations
FILE *openfile(char *filename, char *pattern);
void readdata(void *data, size_t blocksize, FILE *fp);
void writedata(void *data, size_t blocksize, FILE *fp);
long file_length(FILE* fp, long width, long element_size);

