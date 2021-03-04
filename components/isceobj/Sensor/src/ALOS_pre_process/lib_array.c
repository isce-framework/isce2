//////////////////////////////////////
// Cunren Liang, NASA JPL/Caltech
// Copyright 2017
//////////////////////////////////////


#include "resamp.h"

/****************************************************************/
/*                     allocating arrays                        */
/****************************************************************/

signed char *vector_char(long nl, long nh)
/* allocate a signed char vector with subscript range v[nl..nh] */
{
  signed char *v;

  v=(signed char *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(signed char)));
  if (!v){
    fprintf(stderr,"Error: cannot allocate 1-D vector\n");
    exit(1);  
  }
  
  return v-nl+NR_END;
}

void free_vector_char(signed char *v, long nl, long nh)
/* free a signed char vector allocated with vector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}

unsigned char *vector_unchar(long nl, long nh)
/* allocate a unsigned char vector with subscript range v[nl..nh] */
{
  unsigned char *v;

  v=(unsigned char *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(unsigned char)));
  if (!v){
    fprintf(stderr,"Error: cannot allocate 1-D vector\n");
    exit(1);  
  }
  
  return v-nl+NR_END;
}

void free_vector_unchar(unsigned char *v, long nl, long nh)
/* free a unsigned char vector allocated with vector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}

int *vector_int(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
  int *v;

  v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
  if (!v) nrerror("Error: cannot allocate vector_int()");
  return v-nl+NR_END;
}

void free_vector_int(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}

float *vector_float(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
  float *v;

  v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
  if (!v){
    fprintf(stderr,"Error: cannot allocate 1-D vector\n");
    exit(1);  
  }
  
  return v-nl+NR_END;
}

void free_vector_float(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}

double *vector_double(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
  double *v;

  v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
  if (!v){
    fprintf(stderr,"Error: cannot allocate 1-D vector\n");
    exit(1);  
  }
  
  return v-nl+NR_END;
}

void free_vector_double(double *v, long nl, long nh)
/* free a double vector allocated with vector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}

fcomplex *vector_fcomplex(long nl, long nh)
/* allocate a fcomplex vector with subscript range v[nl..nh] */
{
  fcomplex *v;

  v=(fcomplex *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(fcomplex)));
  if (!v) nrerror("cannot allocate fcvector()");
  return v-nl+NR_END;
}

void free_vector_fcomplex(fcomplex *v, long nl, long nh)
/* free a fcomplex vector allocated with fcvector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}

signed char **matrix_char(long nrl, long nrh, long ncl, long nch)
/* allocate a signed char matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  signed char **m;

  /* allocate pointers to rows */
  m=(signed char **) malloc((size_t)((nrow+NR_END)*sizeof(signed char*)));
  if (!m) nrerror("Error: cannot allocate vector2d_float()");
  m += NR_END;
  m -= nrl;

  /* allocate rows and set pointers to them */
  m[nrl]=(signed char *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(signed char)));
  if (!m[nrl]) nrerror("Error: cannot allocate vector2d_float()");
  m[nrl] += NR_END;
  m[nrl] -= ncl;

  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

  /* return pointer to array of pointers to rows */
  return m;
}

void free_matrix_char(signed char **m, long nrl, long nrh, long ncl, long nch)
/* free a signed char matrix allocated by matrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-NR_END));
  free((FREE_ARG) (m+nrl-NR_END));
}

unsigned char **matrix_unchar(long nrl, long nrh, long ncl, long nch)
/* allocate a unsigned char matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  unsigned char **m;

  /* allocate pointers to rows */
  m=(unsigned char **) malloc((size_t)((nrow+NR_END)*sizeof(unsigned char*)));
  if (!m) nrerror("Error: cannot allocate vector2d_float()");
  m += NR_END;
  m -= nrl;

  /* allocate rows and set pointers to them */
  m[nrl]=(unsigned char *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(unsigned char)));
  if (!m[nrl]) nrerror("Error: cannot allocate vector2d_float()");
  m[nrl] += NR_END;
  m[nrl] -= ncl;

  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

  /* return pointer to array of pointers to rows */
  return m;
}

void free_matrix_unchar(unsigned char **m, long nrl, long nrh, long ncl, long nch)
/* free a unsigned char matrix allocated by matrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-NR_END));
  free((FREE_ARG) (m+nrl-NR_END));
}

float **matrix_float(long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  float **m;

  /* allocate pointers to rows */
  m=(float **) malloc((size_t)((nrow+NR_END)*sizeof(float*)));
  if (!m) nrerror("Error: cannot allocate vector2d_float()");
  m += NR_END;
  m -= nrl;

  /* allocate rows and set pointers to them */
  m[nrl]=(float *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(float)));
  if (!m[nrl]) nrerror("Error: cannot allocate vector2d_float()");
  m[nrl] += NR_END;
  m[nrl] -= ncl;

  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

  /* return pointer to array of pointers to rows */
  return m;
}

void free_matrix_float(float **m, long nrl, long nrh, long ncl, long nch)
/* free a float matrix allocated by matrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-NR_END));
  free((FREE_ARG) (m+nrl-NR_END));
}

double **matrix_double(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  double **m;

  /* allocate pointers to rows */
  m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
  if (!m) nrerror("Error: cannot allocate vector2d_double()");
  m += NR_END;
  m -= nrl;

  /* allocate rows and set pointers to them */
  m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
  if (!m[nrl]) nrerror("Error: cannot allocate vector2d_double()");
  m[nrl] += NR_END;
  m[nrl] -= ncl;

  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

  /* return pointer to array of pointers to rows */
  return m;
}

void free_matrix_double(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by matrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-NR_END));
  free((FREE_ARG) (m+nrl-NR_END));
}



/****************************************************************/
/*                   allocating C-style arrays                  */
/****************************************************************/

FILE **array1d_FILE(long nc){

  FILE **fv;

  fv = (FILE **)malloc(nc * sizeof(FILE *));
  if(!fv){
    fprintf(stderr,"Error: cannot allocate 1-D FILE array\n");
    exit(1);
  }

  return fv;
}

void free_array1d_FILE(FILE **fv){
  free(fv);
}

signed char *array1d_char(long nc){

  signed char *fv;

  fv = (signed char*) malloc(nc * sizeof(signed char));
  if(!fv){
    fprintf(stderr,"Error: cannot allocate 1-D signed char vector\n");
    exit(1);
  }

  return fv;
}

void free_array1d_char(signed char *fv){
  free(fv);
}

unsigned char *array1d_unchar(long nc){

  unsigned char *fv;

  fv = (unsigned char*) malloc(nc * sizeof(unsigned char));
  if(!fv){
    fprintf(stderr,"Error: cannot allocate 1-D unsigned char vector\n");
    exit(1);
  }

  return fv;
}

void free_array1d_unchar(unsigned char *fv){
  free(fv);
}

int *array1d_int(long nc){

  int *fv;

  fv = (int*) malloc(nc * sizeof(int));
  if(!fv){
    fprintf(stderr,"Error: cannot allocate 1-D int array\n");
    exit(1);
  }

  return fv;
}

void free_array1d_int(int *fv){
  free(fv);
}

float *array1d_float(long nc){

  float *fv;

  fv = (float*) malloc(nc * sizeof(float));
  if(!fv){
    fprintf(stderr,"Error: cannot allocate 1-D float vector\n");
    exit(1);
  }

  return fv;
}

void free_array1d_float(float *fv){
  free(fv);
}

double *array1d_double(long nc){

  double *fv;

  fv = (double*) malloc(nc * sizeof(double));
  if(!fv){
    fprintf(stderr,"Error: cannot allocate 1-D double vector\n");
    exit(1);
  }

  return fv;
}

void free_array1d_double(double *fv){
  free(fv);
}

fcomplex *array1d_fcomplex(long nc){

  fcomplex *fcv;

  fcv = (fcomplex*) malloc(nc * sizeof(fcomplex));
  if(!fcv){
    fprintf(stderr,"Error: cannot allocate 1-D float complex vector\n");
    exit(1);
  }

  return fcv;

}

void free_array1d_fcomplex(fcomplex *fcv){
  free(fcv);
}

dcomplex *array1d_dcomplex(long nc){

  dcomplex *fcv;

  fcv = (dcomplex*) malloc(nc * sizeof(dcomplex));
  if(!fcv){
    fprintf(stderr,"Error: cannot allocate 1-D double complex vector\n");
    exit(1);
  }

  return fcv;

}

void free_array1d_dcomplex(dcomplex *fcv){
  free(fcv);
}

signed char **array2d_char(long nl, long nc){
/* allocate a signed char 2-D matrix */

  signed char **m;
  int i;

  /* allocate pointers to rows */
  m = (signed char **) malloc(nl * sizeof(signed char *));
  if(!m){
    fprintf(stderr,"Error: cannot allocate 2-D matrix\n");
    exit(1);
  }
 
  /* allocate rows */ 
  m[0] = (signed char*) malloc(nl * nc * sizeof(signed char));
  if(!m[0]){
    fprintf(stderr,"Error: cannot allocate 2-D matrix\n");
    exit(1);
  }

   /* set pointers */
  for(i = 1; i < nl; i++){
    m[i] = m[i-1] + nc;
  }

  return m;
}

void free_array2d_char(signed char **m){
/* free a signed char matrix allocated by farray2d() */
  free(m[0]);
  free(m);
}

unsigned char **array2d_unchar(long nl, long nc){
/* allocate a unsigned char 2-D matrix */

  unsigned char **m;
  int i;

  /* allocate pointers to rows */
  m = (unsigned char **) malloc(nl * sizeof(unsigned char *));
  if(!m){
    fprintf(stderr,"Error: cannot allocate 2-D matrix\n");
    exit(1);
  }
 
  /* allocate rows */ 
  m[0] = (unsigned char*) malloc(nl * nc * sizeof(unsigned char));
  if(!m[0]){
    fprintf(stderr,"Error: cannot allocate 2-D matrix\n");
    exit(1);
  }

   /* set pointers */
  for(i = 1; i < nl; i++){
    m[i] = m[i-1] + nc;
  }

  return m;
}

void free_array2d_unchar(unsigned char **m){
/* free a signed unchar matrix allocated by farray2d() */
  free(m[0]);
  free(m);
}

float **array2d_float(long nl, long nc){
/* allocate a float 2-D matrix */

  float **m;
  int i;

  /* allocate pointers to rows */
  m = (float **) malloc(nl * sizeof(float *));
  if(!m){
    fprintf(stderr,"Error: cannot allocate 2-D matrix\n");
    exit(1);
  }
 
  /* allocate rows */ 
  m[0] = (float*) malloc(nl * nc * sizeof(float));
  if(!m[0]){
    fprintf(stderr,"Error: cannot allocate 2-D matrix\n");
    exit(1);
  }

   /* set pointers */
  for(i = 1; i < nl; i++){
    m[i] = m[i-1] + nc;
  }

  return m;
}

void free_array2d_float(float **m){
/* free a float matrix allocated by farray2d() */
  free(m[0]);
  free(m);
}

double **array2d_double(long nl, long nc){
/* allocate a double 2-D matrix */

  double **m;
  int i;

  /* allocate pointers to rows */
  m = (double **) malloc(nl * sizeof(double *));
  if(!m){
    fprintf(stderr,"Error: cannot allocate 2-D matrix\n");
    exit(1);
  }
 
  /* allocate rows */ 
  m[0] = (double*) malloc(nl * nc * sizeof(double));
  if(!m[0]){
    fprintf(stderr,"Error: cannot allocate 2-D matrix\n");
    exit(1);
  }

   /* set pointers */
  for(i = 1; i < nl; i++){
    m[i] = m[i-1] + nc;
  }

  return m;
}

void free_array2d_double(double **m){
/* free a double matrix allocated by farray2d() */
  free(m[0]);
  free(m);
}

fcomplex **array2d_fcomplex(long nl, long nc){
/* allocate a fcomplex 2-D matrix */

  fcomplex **m;
  int i;

  /* allocate pointers to rows */
  m = (fcomplex **) malloc(nl * sizeof(fcomplex *));
  if(!m){
    fprintf(stderr,"Error: cannot allocate 2-D matrix\n");
    exit(1);
  }
 
  /* allocate rows */ 
  m[0] = (fcomplex*) malloc(nl * nc * sizeof(fcomplex));
  if(!m[0]){
    fprintf(stderr,"Error: cannot allocate 2-D matrix\n");
    exit(1);
  }

   /* set pointers */
  for(i = 1; i < nl; i++){
    m[i] = m[i-1] + nc;
  }

  return m;
}

void free_array2d_fcomplex(fcomplex **m){
/* free a fcomplex matrix allocated by fcarray2d() */
  free(m[0]);
  free(m);
}


/****************************************************************/
/*                         handling error                       */
/****************************************************************/

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
  fprintf(stderr,"Numerical Recipes run-time error...\n");
  fprintf(stderr,"%s\n",error_text);
  fprintf(stderr,"...now exiting to system...\n");
  exit(1);
}
