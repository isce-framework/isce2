//////////////////////////////////////
// Cunren Liang
// California Institute of Technology
// Copyright 2019
//////////////////////////////////////

//this program is tested against resamp.c, the outputs of the two are exactly the same.

#include "resamp.h"

//ALOS I or Q mean = 15.5, so get 15 or 16 randomly here
//#define ZERO_VALUE (char)(15 + rand() % 2)
//I changed the dynamic range when reading data
//ALOS I or Q mean = 63.5, so get 63 or 64 randomly here
#define ZERO_VALUE (char)(63 + rand() % 2)
typedef struct {
  char re;
  char im;
} char_complex;
char_complex *array1d_char_complex(long nc);
void free_array1d_char_complex(char_complex *fcv);
void normalize_kernel(float *kernel, long start_index, long end_index);
int resamp_azimuth(char *slc2, char *rslc2, int nrg, int naz1, int naz2, double prf, double *dopcoeff, double *azcoef, int n, double beta){
  int i;
  int verbose = 0;
  if(verbose){
    printf("\n\ninput parameters:\n");
    printf("slc2: %s\n", slc2);
    printf("rslc2: %s\n", rslc2);
    printf("nrg: %d\n", nrg);
    printf("naz1: %d\n", naz1);
    printf("naz2: %d\n\n", naz2);
    printf("prf: %f\n\n", prf);
    for(i = 0; i < 4; i++){
      printf("dopcoeff[%d]: %e\n", i, dopcoeff[i]);
    }
    printf("\n");
    for(i = 0; i < 2; i++){
      printf("azcoef[%d]: %e\n", i, azcoef[i]);
    }
    printf("\n");
  }
  FILE *slc2fp;
  FILE *rslc2fp;
  int m;
  int interp_method;
  int edge_method;
  float azpos;
  float azoff;
  float az2;
  int azi2;
  float azf;
  int azfn;
  int hnm;
  int hn;
  float *sincc;
  float *kaiserc;
  float *kernel;
  float *azkernel;
  fcomplex *azkernel_fc;
  fcomplex *rgrs;
  fcomplex *azca;
  fcomplex *rgrsb;
  fcomplex *azrs;
  char_complex *inl;
  char_complex *outl;
  float *dop;
  float dopx;
  fcomplex **inb;
  int j, k, k1, k2;
  int tmp1, tmp2;
  int zero_flag;
  float ftmp1, ftmp2;
  fcomplex fctmp1, fctmp2;
  m = 10000;
  interp_method = 0;
  edge_method = 2;
  if((n % 2 == 0) || (n < 3)){
    fprintf(stderr, "number of samples to be used in the resampling must be odd, and larger or equal to than 3\n");
    exit(1);
  }
  slc2fp = openfile(slc2, "rb");
  rslc2fp = openfile(rslc2, "wb");
  hn = n / 2;
  hnm = n * m / 2;
  sincc = vector_float(-hnm, hnm);
  kaiserc = vector_float(-hnm, hnm);
  kernel = vector_float(-hnm, hnm);
  azkernel = vector_float(-hn, hn);
  azkernel_fc = vector_fcomplex(-hn, hn);
  rgrs = vector_fcomplex(-hn, hn);
  azca = vector_fcomplex(-hn, hn);
  rgrsb = vector_fcomplex(-hn, hn);
  azrs = array1d_fcomplex(nrg);
  inl = array1d_char_complex(nrg);
  outl = array1d_char_complex(nrg);
  dop = array1d_float(nrg);
  inb = array2d_fcomplex(naz2, nrg);
  sinc(n, m, sincc);
  kaiser(n, m, kaiserc, beta);
  for(i = -hnm; i <= hnm; i++)
    kernel[i] = kaiserc[i] * sincc[i];
  for(i = 0; i < nrg; i++){
    dop[i] = dopcoeff[0] + dopcoeff[1] * i + dopcoeff[2] * i * i + dopcoeff[3] * i * i * i;
    if(verbose){
      if(i % 500 == 0)
        printf("range sample: %5d, doppler centroid frequency: %8.2f Hz\n", i, dop[i]);
    }
  }
  for(i = 0; i < naz2; i++){
    readdata((char_complex *)inl, (size_t)nrg * sizeof(char_complex), slc2fp);
    for(j =0; j < nrg; j++){
      inb[i][j].re = inl[j].re;
      inb[i][j].im = inl[j].im;
    }
  }
  for(i = 0; i < naz1; i++){
    if((i + 1) % 100 == 0)
      fprintf(stderr,"processing line: %6d of %6d\r", i+1, naz1);
    for(j = 0; j < nrg; j++){
      azrs[j].re = 0.0;
      azrs[j].im = 0.0;
    }
    azpos = i;
    azoff = azcoef[0] + azpos * azcoef[1];
    az2 = i + azoff;
    azi2 = roundfi(az2);
    azf = az2 - azi2;
    azfn = roundfi(azf * m);
    if(edge_method == 0){
      if(azi2 < hn || azi2 > naz2 - 1 - hn){
        for(j = 0; j < nrg; j++){
          outl[j].re = ZERO_VALUE;
          outl[j].im = ZERO_VALUE;
        }
        writedata((char_complex *)outl, (size_t)nrg * sizeof(char_complex), rslc2fp);
        continue;
      }
    }
    else if(edge_method == 1){
      if(azi2 < 0 || azi2 > naz2 - 1){
        for(j = 0; j < nrg; j++){
          outl[j].re = ZERO_VALUE;
          outl[j].im = ZERO_VALUE;
        }
        writedata((char_complex *)outl, (size_t)nrg * sizeof(char_complex), rslc2fp);
        continue;
      }
    }
    else{
      if(azi2 < -hn || azi2 > naz2 - 1 + hn){
        for(j = 0; j < nrg; j++){
          outl[j].re = ZERO_VALUE;
          outl[j].im = ZERO_VALUE;
        }
        writedata((char_complex *)outl, (size_t)nrg * sizeof(char_complex), rslc2fp);
        continue;
      }
    }
    for(k = -hn; k <= hn; k++){
      tmp2 = k * m - azfn;
      if(tmp2 > hnm) tmp2 = hnm;
      if(tmp2 < -hnm) tmp2 = -hnm;
      azkernel[k] = kernel[tmp2];
    }
    normalize_kernel(azkernel, -hn, hn);
    for(j = 0; j < nrg; j++){
      for(k1 = -hn; k1 <= hn; k1++){
        if((azi2 + k1 >= 0)&&(azi2 + k1 <= naz2-1)){
          rgrs[k1].re = inb[azi2 + k1][j].re;
          rgrs[k1].im = inb[azi2 + k1][j].im;
        }
        else{
          rgrs[k1].re = ZERO_VALUE;
          rgrs[k1].im = ZERO_VALUE;
        }
      }
      dopx = dop[j];
      for(k = -hn; k <= hn; k++){
        ftmp1 = 2.0 * PI * dopx * k / prf;
        azca[k].re = cos(ftmp1);
        azca[k].im = sin(ftmp1);
        if(interp_method == 0){
         rgrsb[k] = cmul(rgrs[k], cconj(azca[k]));
          azrs[j].re += rgrsb[k].re * azkernel[k];
          azrs[j].im += rgrsb[k].im * azkernel[k];
        }
        else{
          azkernel_fc[k].re = azca[k].re * azkernel[k];
          azkernel_fc[k].im = azca[k].im * azkernel[k];
          azrs[j] = cadd(azrs[j], cmul(rgrs[k], azkernel_fc[k]));
        }
      }
      if(interp_method == 0){
        ftmp1 = 2.0 * PI * dopx * azf / prf;
        fctmp1.re = cos(ftmp1);
        fctmp1.im = sin(ftmp1);
        azrs[j] = cmul(azrs[j], fctmp1);
      }
    }
    for(j = 0; j < nrg; j++){
      outl[j].re = roundfi(azrs[j].re);
      outl[j].im = roundfi(azrs[j].im);
    }
    writedata((char_complex *)outl, (size_t)nrg * sizeof(char_complex), rslc2fp);
  }
  fprintf(stderr,"processing line: %6d of %6d\n", naz1, naz1);
  free_vector_float(sincc, -hnm, hnm);
  free_vector_float(kaiserc, -hnm, hnm);
  free_vector_float(kernel, -hnm, hnm);
  free_vector_float(azkernel, -hn, hn);
  free_vector_fcomplex(azkernel_fc, -hn, hn);
  free_vector_fcomplex(rgrs, -hn, hn);
  free_vector_fcomplex(azca, -hn, hn);
  free_vector_fcomplex(rgrsb, -hn, hn);
  free_array1d_fcomplex(azrs);
  free_array1d_char_complex(inl);
  free_array1d_char_complex(outl);
  free_array1d_float(dop);
  free_array2d_fcomplex(inb);
  fclose(slc2fp);
  fclose(rslc2fp);
  return 0;
}
char_complex *array1d_char_complex(long nc){
  char_complex *fcv;
  fcv = (char_complex*) malloc(nc * sizeof(char_complex));
  if(!fcv){
    fprintf(stderr,"Error: cannot allocate 1-D char complex array\n");
    exit(1);
  }
  return fcv;
}
void free_array1d_char_complex(char_complex *fcv){
  free(fcv);
}
void normalize_kernel(float *kernel, long start_index, long end_index){
  double sum;
  long i;
  sum = 0.0;
  for(i = start_index; i <= end_index; i++)
    sum += kernel[i];
  if(sum!=0)
    for(i = start_index; i <= end_index; i++)
      kernel[i] /= sum;
}
