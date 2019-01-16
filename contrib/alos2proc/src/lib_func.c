//////////////////////////////////////
// Cunren Liang, NASA JPL/Caltech
// Copyright 2017
//////////////////////////////////////


#include "resamp.h"

long next_pow2(long a){
  long i;
  long x;
  
  x = 2;
  while(x < a){
    x *= 2;
  }
  
  return x;
}

void circ_shift(fcomplex *in, int na, int nc){

  int i;
  int ncm;

  ncm = nc%na;

  if(ncm < 0){
    for(i = 0; i < abs(ncm); i++)
      left_shift(in, na);
  }
  else if(ncm > 0){
    for(i = 0; i < ncm; i++)
      right_shift(in, na);
  }
  else{ //ncm == 0, no need to shift
    i = 0;
  }
}

void left_shift(fcomplex *in, int na){

  int i;
  fcomplex x;

  if(na < 1){
    fprintf(stderr, "Error: array size < 1\n\n");
    exit(1);
  }
  else if(na > 1){
    x.re = in[0].re;
    x.im = in[0].im;
    for(i = 0; i <= na - 2; i++){
      in[i].re = in[i+1].re;
      in[i].im = in[i+1].im;
    }
    in[na-1].re = x.re;
    in[na-1].im = x.im;  
  }
  else{ //na==1, no need to shift
    i = 0;
  }
}

void right_shift(fcomplex *in, int na){

  int i;
  fcomplex x;

  if(na < 1){
    fprintf(stderr, "Error: array size < 1\n\n");
    exit(1);
  }
  else if(na > 1){
    x.re = in[na-1].re;
    x.im = in[na-1].im;
    for(i = na - 1; i >= 1; i--){
      in[i].re = in[i-1].re;
      in[i].im = in[i-1].im;
    }
    in[0].re = x.re;
    in[0].im = x.im;
  }
  else{ //na==1, no need to shift
    i = 0;
  }
}

int roundfi(float a){
  int b;

  if(a > 0)
    b = (int)(a + 0.5);
  else if (a < 0)
    b = (int)(a - 0.5);
  else
    b = a;

  return b;
}

void sinc(int n, int m, float *coef){

  int i;
  int hmn;

  hmn = n * m / 2;

  for(i=-hmn; i<=hmn; i++){
    if(i != 0){
      coef[i] = sin(PI * i / m) / (PI * i / m);
      //coef[i] = sin(pi * i / m) / (pi * i / m);
    }
    else{
      coef[i] = 1.0;
    }
  }

}

//kaiser() is equivalent to kaiser2()
//it is created to just keep the same style of sinc().
void kaiser(int n, int m, float *coef, float beta){

  int i;
  int hmn;
  float a;
  
  hmn = n * m / 2;

  for(i = -hmn; i <= hmn; i++){
    a = 1.0 - 4.0 * i * i / (n * m) / (n * m);
    coef[i] = bessi0(beta * sqrt(a)) / bessi0(beta);
  }
}

void kaiser2(float beta, int n, float *coef){

  int i;
  int hn;
  float a;

  hn = (n - 1) / 2;

  for(i = -hn; i<=hn; i++){
    a = 1.0 - 4.0 * i * i / (n - 1.0) / (n - 1.0);
    coef[i] = bessi0(beta * sqrt(a)) / bessi0(beta);
  }
}

void bandpass_filter(float bw, float bc, int n, int nfft, int ncshift, float beta, fcomplex *filter){

  int i;
  float *kw;
  int hn;
  fcomplex bwx, bcx;

  hn = (n-1)/2;

  if(n > nfft){
    fprintf(stderr, "Error: fft length too small!\n\n");
    exit(1);
  }
  if(abs(ncshift) > nfft){
    fprintf(stderr, "Error: fft length too small or shift too big!\n\n");
    exit(1);
  }

  //set all the elements to zero
  for(i = 0; i < nfft; i++){
    filter[i].re = 0.0;
    filter[i].im = 0.0;
  }

  //calculate kaiser window
  kw = vector_float(-hn, hn);
  kaiser2(beta, n, kw);

  //calculate filter
  for(i = -hn; i <= hn; i++){
    bcx.re = cos(bc * 2.0 * PI * i);
    bcx.im = sin(bc * 2.0 * PI * i);

    if(i == 0){
      bwx.re = 1.0;
      bwx.im = 0.0;
    }
    else{
      bwx.re = sin(bw * PI * i) / (bw * PI * i);
      bwx.im = 0.0;
    }
    
    filter[i+hn] = cmul(bcx, bwx);

    filter[i+hn].re = bw * kw[i] * filter[i+hn].re;
    filter[i+hn].im = bw * kw[i] * filter[i+hn].im;
  }

  //circularly shift filter, we shift the filter to left.
  ncshift = -abs(ncshift);
  circ_shift(filter, nfft, ncshift);

  free_vector_float(kw, -hn, hn);
}


float bessi0(float x)
{
  float ax,ans;
  double y;

  if ((ax=fabs(x)) < 3.75) {
    y=x/3.75;
    y*=y;
    ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
      +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
  } else {
    y=3.75/ax;
    ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
      +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
      +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
      +y*0.392377e-2))))))));
  }
  return ans;
}

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
void four1(float data[], unsigned long nn, int isign)
{
  unsigned long n,mmax,m,j,istep,i;
  double wtemp,wr,wpr,wpi,wi,theta;
  float tempr,tempi;

  n=nn << 1;
  j=1;
  for (i=1;i<n;i+=2) {
    if (j > i) {
      SWAP(data[j],data[i]);
      SWAP(data[j+1],data[i+1]);
    }
    m=nn;
    while (m >= 2 && j > m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  mmax=2;
  while (n > mmax) {
    istep=mmax << 1;
    theta=isign*(6.28318530717959/mmax);
    wtemp=sin(0.5*theta);
    wpr = -2.0*wtemp*wtemp;
    wpi=sin(theta);
    wr=1.0;
    wi=0.0;
    for (m=1;m<mmax;m+=2) {
      for (i=m;i<=n;i+=istep) {
        j=i+mmax;
        tempr=wr*data[j]-wi*data[j+1];
        tempi=wr*data[j+1]+wi*data[j];
        data[j]=data[i]-tempr;
        data[j+1]=data[i+1]-tempi;
        data[i] += tempr;
        data[i+1] += tempi;
      }
      wr=(wtemp=wr)*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
    }
    mmax=istep;
  }
}
#undef SWAP


