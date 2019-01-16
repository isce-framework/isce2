/******************************************************************************
This program comes from PH_STAT.C, originally 'std' (the output array and
so the output file) contained the phase standard deviation. In order to use it 
to generate a Mask file, we modified 'std' in the way that now it represents 
the correlation with 0 in very low correlation area, 1 in case of perfect 
correlation (and of course you may find some values between 0 and 1)
here is the change : std_new = 1 - MIN(std_old,1)    

Modification : Frederic CRAMPE (11/06/98)   

generalized approach:  std_new = 1 - MIN(std_old,thresh)/thresh 
where thresh is input threshold beyond which std_new is set to 0.
thresh is input as the 4th variable.
         
Modification : Mark SIMONS (11/29/98)               
******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "defines.h"

typedef struct{float re,im;} fcomplex;	/* single precision complex data type */ 
 
#define WIN_SZ	  5		/* default window size */

/**
 * Generate mask based on phase standard deviation.
 *
 * @param intFilename interferogram filename
 * @param slopeFilename phase slope data filename
 * @param stdFilename phase standard deviation filename
 * @param thresh phase standard deviation threshold
 * @param width number of samples per row
 * @param nrw size of range smoothing window
 * @param narw size of azimuth smoothing window
 * @param xmin starting range pixel offset
 * @param xmax last range pixel offset
 * @param ymin starting azimuth row offset
 * @param ymax last azimuth row offset
 */
int
phase_mask(char *intFilename, char *slopeFilename, char *stdFilename, double thresh, int width, int nrw, int nazw, int xmin, int xmax, int ymin, int ymax)
{
  fcomplex **cmp,*cmpb; 
  fcomplex **sl, *slb, *tc, *t3;
  fcomplex **sw,*swb,s4;

  double **win,*winb;
  double ar,ai;
  double wt;			/* product of range and azimuth weights */
  double sum;			/* sum of window coefficients */
  double c1,s1;			/* sine and cosine */
  double azw,rw;		/* azimuth and range window weights */
  double azph, ph;
  double ps;			/* phase value */
  double re1,im1;		/* real and imaginary components */
  double ph_av,ph2;		/* mean and varience of the detrended phase data */
  double s4m;

  float sc;
  float *std,*bufz;

  int nlines=0;			/* number of lines in the file */
  int xw,yh;			/* width, height of processed region */
  int i,j,k,l,n,ic;		/* loop counters */
  int icnt;			/* line counter */
     
  FILE *intf,*slf,*stdf;

  if (nrw <= 0 ) {
      nrw = WIN_SZ;
  }
  if (nazw <= 0) {
      nazw = WIN_SZ;
  }


  intf = fopen(intFilename,"r"); 
  if (intf == NULL){fprintf(stderr,"ERROR: cannot open interferogram file: %s\n",intFilename);exit(-1);}

  slf = fopen(slopeFilename,"r"); 
  if (slf == NULL){fprintf(stderr,"ERROR: cannot open slope data file: %s\n",slopeFilename); exit(-1);}

  stdf = fopen(stdFilename,"w"); 
  if (stdf == NULL){fprintf(stderr,"ERROR: cannot open standard deviation file: %s\n",stdFilename); exit(-1);}


  if (xmax <= 0) {
      xmax=width-1;
  }
 
  fseek(intf, 0L, REL_EOF);
  nlines=(int)ftell(intf)/(width*sizeof(fcomplex));
  fprintf(stderr,"#lines in the interferogram file: %d\n",nlines); 
  rewind(intf);
  ymax=nlines-1;


  if (ymax <= 0) {
      ymax = nlines-1;
  } else if (ymax > nlines-1){
    ymax = nlines-1; 
    fprintf(stderr,"insufficient #lines in the file, ymax: %d\n",ymax);
  }

  sc = 1./(float)SQR(nrw*nazw-1);
  if (xmax > width-1) xmax=width-1; 		/* check to see if xmax within bounds */
  xw=xmax-xmin+1;				/* width of array */
  yh=ymax-ymin+1;				/* height of array */ 
  fprintf(stderr,"processing window, xmin,xmax,ymin,ymax: %5d  %5d  %5d  %5d\n",xmin,xmax,ymin,ymax);
  fprintf(stderr,"processing window size, width, height:  %5d  %5d\n",xw,yh);
  
  cmpb  = (fcomplex *)malloc(sizeof(fcomplex)*width*nazw);
  cmp   = (fcomplex **)malloc(sizeof(fcomplex *)*nazw);

  if (cmpb==NULL ||  cmp==NULL){
    fprintf(stderr,"ERROR: failure to allocate space for complex data buffers!\n"); exit(-1);}

  sw   = (fcomplex **)malloc(sizeof(double*)*nazw);
  swb  = (fcomplex *)malloc(sizeof(double)*nazw*nrw);
 
  win   = (double **)malloc(sizeof(double*)*nazw);
  winb  = (double *)malloc(sizeof(double)*nazw*nrw);
  sl    = (fcomplex **)malloc(sizeof(fcomplex *)*nazw);
  slb   = (fcomplex *)malloc(sizeof(fcomplex)*width*nazw);
  std   = (float *)malloc(sizeof(float)*width);
  bufz  = (float *)malloc(sizeof(float)*width);
 
  if (sl==NULL || slb==NULL || winb==NULL || win==NULL || bufz==NULL ||
       std==NULL || sw == NULL || swb == NULL){
    fprintf(stderr,"ERROR: failure to allocate space for memory buffers!\n"); exit(-1);
  }
  
  for(k=0; k < nazw; k++){
    win[k] = winb+k*nrw;
    sw[k] = swb+k*nrw;
  }
 
  sum=0.0;
  fprintf(stderr,"# correlation weights (range,azimuth):   %6d %6d\n",nrw,nazw);
  for(k=0; k < nazw; k++){
    for(j=0; j < nrw; j++){
      rw=1.0-fabs(2.0*(double)(j-nrw/2)/(nrw+1));  
      azw=1.0-fabs(2.0*(double)(k-nazw/2)/(nazw+1)); 
      win[k][j] =  rw*azw;
      sum += win[k][j];
      fprintf(stderr,"indices,radius,weight: %6d %6d %10.3f\n",k-nazw/2,j-nrw/2,win[k][j]);
    }
  }
  fprintf(stderr,"\nsum of unnormalized weights: %10.3f\n",sum);

  for(k=0; k < nazw; k++){
    for(j=0; j < nrw; j++){
      win[k][j] /= sum;
    }
  }
   
  for(j=0; j < width; j++){
    bufz[j]=1.0;
    std[j]=1.0; 
  }
 
  for(i=0; i < nazw; i++){				/* initialize array pointers */
    cmp[i] = cmpb + i*width;
    sl[i]  = slb  + i*width;
  }
 
  for(icnt=0,i=0; i < (ymin+nazw/2); i++){
    fwrite((char *)bufz,sizeof(float),width,stdf); 			 
    icnt++;
  }

  fseek(intf,ymin*width*sizeof(fcomplex), REL_BEGIN); 		/* seek start line of interferogram */
  fread((char *)cmpb,sizeof(fcomplex),width*(nazw-1),intf); 	/* read  interferogram file */

  fseek(slf,ymin*width*sizeof(fcomplex), REL_BEGIN);  		/* seek start line of slopes */
  fread((char *)slb,sizeof(fcomplex),width*(nazw-1), slf); 	/* read  slopes */


  for (i=nazw/2; i < yh-nazw/2; i++){
    if(i%10 == 0)fprintf(stderr,"\rprocessing line: %d", i);
   
    fread((char *)cmp[nazw-1],sizeof(fcomplex),width,intf); 	/* interferogram file */
    fread((char *)sl[nazw-1],sizeof(fcomplex),width,slf); 	/* slope data */

    for (j=xmin+nrw/2; j < xw-nrw/2; j++){    			/* move across the image */   
      ic=0; s4.re=0.0; s4.im=0.0;
      for (k=0; k < nazw; k++){
        azph = (k-nazw/2.0)*sl[nazw/2][j].im;
        for (n=j-nrw/2, l=0; n < j-nrw/2+nrw; n++,l++){
          ph = (l-nrw/2.0)*sl[nazw/2][j].re + azph;        
          wt = win[k][l];
          c1 = cos(ph); s1 = -sin(ph); 
           
          sw[k][l].re = (cmp[k][n].re*c1 - cmp[k][n].im*s1);
          sw[k][l].im = (cmp[k][n].re*s1 + cmp[k][n].im*c1);
          s4.re +=  sw[k][l].re; s4.im +=sw[k][l].im;
          ic++;  
        }
      }
      s4m = sqrt((double)(s4.re*s4.re + s4.im*s4.im));
      if(s4m > 0.0){s4.re /= s4m; s4.im = -s4.im/s4m;}		/* conjugate s4 and make a unit vector */
      else{s4.re=0.0; s4.im=0.0;}

      ph_av = 0.0; ph2 = 0.0;
      
      for(k=0; k < nazw; k++){
        for (l=0; l < nrw; l++){
          wt = win[k][l];          
          re1 = sw[k][l].re*s4.re - sw[k][l].im*s4.im;
          im1 = sw[k][l].im*s4.re + sw[k][l].re*s4.im;
          if(re1 != 0.0){
             ps =  atan2(im1,re1); 
             ph_av += wt*ps;
             ph2 += wt*(ps * ps);
	  } 
        }
      }

      if ((ph2 > 0.) && (ic > 1)){
         std[j] = 1.0 - MIN(thresh,sqrt(ph2 - ph_av*ph_av))/thresh; 
      }     
      else std[j] = 1.0;
    }

    fwrite((char *)std, sizeof(float), width, stdf);
    icnt++;
						/* buffer circular shift */
    t3=sl[0]; tc=cmp[0];			/* save pointer addresses of the oldest line */
    for (k=1; k < nazw; k++){			/* shift addresses */
      sl[k-1] = sl[k]; 
      cmp[k-1] = cmp[k];
    }	
    sl[nazw-1] = t3; cmp[nazw-1] = tc;     
  } 
  
  for(j=0; j < nazw/2; j++){
    fwrite((char *)bufz, sizeof(float), width, stdf);
    icnt++;
  }

  fprintf(stderr,"\noutput lines: %d\n", icnt);
  return 0;
}  
