#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "defines.h"

typedef struct{float re,im;} fcomplex;	/* single precision complex data type */ 
 
#define WIN_SZ	  5		/* default window size */
#define THR     .4

/**
 * @param intFilename interferogram filename
 * @param gradFilename phase gradient filename
 * @param width number of samples per row
 * @param win_sz size of the window for the gradient calculation
 * @param thres phase gradient threshold
 * @param xmin starting range pixel offset
 * @param xmax last range pixel offset
 * @param ymin starting azimuth row
 * @param ymax last azimuth row offset
 */
int phase_slope(char *intFilename, char *gradFilename, int width, int win_sz, double thr, int xmin, int xmax, int ymin, int ymax)
{
  fcomplex *bufcz,**cmp,*cmpb,*tc,*ps;  /* line buffer, complex input data, row pointers */
  fcomplex psr,psaz;

  double *azw, *rw;		/* window weighting */
  double **win, *winb;
  double wt;			/* product of range and azimuth weights */
  double s1;			/* sum of window coefficients */
  double p1,p2,p3;		/* normalization powers */
  double r1;			/* radial distance */
  double psrm,psazm;		/* correlation amplitudes */
  
  float scr,scaz;		/* normalization factors */
  int nlines=0;			/* number of lines in the file */
  int xw,yh;			/* width, height of processed region */
  int i,j,k,l,n;			/* loop counters */
  int icnt;			/* line counter */
  int nrw,nazw;			/* size of filter windows in range, azimuth */
    
  FILE *intf, *psf;

  if (win_sz <= 0) {
      win_sz = WIN_SZ;
  }
  if (thr <= 0) {
      thr = THR;
  }

  intf = fopen(intFilename,"r"); 
  if (intf == NULL){fprintf(stderr,"ERROR: cannot open interferogram file: %s\n",intFilename);exit(-1);}

  psf = fopen(gradFilename,"w"); 
  if (psf == NULL){fprintf(stderr,"ERROR: cannot create range phase gradient file: %s\n",gradFilename);exit(-1);}

  if (xmax <= 0) {
     xmax=width-1;
  }
 
  fseek(intf, 0L, REL_EOF);				/* determine # lines in the file */
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

  if (xmax > width-1) xmax=width-1; 			/* check to see if xmax within bounds */
  xw=xmax-xmin+1;					/* width of array */
  yh=ymax-ymin+1;					/* height of array */ 
  fprintf(stderr,"processing window, xmin,xmax,ymin,ymax: %5d  %5d  %5d  %5d\n",xmin,xmax,ymin,ymax);
  fprintf(stderr,"processing window size, width, height:  %5d  %5d\n",xw,yh);
  fprintf(stderr,"window size size: %5d\n",win_sz);
  fprintf(stderr,"phase gradient correlation threshold: %8.4f\n",thr);
 
  bufcz = (fcomplex *)malloc(sizeof(fcomplex)*width);
  cmpb  = (fcomplex *)malloc(sizeof(fcomplex)*width*win_sz);
  cmp   = (fcomplex **)malloc(sizeof(fcomplex *)*win_sz);
  if (bufcz==NULL || cmpb==NULL ||  cmp==NULL){
    fprintf(stderr,"ERROR: failure to allocate space for complex data buffers!\n"); exit(-1);}
 
  nrw=win_sz;
  nazw=win_sz;

  ps    = (fcomplex *)malloc(sizeof(fcomplex)*width);
  rw    = (double *)malloc(sizeof(double)*nrw);
  azw   = (double *)malloc(sizeof(double)*nazw);
  winb  = (double *)malloc(sizeof(double)*nazw*nrw);
  win   = (double **)malloc(sizeof(double*)*nazw);
  rw    = (double *)malloc(sizeof(double)*nrw);
  azw   = (double *)malloc(sizeof(double)*nazw);
  winb  = (double *)malloc(sizeof(double)*nazw*nrw);
  win   = (double **)malloc(sizeof(double*)*nazw);
  if (ps == NULL || rw==NULL || azw==NULL || winb==NULL || win==NULL){
    fprintf(stderr,"ERROR: failure to allocate space for memory buffers!\n"); exit(-1);}
  for(k=0; k < nazw; k++)win[k] = winb+k*nrw;
 
#ifdef LINEAR
  fprintf(stderr,"\nrange correlation weights:\n");
  for(j=0; j < nrw; j++){
     rw[j]=1.0-fabs(2.0*(double)(j-nrw/2)/(win_sz+1));  
     fprintf(stderr,"index,coefficient: %6d %10.5f\n",j-nrw/2,rw[j]);
  }
  fprintf(stderr,"\nazimuth correlation weights:\n");    
  for(k=0; k< nazw; k++){
    azw[k]=1.0-fabs(2.0*(double)(k-nazw/2)/(win_sz+1)); 
    fprintf(stderr,"index,coefficient: %6d %10.5f\n",j-nazw/2,azw[k]);
  }
#else


  s1=0.0;
  fprintf(stderr,"# correlation weights (range,azimuth):   %6d %6d\n",nrw,nazw);
  for(k=0; k < win_sz; k++){
    for(j=0; j < win_sz; j++){
      r1 = sqrt(SQR(k-win_sz/2.0)+SQR(j-win_sz/2.0));
       win[k][j] = exp(-SQR(r1)/(SQR(win_sz/2.0)));
      s1 += win[k][j];
      fprintf(stderr,"indices,radius,weight: %6.2f %6.2f %10.3f %10.3f\n",k-nazw/2.,j-nrw/2.,r1,win[k][j]);
    }
  }
  fprintf(stderr,"\nsum of unnormalized weights: %10.3f\n",s1);

  fprintf(stderr,"\nnormalized window coefficients:\n");
  for(k=0; k < nazw; k++){
    for(j=0; j < nrw; j++){
      win[k][j] /= s1;
      fprintf(stderr,"indicies,weight: %4d %4d %10.3f\n",k,j,win[k][j]);
     }
  }
 
#endif
  psr.re=0.0; psr.im=0.0;
  psaz.re=0.0; psaz.im=0.0;

  for(j=0; j < width; j++){
    bufcz[j].re=0.0; bufcz[j].im=0.0; 
    ps[j].re=0.0; ps[j].im=0.0;
  }

  for(i=0; i < win_sz; i++){						/* initialize array pointers */
    cmp[i] = cmpb + i*width;
  }
 
  for(icnt=0,i=0; i < (ymin+win_sz/2); i++){
    fwrite((char *)bufcz,sizeof(fcomplex),width,psf); 			/* write null lines */
    icnt++;
  }

  fseek(intf,ymin*width*sizeof(fcomplex), REL_BEGIN); 			/* seek start line of interferogram */
  fread((char *)cmpb,sizeof(fcomplex),width*(win_sz-1),intf); 		/* read  interferogram file */
 
  for (i=win_sz/2; i < yh-win_sz/2; i++){
    if(i%10 == 0)fprintf(stderr,"\rprocessing line: %d", i);
   
    fread((char *)cmp[win_sz-1],sizeof(fcomplex),width,intf); 		/* interferogram file */

    for (j=xmin+win_sz/2; j < xw-win_sz/2; j++){    			/* move across the image */   
      psr.re=0.0; psr.im=0.0; psaz.re=0.0; psaz.im=0.0; p1=0.0; p2=0.0; p3=0.0;

      for (k=1; k < win_sz; k++){
        for (n=j-win_sz/2+1,l=0; n < j-win_sz/2+win_sz; n++,l++){
          wt = win[k][l];
          psr.re += (cmp[k][n].re*cmp[k][n-1].re + cmp[k][n].im*cmp[k][n-1].im)*wt;
          psr.im += (cmp[k][n].im*cmp[k][n-1].re - cmp[k][n].re*cmp[k][n-1].im)*wt;
          psaz.re += (cmp[k][n].re*cmp[k-1][n].re + cmp[k][n].im*cmp[k-1][n].im)*wt;
          psaz.im += (cmp[k][n].im*cmp[k-1][n].re - cmp[k][n].re*cmp[k-1][n].im)*wt;
	  p1 += wt*(SQR(cmp[k][n].re)+SQR(cmp[k][n].im));
          p2 += wt*(SQR(cmp[k][n-1].re)+SQR(cmp[k][n-1].im));
          p3 += wt*(SQR(cmp[k-1][n].re)+SQR(cmp[k-1][n].im));  
        }
      }

      scr = sqrt(p1*p2);
      scaz = sqrt(p1*p3);

      if (scr > 0.0){psr.re /= scr; psr.im /= scr;}
      else{psr.re = 0.0; psr.im = 0.0;}

      if (scaz > 0.0){psaz.re /= scaz; psaz.im /= scaz;}
      else{psaz.re = 0.0; psaz.im = 0.0;}     

      psrm=sqrt(SQR(psr.re)+SQR(psr.im));
      psazm=sqrt(SQR(psaz.re)+SQR(psaz.im));

      if ((psrm > thr) && (psazm > thr)){
        ps[j].re = atan2((double)psr.im,(double)psr.re);
        ps[j].im = atan2((double)psaz.im,(double)psaz.re);
      }
      else{ps[j].re = 0.0; ps[j].im = 0.0;}
    }
   
    fwrite((char *)ps, sizeof(fcomplex), width, psf);
    icnt++;
								/* buffer circular shift */
    tc=cmp[0];							/* save pointer addresses of the oldest line */
    for (k=1; k < win_sz; k++){					/* shift addresses */
      cmp[k-1]=cmp[k];
    }	
    cmp[win_sz-1]=tc;						/* new data will overwrite the oldest line */    
  } 
  
  for(j=0; j < win_sz/2; j++){
    fwrite((char *)bufcz, sizeof(fcomplex), width, psf); 	/* write null lines */
    icnt++;
  }

  fprintf(stderr,"\noutput lines: %d\n", icnt);

  return 0;
}  
