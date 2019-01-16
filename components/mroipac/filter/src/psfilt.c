#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "psfilt.h"
#include "defines.h"
	
unsigned int nfft[3];
int xmin=0;						/* window column minima */
int ymin=0;						/* window row minima */
int width, ymax, xmax;					/* interferogram width, window maxima */

void psfilt(char *int_filename, char *sm_filename, int width, int nlines, double alpha, int step, int xmin, int xmax, int ymin, int ymax)
{
   fcomplex *bufcz, **cmp;		/* interferogram line buffer, complex input data, row pointers */
   fcomplex **sm, **seg_fft, *seg_fftb;	/* smoothed interferogram, 2d fft of segment */
   fcomplex **tmp, **tmp1;		/* arrays of pointers for temp storage of line pointers in circular buffers */
   double **wf, *wfb;			/* 2-D weights */

   float *data;				/* pointer to floats for FFT, union with seg_fft */
   double rw,azw;			/* range and azimuth weights used in window function */

   int offs;				/* width and height of file segment*/
   int xw,yh;				/* width, height of processed region */
   int i,j,i1,j1;			/* loop counters */
   int ndim;				/* number of dimensions of FFT */
   int isign;				/* direction of FFT */     
   int nlc;				/* number of guides, number of low correlation pixels */
   int lc;				/* line counter */

   FILE *int_file, *sm_file;

   int_file = fopen(int_filename,"rb");
   if (int_file == NULL){fprintf(stderr,"cannot open interferogram file: %s\n",int_filename); exit(-1);}
   sm_file = fopen(sm_filename,"wb");
   if (sm_file == NULL){fprintf(stderr,"cannot create smoothed interferogram file: %s\n",sm_filename); exit(-1);}

   if (ymax > nlines-1){
     ymax = nlines-1; 
    fprintf(stderr,"WARNING: insufficient #lines in the file for given input range: ymax: %d\n",ymax);     
   }

   if (xmax > width-1) xmax=width-1;	/* check to see if xmax within bounds */
   xw=xmax-xmin+1;			/* width of array */
   yh=ymax-ymin+1;			/* height of array */ 
   offs=ymin;				/* first line of file to start reading/writing */
   fprintf(stdout,"array width, height, offset: %5d %5d %5d\n",xw,yh,offs);
 
   cmp = cmatrix(0, NFFT-1, -NFFT,width+NFFT);			/* add space around the arrays */ 
   sm = cmatrix(0,NFFT-1,-NFFT,width+NFFT); 

   tmp = (fcomplex **)malloc(sizeof(fcomplex *)*step); 
   tmp1 = (fcomplex **)malloc(sizeof(fcomplex *)*step); 
   if (tmp == NULL || tmp1==NULL){fprintf(stderr,"ERROR: failure to allocate space for circular buffer pointers\n"); exit(-1);}

   bufcz = (fcomplex *)malloc(sizeof(fcomplex)*width); 
   if(bufcz == NULL){fprintf(stderr,"ERROR: failure to allocate space for input line buffer\n"); exit(-1);}

   seg_fftb = (fcomplex *)malloc(sizeof(fcomplex)*NFFT*NFFT);
   if(seg_fftb == NULL){fprintf(stderr,"ERROR: failure to allocate space for FFT data\n"); exit(-1);}
   seg_fft = (fcomplex **)malloc(sizeof(fcomplex *)*NFFT); 
   if(seg_fft == NULL){fprintf(stderr,"ERROR: failure to allocate space for FFT data pointers\n"); exit(-1);}

   wfb = (double *)malloc(sizeof(double)*NFFT*NFFT);
   if (wfb == NULL){fprintf(stderr,"ERROR: weight memory allocation failure...\n"); exit(-1);}
   wf = (double **)malloc(sizeof(double *)*NFFT); 
   if (wf == NULL){fprintf(stderr,"ERROR: weight pointers memory allocation failure...\n"); exit(-1);}

   for(i=0; i < NFFT; i++) seg_fft[i] = seg_fftb  + i*NFFT;
   for(j=0; j < NFFT; j++) wf[j] = wfb + j*NFFT;

   for(j=0; j < width; j++){bufcz[j].re=0.; bufcz[j].im=0.;}

   for(i=0; i < NFFT; i++){				/* initialize circular data buffers */
     for(j= -NFFT; j < width+NFFT; j++){
       cmp[i][j].re = 0.0; cmp[i][j].im = 0.0;
       sm[i][j].re = 0.0; sm[i][j].im = 0.0;   
     }
   }
 
   for (i=0; i < NFFT; i++){
     for (j=0; j < NFFT; j++){
       azw = 1.0 - fabs(2.0*(double)(i-NFFT/2)/(NFFT+1));  
       rw  = 1.0 - fabs(2.0*(double)(j-NFFT/2)/(NFFT+1));  
       wf[i][j]=azw*rw/(double)(NFFT*NFFT);
#ifdef DEBUG
       fprintf(stderr,"i,j,wf: %5d %5d %12.4e\n",i,j,wf[i][j]);
#endif
     }
   }
 
   nfft[1] = NFFT;
   nfft[2] = nfft[1];
   nfft[0] = 0;
   ndim = 2;
   isign = 1;				/* initialize FFT parameter values, inverse FFT */

   
   fseek(int_file, offs*width*sizeof(fcomplex), SEEK_SET); 	/* seek offset to start line of interferogram */
   for (i=0; i < step; i++)fread((char *)cmp[i], sizeof(fcomplex), width, int_file);
   lc=0;

   for (i=0; i < yh; i += step){
     for(i1=step; i1 < NFFT; i1++){
       fread((char *)cmp[i1], sizeof(fcomplex), width, int_file);
       if (feof(int_file) != 0){				/* fill with zero if at end of file */
         for(j1= -NFFT; j1 < width+NFFT; j1++){cmp[i1][j1].re=0.0; cmp[i1][j1].im=0.0;}
       }
       for(j1= -NFFT; j1 < width+NFFT; j1++){
         sm[i1][j1].re=0.0; sm[i1][j1].im=0.0; 			/* clear out area for new sum */
       }     
     }
     if(i%(2*step) == 0)fprintf(stderr,"\rline: %5d",i);
  
     for (j=0; j < width; j += step){
       psd_wgt(cmp, seg_fft, alpha, j, i);	
       fourn((float *)seg_fft[0]-1,nfft,ndim,isign);	/* 2D inverse FFT of region, get back filtered fringes */

       for (i1=0; i1 < NFFT; i1++){			/* save filtered output values */
         for (j1=0; j1 < NFFT; j1++){
           if(cmp[i1][j+j1].re !=0.0){
             sm[i1][j+j1].re += wf[i1][j1]*seg_fft[i1][j1].re; 
             sm[i1][j+j1].im += wf[i1][j1]*seg_fft[i1][j1].im;
           }
           else{
             sm[i1][j+j1].re=0.0; 
             sm[i1][j+j1].im=0.0;
           }
         }
       }
     }         
     for (i1=0; i1 < step; i1++){	
       if (lc < yh)fwrite((char *)sm[i1], sizeof(fcomplex), width, sm_file); 
       lc++;
     }
     for (i1=0; i1 < step; i1++){tmp[i1] = cmp[i1]; tmp1[i1] = sm[i1];}		/* save pointers to lines just written out */
     for (i1=0; i1 < step; i1++){cmp[i1] = cmp[i1+step]; sm[i1] = sm[i1+step];}	/* shift the data just processed  */
     for (i1=0; i1 < step; i1++){cmp[step+i1] = tmp[i1]; sm[step+i1]=tmp1[i1];} /* copy pointers back */
   }
   
   for(i=lc; i < yh; i++){				/* write null lines of filtered complex data */	
     fwrite((char *)bufcz, sizeof(fcomplex), width, sm_file);
     lc++;
   } 

   fprintf(stdout,"\nnumber of lines written to file: %d\n",lc);
   free(wfb);
   free(wf);
   free(seg_fftb);
   free(seg_fft);
   free(bufcz);
   free(tmp);
   free(tmp1);
   // free_cmatrix doesn't work
   /*free_cmatrix(cmp,0, NFFT-1, -NFFT,width+NFFT);
   free_cmatrix(sm,0,NFFT-1,-NFFT,width+NFFT);*/
   fclose(int_file);
   fclose(sm_file);
} 
 
void psd_wgt(fcomplex **cmp, fcomplex **seg_fft, double alpha, int ix, int iy)
/* 
  subroutine to perform non-linear spectral filtering  17-Feb-97 clw
*/

{
  double psd,psd_sc;	/* power spectrum, scale factor */
  int i,j;		/* loop counters */
  int ndim,isign;	/* number of dimensions in fft */
  
  int ic;
 
  unsigned int nfft[3];
  ic = 0;
   
  ndim=2, isign = -1, nfft[1]=NFFT, nfft[2]=NFFT, nfft[0]=0; /* fft initialization */
  
  for (i=0; i < NFFT; i++){  					/* load up data array */
    for (j=ix; j < ix+NFFT; j++){
      seg_fft[i][j-ix].re = cmp[i][j].re;
      seg_fft[i][j-ix].im = cmp[i][j].im;
    }
  }

  fourn((float *)seg_fft[0]-1, nfft, ndim, isign);		/* 2D forward FFT of region */

  for (i=0; i < NFFT; i++){
    for (j=0; j < NFFT; j++){
      psd = seg_fft[i][j].re * seg_fft[i][j].re + seg_fft[i][j].im * seg_fft[i][j].im;
      psd_sc = pow(psd,alpha/2.);
      seg_fft[i][j].re *= psd_sc;
      seg_fft[i][j].im *= psd_sc;    
    }
  }
}


#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
void fourn(float data[], unsigned int nn[], int ndim, int isign)
{
	int idim;
	unsigned long i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
	unsigned long ibit,k1,k2,n,nprev,nrem,ntot;
	float tempi,tempr;
	double theta,wi,wpi,wpr,wr,wtemp;

	for (ntot=1,idim=1;idim<=ndim;idim++)
		ntot *= nn[idim];
	nprev=1;
	for (idim=ndim;idim>=1;idim--) {
		n=nn[idim];
		nrem=ntot/(n*nprev);
		ip1=nprev << 1;
		ip2=ip1*n;
		ip3=ip2*nrem;
		i2rev=1;
		for (i2=1;i2<=ip2;i2+=ip1) {
			if (i2 < i2rev) {
				for (i1=i2;i1<=i2+ip1-2;i1+=2) {
					for (i3=i1;i3<=ip3;i3+=ip2) {
						i3rev=i2rev+i3-i2;
						SWAP(data[i3],data[i3rev]);
						SWAP(data[i3+1],data[i3rev+1]);
					}
				}
			}
			ibit=ip2 >> 1;
			while (ibit >= ip1 && i2rev > ibit) {
				i2rev -= ibit;
				ibit >>= 1;
			}
			i2rev += ibit;
		}
		ifp1=ip1;
		while (ifp1 < ip2) {
			ifp2=ifp1 << 1;
			theta=isign*6.28318530717959/(ifp2/ip1);
			wtemp=sin(0.5*theta);
			wpr = -2.0*wtemp*wtemp;
			wpi=sin(theta);
			wr=1.0;
			wi=0.0;
			for (i3=1;i3<=ifp1;i3+=ip1) {
				for (i1=i3;i1<=i3+ip1-2;i1+=2) {
					for (i2=i1;i2<=ip3;i2+=ifp2) {
						k1=i2;
						k2=k1+ifp1;
						tempr=(float)wr*data[k2]-(float)wi*data[k2+1];
						tempi=(float)wr*data[k2+1]+(float)wi*data[k2];
						data[k2]=data[k1]-tempr;
						data[k2+1]=data[k1+1]-tempi;
						data[k1] += tempr;
						data[k1+1] += tempi;
					}
				}
				wr=(wtemp=wr)*wpr-wi*wpi+wr;
				wi=wi*wpr+wtemp*wpi+wi;
			}
			ifp1=ifp2;
		}
		nprev *= n;
	}
}

#undef SWAP

fcomplex **cmatrix(int nrl, int nrh, int ncl, int nch)
/* allocate a fcomplex matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	int i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	fcomplex **m;

	/* allocate pointers to rows */
	m=(fcomplex **)malloc((size_t)((nrow+NR_END)*sizeof(fcomplex*)));
	if (!m) nrerror("ERROR: allocation failure 1 in cmatrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(fcomplex *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(fcomplex)));
	if (!m[nrl]) nrerror("ERROR: allocation failure 2 in cmatrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

void free_cmatrix(fcomplex **m, int nrl, int nrh, int ncl, int nch)
/* free a float matrix allocated by matrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-NR_END));
  free((FREE_ARG) (m+nrl-NR_END));
}

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stdout,"Numerical Recipes run-time error...\n");
	fprintf(stdout,"%s\n",error_text);
	fprintf(stdout,"...now exiting to system...\n");
	exit(1);
}
