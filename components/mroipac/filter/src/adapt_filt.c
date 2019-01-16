#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define PLUS 1
#define MINU 2
#define CHG  3
#define GUID 4
#define LSNR 8
#define VIST 16
#define BRPT 32
#define CUT  64
#define LAWN 128
#define TREE 128

#define PW   		4		/* window around peak to estimate fringe SNR */
#define NFFT 		32		/* size of FFT */
#define STEP		NFFT/2		/* stepsize in range and azimuth for filter */
#define FILT_WIDTH    	2.0		/* default filter width (pixels) */

#define REL_BEGIN 0	/* fseek relative to beginning of file */
#define REL_CUR   1	/* fseek relative to current position in the file */
#define REL_EOF   2	/* fseek relative to end of file */

#define SQR(a)    ( (a)*(a) )

#define PI	3.14159265359 
#define TWO_PI  6.28318530718
#define SQRT2   1.41421356237 	/* square root of 2 */
#define RTD	57.2957795131	/* radians to degrees */
#define DTR	.0174532925199	/* degrees to radians */
#define C	2.99792458e8

#define BETAP	  2.120		/* Kaiser param for smoothing filter   (-40 dB stopband ripple ) */

/*      Kaiser Window Parameter

 Beta    passband ripple   stopband ripple
*********************************************
   1.000      .86                -20
   2.120      .27                -30
   3.384      .0864              -40
   4.538      .0274              -50
   5.658      .00868             -60
   6.764      .00275             -70
   7.865      .000868            -80
   8.960      .000275            -90
*********************************************
*/

#define NR_END 1
#define FREE_ARG char*

typedef struct{float re,im;} fcomplex;

void fourn(float *, unsigned int *, int ndim, int isign); 
double bessi0(double x);				/* modified Bessel function of order 0 */
double frg_fft(fcomplex **cmp, fcomplex **seg_fft, int, int, int *, int *, int);
void lp2d(double **w, int nps, double bwx, double bwy);
void mfilt(fcomplex **seg_fft, double **win, int imx, int jmx, int nx, int ny);

void start_timing();					/* timing routines */
void stop_timing();
	
unsigned int nfft[3];
int xmin=0;						/* window column minima */
int ymin=0;						/* window row minima */
int width, ymax, xmax;					/* interferogram width, window maxima */

fcomplex **cmatrix(int nrl, int nrh, int ncl, int nch);
void free_cmatrix(fcomplex **m, int nrl, int nrh, int ncl, int nch);
void nrerror(char error_text[]);

int main(int argc, char **argv)
{
   fcomplex *bufcz, **cmp;		/* interferogram line buffer, complex input data, row pointers */
   fcomplex **sm, **seg_fft, *seg_fftb;	/* smoothed interferogram, 2d fft of segment */
   fcomplex **tmp;
   double **win, *winb, **wf, *wfb;

   double fraclc;			/* fraction of image which is low correlation, or guides */
   double low_snr_thr;			/* low fringe SNR threshold, used to set low SNR flag */
   double reg_snr;			/* fringe SNR */
   double bwx,bwy;			/* bandwidths in x and y for the fringe filter */
   double fltw;				/* filter width in fft bins */	
   float *data;				/* pointer to floats for FFT, union with seg_fft */
   double ssq,sq,sw;			/* squares of filter coefficients, sum of weights */

   int nlines=0;			/* number of lines in the file */
   int imx,jmx;				/* indices of peak in the PSD */
   int offs;				/* width and height of file segment*/
   int step;				/* step size in x and y for filtering of interferogram */
   int xw,yh;				/* width, height of processed region */
   int i,j,i1,j1;			/* loop counters */
   int ndim;				/* number of dimensions of FFT */
   int isign;				/* direction of FFT */     
   int nlc;				/* number of guides, number of low correlation pixels */
   int lc;				/* line counter */
    
   unsigned char *bufz;			/* flag array buffer, flag array pointers, buffer with zeroes */

   FILE *int_file, *sm_file, *filtf;
   
    fprintf(stdout,"*** adaptive smoothing of interferogram v3.0 clw ***\n");
  if(argc < 4){
     fprintf(stderr,"\nusage: %s <interferogram> <smoothed interferogram> <width> [low_SNR_thr] [filt_width] [xmin] [xmax] [ymin] [ymax]\n\n",argv[0]) ;
    
     fprintf(stderr,"input parameters: \n");
     fprintf(stderr,"  interferogram      complex interferogram image filename\n");
     fprintf(stderr,"  smoothed interf.   smoothed interferogram filename\n");
     fprintf(stderr,"  width              number of samples/row\n");
     fprintf(stderr,"  low_snr_thr        low SNR threshold (default = .25);\n");
     fprintf(stderr,"  filt_width         filter width in pixels (default = 2.0)\n");  
     fprintf(stderr,"  xmin               offset to starting range pixel(default = 0)\n");    
     fprintf(stderr,"  xmax               offset last range pixel (default = width-1)\n");    
     fprintf(stderr,"  ymin               offset to starting azimuth row (default = 0)\n");    
     fprintf(stderr,"  ymax               offset to last azimuth row (default = nlines-1)\n\n");    
     exit(-1);
   }

   int_file = fopen(argv[1],"r"); 
   if (int_file == NULL){fprintf(stderr,"cannot open interferogram file: %s\n",argv[1]); exit(-1);}

   sm_file = fopen(argv[2],"w"); 
   if (sm_file == NULL){fprintf(stderr,"cannot create smoothed interferogram file: %s\n",argv[2]); exit(-1);}

   filtf = fopen("adapt_filt.dat","w"); 
   if (filtf == NULL){fprintf(stderr,"cannot create filter coefficient file: adapt_filt.dat\n"); exit(-1);}

   sscanf(argv[3],"%d",&width);  
   xmax=width-1;	 
 
   fseek(int_file, 0L, REL_EOF);		/* determine # lines in the file */
   nlines=(int)ftell(int_file)/(width*2*sizeof(float));
   fprintf(stderr,"#lines in the interferogram file: %d\n",nlines); 
   rewind(int_file);

   low_snr_thr = .25;
   step = STEP;
   fltw=FILT_WIDTH;
   if(argc > 4)sscanf(argv[4],"%lf",&low_snr_thr); 
   if(argc > 5)sscanf(argv[5],"%lf",&fltw);
   fprintf(stdout,"low SNR threshold: %8.4f\n",low_snr_thr);
   fprintf(stdout,"bandpass filter width (pixels):  %8.4f\n",fltw);
   fprintf(stdout,"range and azimuth step size (pixels): %5d\n",step);

   ymax=nlines-1;				/* default value of ymax */
   if(argc > 6)sscanf(argv[6],"%d",&xmin);	/* window to process */
   if(argc > 7)sscanf(argv[7],"%d",&xmax);
   if(argc > 8)sscanf(argv[8],"%d",&ymin);
   if(argc > 9)sscanf(argv[9],"%d",&ymax);
 
   if (ymax > nlines-1){
     ymax = nlines-1; 
    fprintf(stderr,"WARNING: insufficient #lines in the file for given input range: ymax: %d\n",ymax);     
   }

   if (xmax > width-1) xmax=width-1;	/* check to see if xmax within bounds */
   xw=xmax-xmin+1;			/* width of array */
   yh=ymax-ymin+1;			/* height of array */ 
   offs=ymin;				/* first line of file to start reading/writing */
   fprintf(stdout,"array width, height, offset: %5d %5d %5d\n",xw,yh,offs);
 

   bwx= TWO_PI*fltw/NFFT;  
   bwy= TWO_PI*fltw/NFFT;
   fprintf(stdout,"filter bandwidth (radians): %10.3lf\n",bwx);

/******************* allocating memory *******************/

   start_timing();

   cmp = cmatrix(0, NFFT-1, -NFFT,width+NFFT);			/* add space around the arrays */ 
   sm = cmatrix(0,step-1,-NFFT,width+NFFT);
   fprintf(stderr,"allocating more memory\n");
   tmp = (fcomplex **)malloc(sizeof(fcomplex *)*step);
   if (tmp == NULL){fprintf(stderr,"ERROR: failure to allocate space for tmp line buffers\n"); exit(-1);}

   bufcz = (fcomplex *)malloc(sizeof(fcomplex)*width);
   if(bufcz == NULL){fprintf(stderr,"ERROR: failure to allocate space for input line buffer\n"); exit(-1);}

   bufz = (unsigned char *)malloc(width);
   if(bufz == NULL){fprintf(stderr,"ERROR: failure to allocate space for null output line\n"); exit(-1);}

   seg_fftb = (fcomplex *)malloc(sizeof(fcomplex)*NFFT*NFFT);
   if(seg_fftb == NULL){fprintf(stderr,"ERROR: failure to allocate space for complex data\n"); exit(-1);}

   seg_fft = (fcomplex **)malloc(sizeof(fcomplex *)*NFFT);
   if(seg_fft == NULL){fprintf(stderr,"ERROR: failure to allocate space for complex data pointers\n"); exit(-1);}
 
   win = (double **)malloc(sizeof(double *)*NFFT);
   if (win == NULL){fprintf(stderr,"ERROR: window pointers memory allocation failure...\n"); exit(-1);}

   wf = (double **)malloc(sizeof(double *)*NFFT);
   if (wf == NULL){fprintf(stderr,"ERROR: filter pointers memory allocation failure...\n"); exit(-1);}

   winb = (double *)malloc(sizeof(double)*NFFT*NFFT);
   if (winb == NULL){fprintf(stderr,"ERROR: window memory allocation failure...\n"); exit(-1);}

   wfb = (double *)malloc(sizeof(double)*NFFT*NFFT);
   if (wfb == NULL){fprintf(stderr,"ERROR: filter memory allocation failure...\n"); exit(-1);}

   for(i=0; i < NFFT; i++)seg_fft[i] = seg_fftb  + i*NFFT;
   for(j=0; j < NFFT; j++)win[j] = winb + j*NFFT;
   for(j=0; j < NFFT; j++)wf[j] = wfb + j*NFFT;

   for(j=0; j < width; j++)bufz[j]=0;	
   for(j=0; j < width; j++){bufcz[j].re=0.; bufcz[j].im=0.;}

   for(i=0; i < NFFT; i++){
     for(j= -NFFT; j < width+NFFT; j++){
       cmp[i][j].re=0.0; cmp[i][j].im=0.0;
     }
   }
   for(i=0; i < step; i++){
     for(j= -NFFT; j < width+NFFT; j++){
       sm[i][j].re=0.0; sm[i][j].im=0.0;
     }
   }

   nfft[1]=NFFT;
   nfft[2]=nfft[1];
   nfft[0]=0;
   ndim=2;
   isign= (-1);				/* initialize FFT parameter values, inverse FFT */
   data=(float *)&seg_fft[0][0];	/* let data be an array of floats in a union with the fcomplex data */
   data--;				/* decrement addresses so that indices start at 1 */

   lp2d(wf,NFFT,bwx,bwy);  		/* lowpass Kaiser window SINC filter, unit amplitude over passband */


   for (i=0; i < NFFT; i++){			
     for (j=0; j < NFFT; j++){
	seg_fft[i][j].re  = (float)wf[i][j];
	seg_fft[i][j].im = 0.0;
     }
   }
   ssq = 0.0;
   for (i=0; i < NFFT; i++){			
     for (j=0; j < NFFT; j++){
       ssq += SQR(seg_fft[i][j].re);
     }
   }
   fprintf(stderr,"sum of squared filter coefficients (frequency domain): %12.5e\n",ssq);

   fwrite((char *)&seg_fft[0][0],sizeof(fcomplex),NFFT*NFFT,filtf);
   fourn(data,nfft,ndim,1);
   fwrite((char *)&seg_fft[0][0],sizeof(fcomplex),NFFT*NFFT,filtf);
   fclose(filtf);

   ssq = 0.0; sw = 0.0;
   fprintf(stderr,"\n   i      j    Re(w[i][j])\n");
   fprintf(stderr,"********************************************\n");
   for (i=0; i < NFFT; i++){			
     for (j=0; j < NFFT; j++){
       if ((i <= fltw || i >= NFFT-fltw) && (j <= fltw || j >= NFFT-fltw)) 
         fprintf(stderr,"%4d   %4d    %10.6f\n",i,j,seg_fft[i][j].re); 
       ssq += SQR(seg_fft[i][j].re);
       sw += seg_fft[i][j].re;
     }
   }
   fprintf(stderr,"\nsum of filter coefficients squared (time domain): %12.6e\n",ssq);
   fprintf(stderr,"sum of filter coefficients (time domain): %10.6f\n",sw);

   for (i=0; i < NFFT; i++){
     for (j=0; j < NFFT; j++)wf[i][j] /= sw; 
   }

/****************  filter interferogram *******************/
   
   fseek(int_file, offs*width*sizeof(fcomplex), REL_BEGIN); 	/* seek offset to start line of interferogram */
   for (i=NFFT/2-step/2; i < NFFT-step; i++)fread((char *)cmp[i], sizeof(fcomplex), width, int_file);
 
   nlc=0;							/* initialize counter for low fringe SNR  and line counter*/
   lc=0;

   for (i=0; i < yh; i += step){
     for(i1=0; i1 < step; i1++){
       fread((char *)cmp[NFFT-step+i1], sizeof(fcomplex), width, int_file);
       if (feof(int_file) != 0){						/* fill with zero if at end of file */
         for(j1=0; j1 < width; j1++){cmp[NFFT-step+i1][j1].re=0.0; cmp[NFFT-step+i1][j1].re=0.0;}
       }
     }
  
     for (j=0; j < width; j += step){

       reg_snr = frg_fft(cmp, seg_fft, j, i, &imx, &jmx, PW);	/* calculate fringe SNR */

       if ((i-NFFT/2)%(16*step) == 0){
         if (j == 0)
           fprintf(stderr,"\n   x     y     fx   fy     SNR\n**************************************\n");
         if (j%128 == 0)fprintf(stderr,"%5d %5d %4d %4d  %8.3f\n",i,j,imx,jmx,reg_snr );
       }

       if (reg_snr <= low_snr_thr){			/* low regional fringe SNR? */
         nlc += SQR(step);				/* increment LSNR counter */
         imx=0; jmx=0;					/* best guess for fringe spectral peak is at DC */
       }
 
       mfilt(seg_fft, wf, imx, jmx, NFFT, NFFT);	/* multiply by the window */  
       fourn(data,nfft,ndim,isign);			/* 2D inverse FFT of region, get back filtered fringes */
       for (i1=0; i1 < step; i1++){			/* save filtered output values */
         for (j1=0; j1 < step; j1++){
	   if(cmp[i1+step/2][j1+j-step/2].re != 0.0){
              sm[i1][j1+j-step/2] = seg_fft[NFFT/2-step/2+i1][NFFT/2-step/2+j1];
           }
           else{
             sm[i1][j1+j-step/2].re=0.0; sm[i1][j1+j-step/2].im=0.0;
           }
         }
       }
     }         
     for (i1=0; i1 < step; i1++){	
       if (lc < yh)fwrite((char *)sm[i1], sizeof(fcomplex), width, sm_file); 
       lc++;
     }
     for (i1=0; i1 < step; i1++)tmp[i1]=cmp[i1];			/* rotate circular buffer */
     for (i1=0; i1 < NFFT-step; i1++)cmp[i1]=cmp[i1+step];
     for (i1=0; i1 < step; i1++)cmp[NFFT-step+i1]=tmp[i1];	
   }
   
   for(i=lc; i < yh; i++){				/* write null lines of filtered complex data */	
     fwrite((char *)bufcz, sizeof(fcomplex), width, sm_file);
     lc++;
   } 

   fraclc= (double)nlc/(xw*yh);
   fprintf(stdout,"\nnumber of low SNR points %d\n",nlc);
   fprintf(stdout,"fraction low SNR points: %8.5f\n",fraclc);
   fprintf(stdout,"number of lines written to file: %d\n",lc);
   stop_timing();
   return(0);  
} 
 
double frg_fft(fcomplex **cmp, fcomplex **seg_fft, int ix, int iy, int *im, int *jm , int pw)
/* 
  subroutine to calculate correlation coefficient using the fft  5-aug-93 clw
*/

{
  double ai,ap,an;	/* sum of intensities of image, peak powers, noise power*/
  double amx;		/* maximum PSD value */
  double frv;		/* fringe visibility*/
  static double psd[NFFT][NFFT];	/* power spectral density array */
  float *dt;

  int i,j;		/* loop counters */
  int it,jt;		/* actual index in spectrum */
  int ndim,isign;	/* number of dimensions in fft */
  int wsz,psz;
  int ic;

  unsigned int nfft[3];
    
  dt=(float *)&seg_fft[0][0];	
  
  ndim=2, isign=1, nfft[1]=NFFT, nfft[2]=NFFT, nfft[0]=0; /* fft initialization */
  ai=0., ap=0;		/* image, peak power sum initialization*/
  amx=0.;		/* peak value of PSD */
  *im=0, *jm=0;
  wsz=NFFT*NFFT;
  psz=pw*pw;
  ic=0;
  
  for (i=0; i < NFFT; i++){  /* load up data array */
    for (j=ix-NFFT/2; j < ix+NFFT/2; j++){
      dt[ic++]=cmp[i][j].re; dt[ic++]=cmp[i][j].im;
    }
  }

  fourn(dt-1, nfft, ndim, isign);		/* 2D forward FFT of region */

   for (i=0; i < NFFT; i++){
    for (j=0; j < NFFT; j++){
      psd[i][j] = seg_fft[i][j].re * seg_fft[i][j].re + seg_fft[i][j].im * seg_fft[i][j].im;
      ai += psd[i][j];
      if(psd[i][j] > amx){	
        amx = psd[i][j];
	*im=i;
	*jm=j;
      }      
    }
  }

  for (i = 0; i < pw; i++){
    for (j = 0; j < pw; j++){
      it = (i + *im - pw/2 + NFFT)%NFFT;
      jt = (j + *jm - pw/2 + NFFT)%NFFT;
      ap += psd[it][jt];
    }
  }   

  an = (ai-ap)*wsz/(wsz-psz);
  if (an != 0.0) frv = (ap - an*psz/wsz)/an;
  else frv=0.0;
  return(frv);
}

void lp2d(double **w, int nps, double bwx, double bwy)
/* 
	2-D low-pass filter, Kaiser window of SINC function.
	Bandwidths are in the range 0 to TWO_PI, specified in the x and y dimensions.
	The number of points in the filter windows for x, and y are identical and
	given by nps.

	26-april-94 clw
*/
{
  int i,j;
  double wx,wy;
  double cvx,cvy;
  double *fx, *fy;

  fx = (double *)malloc(sizeof(double)*nps);
  if (fx == NULL){
    fprintf(stderr,"ERROR: unable to allocate space for %d x filter coefficients\n",nps);
    exit (-1);
  }

  fy = (double *)malloc(sizeof(double)*nps);
  if (fy == NULL){
    fprintf(stderr,"ERROR: unable to allocate space for %d y filter coefficients\n",nps);
    exit (-1);
  }
  
  fprintf(stderr,"\n  i        window        filter coeff.\n");
  fprintf(stderr,"*******************************************\n");  
  wx=1.0;
  fx[0]=bwx/(2.*PI);
  fprintf(stderr,"%4d    %12.5le    %12.5le\n",0,wx,fx[0]);
  for (i=1; i <= nps/2; i++){
    wx = bessi0(BETAP*sqrt(1.0-SQR(2.0*i/nps)))/bessi0(BETAP); 
    fx[i] = wx*sin(bwx/2.*(double)i)/((double)i*PI); 
    fx[nps-i]=fx[i];
    fprintf(stderr,"%4d    %12.5le    %12.5le\n",i,wx,fx[i]);
  } 

  fy[0]=bwy/(2.*PI);
  for (i=1; i <= nps/2; i++){						/* i < nps/2 */
    wy = bessi0(BETAP*sqrt(1.0-SQR(2.0*i/nps)))/bessi0(BETAP); 
    fy[i] = wy*sin(bwy/2.*(double)i)/((double)i*PI); 
    fy[nps-i]=fy[i];
/*    fprintf(stderr,"i,wy,fy[i]: %4d  %10.5f  %10.5f\n",i,wy,fy[i]); */
  } 

  for (i=0; i < nps; i++){
    for (j=0; j < nps; j++){
      w[i][j] = fx[i]*fy[j];
    }
  }

  free(fx);
  free(fy);
}

 
void mfilt(fcomplex **seg_fft, double **wf, int imx, int jmx, int nx, int ny)
{
  int i,j;
  int it,jt;


  for(i=0; i < ny; i++){
    for (j=0; j < nx; j++){
      it=(i-imx+ny)%ny;		/* shift filter coefficients */
      jt=(j-jmx+nx)%nx;
      seg_fft[i][j].re = (float)(seg_fft[i][j].re * wf[it][jt]);
      seg_fft[i][j].im = (float)(seg_fft[i][j].im * wf[it][jt]);
    }
  }
}

double bessi0(double x)
{
	double ax,ans;
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

#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <limits.h>

struct tms buffer;
int user_time, system_time, start_time;

void start_timing()
{
  start_time = (int) times(&buffer);
  user_time = (int) buffer.tms_utime;
  system_time = (int) buffer.tms_stime;
}

void stop_timing()
{
  int  end_time,elapsed_time;
  int clk_tck;

  clk_tck = (int)sysconf(_SC_CLK_TCK);

  end_time = (int) times(&buffer);
  user_time = (int) (buffer.tms_utime - user_time);
  system_time = (int) (buffer.tms_stime - system_time);
  elapsed_time = (end_time - start_time);

  fprintf(stdout,"\n\nuser time    (s):  %10.3f\n", (double)user_time/clk_tck);
  fprintf(stdout,"system time  (s):  %10.3f\n", (double)system_time/clk_tck); 
  fprintf(stdout,"elapsed time (s):  %10.3f\n\n", (double) elapsed_time/clk_tck);
}

