#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

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

#define NFFT 	32	/* size of FFT */
#define STEP	NFFT/2	/* stepsize in range and azimuth for filter */
#define ALPHA	0.5	/* default exponent for weighting of the spectrum */

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

#define NR_END 1
#define FREE_ARG char*

typedef struct{float re,im;} fcomplex;

void fourn(float *, unsigned int *, int ndim, int isign); 
void psd_wgt(fcomplex **cmp, fcomplex **seg_fft, double, int, int, int);

void start_timing();					/* timing routines */
void stop_timing();
	
unsigned int nfft[3];
int xmin=0;						/* window column minima */
int ymin=0;						/* window row minima */
int ymax, xmax;          /* interferogram width, window maxima */

fcomplex **cmatrix(int nrl, int nrh, int ncl, int nch);
void free_cmatrix(fcomplex **m, int nrl, int nrh, int ncl, int nch);
void nrerror(char error_text[]);
signed char IsFinite(double d);

int psfilt1(char *inputfile, char *outputfile, int width, double alpha, int fftw, int step)
{
   fcomplex *bufcz, **cmp;		/* interferogram line buffer, complex input data, row pointers */
   fcomplex **sm, **seg_fft, *seg_fftb;	/* smoothed interferogram, 2d fft of segment */
   fcomplex **tmp, **tmp1;		/* arrays of pointers for temp storage of line pointers in circular buffers */
   double **wf, *wfb;			/* 2-D weights */

   float *data;				/* pointer to floats for FFT, union with seg_fft */
   //double alpha;			/* exponent used to to determine spectal weighting function */
   double rw,azw;			/* range and azimuth weights used in window function */
 
   int nlines=0;			/* number of lines in the file */
   int offs;				/* width and height of file segment*/
   //int fftw;       /* fft window size*/
   //int step;				/* step size in range and azimuth for filtering of interferogram */
   int xw,yh;				/* width, height of processed region */
   int i,j,i1,j1;			/* loop counters */
   int ndim;				/* number of dimensions of FFT */
   int isign;				/* direction of FFT */     
   int nlc;				/* number of guides, number of low correlation pixels */
   int lc;				/* line counter */
    
   FILE *int_file, *sm_file;

   double psd,psd_sc;  /* power spectrum, scale factor */
   int ii, jj;
   fftwf_plan p_forward;
   fftwf_plan p_backward;

   int k;
   float sf; // scale factor for FFT, otherwise FFT will magnify the data by FFT length
             // usually the magnitude of the interferogram is very large, so the data are
             // multiplied by this factor before FFT, rather than after FFT in this program.
   float sf0; // an extra factor to scale the data
   sf0  = 1.0;

   
   fprintf(stderr,"*** Weighted power spectrum interferogram filter v1.0 clw 19-Feb-97 ***\n");
    if(0){
     //fprintf(stderr,"\nUsage: %s <interferogram> <smoothed interferogram> <width> [alpha] [fftw] [step] [xmin] [xmax] [ymin] [ymax]\n\n",argv[0]) ;
    
     fprintf(stderr,"input parameters: \n");
     fprintf(stderr,"  interferogram      complex interferogram image filename\n");
     fprintf(stderr,"  smoothed interf.   smoothed interferogram filename\n");
     fprintf(stderr,"  width              number of samples/row\n");
     fprintf(stderr,"  alpha              spectrum amplitude scale factor (default=.5)\n");  
     fprintf(stderr,"  fftw               fft window size in both range and azimuth directions \n");
     fprintf(stderr,"  step               moving step size in both range and azimuth directions (default = fftw/2)\n");
     fprintf(stderr,"  xmin               offset to starting range pixel (default = 0)\n");    
     fprintf(stderr,"  xmax               offset last range pixel (default = width-1)\n");    
     fprintf(stderr,"  ymin               offset to starting azimuth row (default = 0)\n");    
     fprintf(stderr,"  ymax               offset to last azimuth row (default = nlines-1)\n\n");    
     exit(-1);
   }

   start_timing();
   int_file = fopen(inputfile,"rb"); 
   if (int_file == NULL){fprintf(stderr,"cannot open interferogram file: %s\n",inputfile); exit(-1);}

   sm_file = fopen(outputfile,"wb"); 
   if (sm_file == NULL){fprintf(stderr,"cannot create smoothed interferogram file: %s\n",outputfile); exit(-1);}

   //sscanf(argv[3],"%d",&width);  
   xmax = width-1;	 
 
   fseeko(int_file, 0L, REL_EOF);		/* determine # lines in the file */
   nlines=(int)(ftello(int_file)/(width*2*sizeof(float)));
   fprintf(stderr,"#lines in the interferogram file: %d\n",nlines); 
   rewind(int_file);

   
   //alpha = ALPHA;
   //if(argc >= 4)sscanf(argv[4],"%lf",&alpha);
   fprintf(stdout,"spectrum weighting exponent: %8.4f\n",alpha);

   //fftw = NFFT;
   //if (argc >5)sscanf(argv[5],"%d",&fftw);
   fprintf(stdout,"FFT window size: %5d\n",fftw);

   sf = fftw * fftw * sf0;

   //step = fftw/2;
   //if (argc >6)sscanf(argv[6],"%d",&step);
   if (step <= 0 || step > fftw){
     fprintf(stdout,"WARNING: wrong step size: %5d, using %5d instead\n",step, fftw/2);
     step = fftw/2;
   }
   fprintf(stdout,"range and azimuth step size (pixels): %5d\n",step); 
   
   ymax=nlines-1;				/* default value of ymax */
   //if(argc > 7)sscanf(argv[7],"%d",&xmin);	/* window to process */
   //if(argc > 8)sscanf(argv[8],"%d",&xmax);
   //if(argc > 9)sscanf(argv[9],"%d",&ymin);
   //if(argc > 10)sscanf(argv[10],"%d",&ymax);
 
   if (ymax > nlines-1){
     ymax = nlines-1; 
    fprintf(stderr,"WARNING: insufficient #lines in the file for given input range: ymax: %d\n",ymax);     
   }

   if (xmax > width-1) xmax=width-1;	/* check to see if xmax within bounds */
   xw=xmax-xmin+1;			/* width of array */
   yh=ymax-ymin+1;			/* height of array */ 
   offs=ymin;				/* first line of file to start reading/writing */
   fprintf(stdout,"array width, height, offset: %5d %5d %5d\n",xw,yh,offs);
 
   cmp = cmatrix(0, fftw-1, -fftw,width+fftw);			/* add space around the arrays */ 
   sm = cmatrix(0,fftw-1,-fftw,width+fftw);

   tmp = (fcomplex **)malloc(sizeof(fcomplex *)*step);
   tmp1 = (fcomplex **)malloc(sizeof(fcomplex *)*step);
   if (tmp == NULL || tmp1==NULL){fprintf(stderr,"ERROR: failure to allocate space for circular buffer pointers\n"); exit(-1);}

   bufcz = (fcomplex *)malloc(sizeof(fcomplex)*width);
   if(bufcz == NULL){fprintf(stderr,"ERROR: failure to allocate space for input line buffer\n"); exit(-1);}

   seg_fftb = (fcomplex *)malloc(sizeof(fcomplex)*fftw*fftw);
   if(seg_fftb == NULL){fprintf(stderr,"ERROR: failure to allocate space for FFT data\n"); exit(-1);}
   seg_fft = (fcomplex **)malloc(sizeof(fcomplex *)*fftw);
   if(seg_fft == NULL){fprintf(stderr,"ERROR: failure to allocate space for FFT data pointers\n"); exit(-1);}

   wfb = (double *)malloc(sizeof(double)*fftw*fftw);
   if (wfb == NULL){fprintf(stderr,"ERROR: weight memory allocation failure...\n"); exit(-1);}
   wf = (double **)malloc(sizeof(double *)*fftw);
   if (wf == NULL){fprintf(stderr,"ERROR: weight pointers memory allocation failure...\n"); exit(-1);}

   for(i=0; i < fftw; i++) seg_fft[i] = seg_fftb  + i*fftw;
   for(j=0; j < fftw; j++) wf[j] = wfb + j*fftw;

   for(j=0; j < width; j++){bufcz[j].re=0.; bufcz[j].im=0.;}

   for(i=0; i < fftw; i++){				/* initialize circular data buffers */
     for(j= -fftw; j < width+fftw; j++){
       cmp[i][j].re = 0.0; cmp[i][j].im = 0.0;
       sm[i][j].re = 0.0; sm[i][j].im = 0.0;   
     }
   }
 
   for (i=0; i < fftw; i++){
     for (j=0; j < fftw; j++){
       azw = 1.0 - fabs(2.0*(double)(i-fftw/2)/(fftw+1));  
       rw  = 1.0 - fabs(2.0*(double)(j-fftw/2)/(fftw+1));  
       wf[i][j]=azw*rw/(double)(fftw*fftw);
#ifdef DEBUG
       fprintf(stderr,"i,j,wf: %5d %5d %12.4e\n",i,j,wf[i][j]);
#endif
     }
   }
 
   nfft[1] = fftw;
   nfft[2] = nfft[1];
   nfft[0] = 0;
   ndim = 2;
   isign = 1;				/* initialize FFT parameter values, inverse FFT */

   
   fseek(int_file, offs*width*sizeof(fcomplex), REL_BEGIN); 	/* seek offset to start line of interferogram */
   for (i=0; i < fftw - step; i++){
    fread((char *)cmp[i], sizeof(fcomplex), width, int_file);
    for(k = 0; k < width; k++){
      cmp[i][k].re /= sf;
      cmp[i][k].im /= sf;
    }
   }
   lc=0;

   p_forward  = fftwf_plan_dft_2d(fftw, fftw, (fftw_complex *)seg_fft[0], (fftw_complex *)seg_fft[0], FFTW_FORWARD, FFTW_MEASURE);
   p_backward = fftwf_plan_dft_2d(fftw, fftw, (fftw_complex *)seg_fft[0], (fftw_complex *)seg_fft[0], FFTW_BACKWARD, FFTW_MEASURE);

   for (i=0; i < yh; i += step){
     for(i1=fftw - step; i1 < fftw; i1++){
       fread((char *)cmp[i1], sizeof(fcomplex), width, int_file);
       for(k = 0; k < width; k++){
        cmp[i1][k].re /= sf;
        cmp[i1][k].im /= sf;
       }
       if (feof(int_file) != 0){				/* fill with zero if at end of file */
         for(j1= -fftw; j1 < width+fftw; j1++){cmp[i1][j1].re=0.0; cmp[i1][j1].im=0.0;}
       }
       for(j1= -fftw; j1 < width+fftw; j1++){
         sm[i1][j1].re=0.0; sm[i1][j1].im=0.0; 			/* clear out area for new sum */
       }     
     }
     if(i%(2*step) == 0)fprintf(stderr,"\rprogress: %3d%%", (int)(i*100/yh + 0.5));
  
     for (j=0; j < width; j += step){
       //psd_wgt(cmp, seg_fft, alpha, j, i, fftw);	
       ////////////////////////////////////////////////////////////////////////////////////////
       //replace function psd_wgt with the following to call FFTW, crl, 23-APR-2020
       for (ii=0; ii < fftw; ii++){            /* load up data array */
         for (jj=j; jj < j+fftw; jj++){
           seg_fft[ii][jj-j].re = cmp[ii][jj].re;
           seg_fft[ii][jj-j].im = cmp[ii][jj].im;
         }
       }

       //fourn((float *)seg_fft[0]-1, nfft, ndim, -1);   /* 2D forward FFT of region */
       fftwf_execute(p_forward);

       for (ii=0; ii < fftw; ii++){
         for (jj=0; jj < fftw; jj++){
           psd = seg_fft[ii][jj].re * seg_fft[ii][jj].re + seg_fft[ii][jj].im * seg_fft[ii][jj].im;
           psd_sc = pow(psd,alpha/2.);
           seg_fft[ii][jj].re *= psd_sc;
           seg_fft[ii][jj].im *= psd_sc;    
         }
       }
       /////////////////////////////////////////////////////////////////////////////////////////
       //fourn((float *)seg_fft[0]-1,nfft,ndim,isign);	/* 2D inverse FFT of region, get back filtered fringes */
       fftwf_execute(p_backward);

       for (i1=0; i1 < fftw; i1++){			/* save filtered output values */
         for (j1=0; j1 < fftw; j1++){
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
       if (lc < yh){
        for(k = 0; k < width; k++){
          if(!IsFinite(sm[i1][k].re))
            sm[i1][k].re = 0.0;
          if(!IsFinite(sm[i1][k].im))
            sm[i1][k].im = 0.0;
          if(!IsFinite(sqrt(sm[i1][k].re*sm[i1][k].re + sm[i1][k].im*sm[i1][k].im))){
            sm[i1][k].re = 0.0;
            sm[i1][k].im = 0.0;
          }
        }
        fwrite((char *)sm[i1], sizeof(fcomplex), width, sm_file); 
       }
       lc++;
     }
     for (i1=0; i1 < step; i1++){tmp[i1] = cmp[i1]; tmp1[i1] = sm[i1];}		/* save pointers to lines just written out */
     for (i1=0; i1 < fftw - step; i1++){cmp[i1] = cmp[i1+step]; sm[i1] = sm[i1+step];}	/* shift the data just processed  */
     for (i1=0; i1 < step; i1++){cmp[fftw - step+i1] = tmp[i1]; sm[fftw - step+i1]=tmp1[i1];} /* copy pointers back */
   }
   fprintf(stderr,"\rprogress: %3d%%", 100);
   
   for(i=lc; i < yh; i++){				/* write null lines of filtered complex data */	
     for(k = 0; k < width; k++){
      if(!IsFinite(bufcz[k].re))
        bufcz[k].re = 0.0;
      if(!IsFinite(bufcz[k].im))
        bufcz[k].im = 0.0;  
      if(!IsFinite(sqrt(bufcz[k].re*bufcz[k].re + bufcz[k].im*bufcz[k].im))){
        bufcz[k].re = 0.0;
        bufcz[k].im = 0.0;
      }
     }
     fwrite((char *)bufcz, sizeof(fcomplex), width, sm_file);
     lc++;
   } 

   fprintf(stdout,"\nnumber of lines written to file: %d\n",lc);
   stop_timing();

   fftwf_destroy_plan(p_forward);
   fftwf_destroy_plan(p_backward);

   free(bufcz);
   //free_cmatrix(cmp, 0, fftw-1, -fftw,width+fftw);
   //free_cmatrix(sm, 0,fftw-1,-fftw,width+fftw);
   free(seg_fft);
   free(seg_fftb);
   free(tmp);
   free(tmp1);
   free(wf);
   free(wfb);
   fclose(int_file); 
   fclose(sm_file);

   return(0);  
} 
 
void psd_wgt(fcomplex **cmp, fcomplex **seg_fft, double alpha, int ix, int iy, int fftw)
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
   
  ndim=2, isign = -1, nfft[1]=fftw, nfft[2]=fftw, nfft[0]=0; /* fft initialization */
  
  for (i=0; i < fftw; i++){  					/* load up data array */
    for (j=ix; j < ix+fftw; j++){
      seg_fft[i][j-ix].re = cmp[i][j].re;
      seg_fft[i][j-ix].im = cmp[i][j].im;
    }
  }

  fourn((float *)seg_fft[0]-1, nfft, ndim, isign);		/* 2D forward FFT of region */

  for (i=0; i < fftw; i++){
    for (j=0; j < fftw; j++){
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

/* function: IsFinite()
 * --------------------
 * This function takes a double and returns a nonzero value if 
 * the arguemnt is finite (not NaN and not infinite), and zero otherwise.
 * Different implementations are given here since not all machines have
 * these functions available.
 */
signed char IsFinite(double d){

  return(finite(d));
  /* return(isfinite(d)); */
  /* return(!(isnan(d) || isinf(d))); */
  /* return(TRUE) */
}

