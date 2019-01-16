#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "defines.h"

typedef struct {
  int x,y;} Xpoint;

int ii[2][MAX_CRAB], jj[2][MAX_CRAB];	/* arrays for keeping locations of points for crabgrass */
int nn[2];				/* array containing lengths of the ping-pong lists */
int npu;				/* number of points unwrapped */
Xpoint bridge[2];			/* bridge for phase unwrapping */

void crab_step(unsigned char **, float **,  int, int, int) ;
void dmphsm(char*, char*, int*, int*, int*, int*);

unsigned char *fbf;

/**
 * Phase unwrapping grass
 *
 * @param intFilename interferogram filename
 * @param flagFilename phase unwrapping flag filename
 * @param unwFilename unwrapped phase output filename
 * @param width number of samples per row
 * @param start starting line (default=1)
 * @param xmin starting range pixel offset
 * @param xmax last range pixel offset
 * @param ymin starting azimuth row offset
 * @param ymax last azimuth row offset
 * @param xinit starting range pixel for unwrapping
 * @param yinit starint row for unwrapping
 */
int
grass(char *intFilename, char *flagFilename, char *unwFilename, int width, int start, int xmin, int xmax, int ymin, int ymax, int xinit, int yinit)
{
    double frac;		/* fraction of image which was unwrapped */
    double p_min,p_max;		/* minimum and maximum phase values */	
    float *buf, *buf_out;	/* single row complex data, output line */
    float **phase, *ph;		/* phase array for unwrapping */
    int nlines;			/* number of lines in the file */
    int xw,yh;			/* width, height of processed region */
    int offs;			/* offset number of lines to read from start of file*/
    int i,j;			/* loop counters */
    int m;			/* current ping-pong list */
    int pp=0;			/* ping-pong counter */
    int ppd;			/* ping-pong iteration for display */
    int ww, wh, cv_width, cv_height;  
    unsigned char **gzw, *fbf;	/* set of pointers to the rows of flag array */
    FILE *int_file, *flag_file, *unw_file;

  if (xmax <= 0) {
     xmax=width-1;				 	/* default value of xmax */
  } else if (xmax > width-1) {
      xmax=width-1;	/* check that xmax within bounds */
  }
 

  int_file = fopen(intFilename,"r"); 
  if (int_file == NULL){
      fprintf(stderr, "interferogram file does not exist!\n");
      exit(-1);
  }

  flag_file = fopen(flagFilename,"r+"); 
  if (flag_file == NULL){
      fprintf(stderr, "flag file does not exist!\n");
      exit(-1);
  }

  unw_file = fopen(unwFilename,"w"); 
  if (unw_file == NULL){
      fprintf(stderr, "cannot create magnitude/upwrapped phase file!\n");
      exit(-1);
  }

  fseek(int_file, 0L, REL_EOF);		/* determine # lines in the file */
  nlines=(int)ftell(int_file)/(width*8);
  fprintf(stderr,"# interferogram lines: %d\n",nlines); 
  rewind(int_file);

  if (ymax <= 0) {
    ymax=nlines-start;				/* default value of ymax */
  } else if (ymax > nlines-start) {
    ymax = nlines-start;
    fprintf(stderr,"insufficient #lines in the file, ymax: %d\n",ymax);
  }


  xw=xmax-xmin+1;	/* width of each line to process */
  yh=ymax-ymin+1;	/* height of array */
  offs=start+ymin-1;	/* first line of file to start reading/writing */
  fprintf(stderr,"array width,height (x,y), starting line:    %6d %6d %6d \n",xw,yh,offs);

  if (xinit < 0 ) {
    xinit = xmin+xw/2;	/* default position to start phase unwrapping */
  }
  if (yinit < 0) {
    yinit = ymin+yh/2;	/* default position to start phase unwrapping */
  }
// Initialize seed location here if xinit and yinit were passed in 
  fprintf(stderr,"initial seed location (x,y): %6d %6d \n",xinit,yinit);

/******************* Allocate space *********************/  

  ph = (float *) malloc(sizeof(float)*width*yh);
  if(ph == (float *) NULL) {
    fprintf(stderr,"failure to allocate space for phase array\n");
    exit(-1) ;
  }

  buf = (float *)malloc(2*sizeof(float)*width);
  if(buf == (float *) NULL) {
    fprintf(stderr,"failure to allocate space for input line buffer\n");
    exit(-1) ;
  }

  buf_out = (float *) malloc(2*sizeof(float)*width);
  if(buf_out == (float *) NULL) {
    fprintf(stderr,"failure to allocate space output line buffer\n");
    exit(-1) ;
  }

  fbf = (unsigned char *) malloc(sizeof(unsigned char)*width*yh);
  if(fbf == (unsigned char *) NULL) {
    fprintf(stderr,"failure to allocate space for flag array\n");
    exit(-1) ;
  }

  gzw = (unsigned char **)malloc(sizeof(unsigned char *) * yh); /* allocate flag pointers */
  phase = (float **)malloc(sizeof(float *) * yh);		/* allocate  phase pointers */

/**************** Read interferogram ******************/
      
  fprintf(stderr,"initializing phase array values...\n");

  for (i=0; i < yh; i++){				/* clear phase array */ 
    for (j=0; j < width; j++){
      ph[i*width+j]=0.;
    }
  }

  fprintf(stdout,"reading interferogram...\n");
  fseek(int_file, offs*width*2*sizeof(float), REL_BEGIN); 	/*seek starting line */
  
  for(i=0; i < yh; i++) {
    fread((char *)buf, sizeof(float), 2*width, int_file);	/* read next line */
    if(i%100 == 0){
      fprintf(stdout,"\rinterferogram input line %d", i);
      fflush(stdout);
    }
    for(j=0; j < width; j++){
      if((buf[2*j]==0.) && (buf[2*j+1]==0.)) ph[i*width+j]=0.;  /* phase undefined */
      else ph[i*width+j] = atan2((double)buf[2*j+1],(double)buf[2*j]);
    }
  }

  for (i=0; i < yh; i++){				/* set-up pointers for phase data */
    phase[i] = (float *)(ph + i*width + xmin);
  }

/**************** Read in flag data *******************/


  fseek(flag_file, offs*width*sizeof(unsigned char), REL_BEGIN); 

  fprintf(stderr,"\nreading flag file...\n");
  fread((char *)fbf, sizeof(unsigned char), width*yh, flag_file); 
    
  for (i=0; i< yh; i++){		 
      gzw[i] = (unsigned char *)(fbf + i*width + xmin);
  }

  cv_height=yh;
  cv_width=width;
  ww=Min(WIN_WIDTH_MAX,cv_width+SCROLL_WIDTH);
  wh=Min(WIN_HEIGHT_MAX,cv_height+SCROLL_WIDTH);

/**************** Initialize crabgrass *****************/
  while((gzw[yinit][xinit]&CUT) != 0) {		/*initial point cannot be on a cut */
    xinit++;
    yinit++;
  }

  fprintf(stderr,"actual seed location (x,y): %d  %d\n",xinit,yinit);

  ii[0][0]=yinit;		/* initialize list with the seed */
  jj[0][0]=xinit;
  nn[0]=1;			/* initial list length */
  npu=0;			/* number of pixels unwrapped */
  m=0;				/* current ping-pong list */
  /*fprintf(stderr,"enter interation number for flag display: ");
  scanf("%d",&ppd);*/
  ppd = 10000;
/**************** Grow crabgrass ***********************/

  while(nn[m] != 0) {		/* continue as long as list not zero length */
    crab_step(gzw, phase, yh,xw, m); 
    m=1-m;			/* ping-pong */
    if (pp%50 == 0){
      fprintf(stderr,"\rping-pong interation, list size %d %d", pp, nn[m]);
     }
    if(pp == ppd){
/*      dmphsm(argv[1], (char *)fbf, &cv_width, &cv_height, &ww, &wh); 
      fprintf(stderr,"\nenter next interation number for flag display: ");
       scanf("%d",&ppd);*/
    }
    pp++;			/* increment ping-pong step counter */
  }

/*  dmphsm(argv[1], (char *)fbf, &cv_width, &cv_height, &ww, &wh); */
  fprintf(stderr,"\ntotal ping-pong interations %d\n", pp);
  frac = npu/(float)(xw*yh);
  fprintf(stderr,"fraction of the image unwrapped: %8.5f\n",frac);
  p_min=p_max=0.0;

  for(i=0; i < yh; i++){	/* determine min and max phase values */
    for(j=0; j < xw; j++) {
      if(phase[i][j] < p_min) p_min=phase[i][j];
      if(phase[i][j] > p_max) p_max=phase[i][j];
    }
  }

  printf("minimum phase, maximum phase:  %12.3f   %12.3f\n",p_min,p_max);
  printf("phase difference:              %12.3f \n",p_max-p_min);
       
/************** write out unwrapped phase **************/
  for (i=0; i < width; i++) {		/* generate NULL line */
    buf[i]=0.;
    buf[i+width]=0.;
  }

  fprintf(stderr,"writing output file...\n");

  for (i=0; i< ymin; i++){		/* clear out ymin lines */
    fwrite((char *)buf, sizeof(float), 2*width, unw_file); 
  }

  fseek(int_file, offs*width*2*sizeof(float), REL_BEGIN); 	/*seek starting line */

  for (i=0; i < yh; i++){
    fread((char *)buf, sizeof(float), 2*width, int_file);	/* read next line */
    if (i%100 == 0) fprintf(stderr,"\routput line: %d", i);


    for (j=0; j< width; j++){
      if ((j >= xmin) && (j <= xmax) && ((gzw[i][j-xmin]&LAWN) != 0)){	/* check if on the LAWN too */
        buf_out[j] = hypot((double)buf[2*j],(double)buf[2*j+1]);
        buf_out[j+width] = phase[i][j-xmin];	/* take into account pointer offsets */
      }
      else {
        buf_out[j]=0.;
        buf_out[j+width]=0.;
      }
    }   
    fwrite((char *)buf_out,sizeof(float),2*width, unw_file);


  } 
  fprintf(stdout,"\nwriting flag file...\n");
  fseek(flag_file, offs*width*sizeof(unsigned char), REL_BEGIN); 	/*seek starting line */
  fwrite((char *)fbf, sizeof(unsigned char), yh*width, flag_file); 

  fclose(int_file);
  fclose(flag_file);
  fclose(unw_file);

  return 0;
}

void crab_step(unsigned char **gzw, float **phase, int data_w, int data_h, int m)
{
  int i,j,k,l,i1,j1;
  double u;
  static int dir_x[]={ 0, 0,-1, 1};   
  static int dir_y[]={-1, 1, 0, 0};	 
  
  if(nn[m]==0) return;  /* if list zero length */
  nn[1-m]=0;		/* initialize new list length */

  for(k=0; k < nn[m]; k++) {	/* go through the list, growing around each pixel if possible */
    i=ii[m][k]; 
    j=jj[m][k];

/*   if( (gzw[i][j] & (CUT)) == 0) {	*/ /* CUT?, don't grow cuts*/

   if( (gzw[i][j] & (CUT | LSNR) ) == 0) {	 /*CUT or LSNR?,  don't grow cuts, or LSNR regions */

      u=(double)phase[i][j];

      for(l=0; l< 4; l++) {	/* check neighbors */
	i1=i+dir_x[l]; 
        j1=j+dir_y[l];
	if((i1 < 0)||(i1 >= data_w) || (j1<0) || (j1 >= data_h)) continue; /* check boundries */
	if((gzw[i1][j1] & LAWN) != 0) continue;	/* check if already grown */
	npu++;						/* increment number of pixels grown */
	phase[i1][j1] += TWO_PI*nint((u-(double)phase[i1][j1])/TWO_PI); /* unwrap the phase */
	gzw[i1][j1] |=LAWN;			/* set pixel flag as grown */

 	if(nn[1-m] < MAX_CRAB) { 
	  ii[1-m][nn[1-m]]=i1; 			/* place this pixel onto the new list */
          jj[1-m][nn[1-m]]=j1; 
          nn[1-m]++;				/* increment the length counter of the new list */
        }
	else {
          fprintf(stderr,"warning: subroutine crab_step, crab_grass table over flow\n"); 
          return;		/* grow the new list */
        }

      }
    }
  }
}



