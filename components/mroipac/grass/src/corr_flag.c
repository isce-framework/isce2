#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "defines.h"

/**
 * Correlation threshold for phase unwrapping.
 *
 * @param corFilename interferometric correlation file
 * @param flagFilename phase unwrappign flag filename
 * @param width number of samples per row
 * @param thr correlation threshold
 * @param start starting line (default=1)
 * @param xmin starting range pixel offset
 * @param xmax last range pixel offset
 * @param ymin starting azimuth row offset
 * @param ymax last azimuth row offset
 */
int
corr_flag(char *corFilename, char *flagFilename, int width, double thr, int start, int xmin, int xmax, int ymin, int ymax, int cbands)
{
    float **cc, *ccb;		/* correlation data */

    int nlines;			/* number of lines in the file */
    int xw,yh;			/* width, height of processed region */
    int offs;			/* offset number of lines to read from start of file*/
    int i,j;

    unsigned char *fbf;		/* flag array */
    unsigned char **fb;		/* set of pointers to the rows of flag array */
    unsigned char *bz;
    FILE *flag_file, *c_file;
   
  c_file = fopen(corFilename,"r"); 
  if (c_file == NULL){
    fprintf(stderr, "cannot open correlation file!\n");
    exit(-1);
  }

  if (xmax <= 0) {
     xmax=width-1;				 	/* default value of xmax */
  }
  if (xmax > width-1) xmax=width-1;	/* check to see if xmax within bounds */
  fprintf(stdout,"line width, correlation threshold: %d  %8.3lf\n",width, thr);

  fseek(c_file, 0L, REL_EOF);			/* determine # lines in the file */

  nlines=(int)ftell(c_file)/(cbands*4*width);

  fprintf(stdout,"#lines in the correlation file: %d\n",nlines); 
  rewind(c_file);

  if (ymax <= 0) {
     ymax=nlines-start;				/* default value of ymax */
  } else if (ymax > nlines-start){
    ymax = nlines-start;
    fprintf(stdout,"insufficient #lines in the file, ymax: %d\n",ymax);
  }

  xw=xmax-xmin+1;	/* width of each line to process */
  yh=ymax-ymin+1;	/* height of array */
  offs=start+ymin-1;	/* first line of file to start reading/writing */

  bz = (unsigned char *)malloc(sizeof(unsigned char)*width);
  if(bz ==  NULL) {
    fprintf(stdout,"failure to allocate space null line\n");
    exit(1) ;
  }
  for (i=0; i < width; i++) bz[i]=LSNR;

  ccb = (float *) malloc(sizeof(float)*cbands*width*yh);
  if(ccb ==  NULL) {
        fprintf(stdout,"failure to allocate space for correlation data\n");
        exit(1) ;
  }

  fbf = (unsigned char *) malloc(sizeof(unsigned char)*width*yh);
  if(fbf == (unsigned char *) NULL) {
    fprintf(stdout,"failure to allocate space for flag array\n");
    exit(1) ;
  }

  fb=(unsigned char **)malloc(sizeof(unsigned char**)*yh);		/* row pointers of flag data */
  cc = (float **) malloc(sizeof(float*)*yh);				/* row pointers of corr data */
  if(cc ==  NULL || fb == NULL) {
    fprintf(stdout,"failure to allocate space for line pointers!\n");
    exit(1) ;
  }
  
  for (i=0; i< yh; i++){
    fb[i] = (unsigned char *)(fbf + i*width + xmin);
    cc[i] =  (float *)(ccb + cbands*i*width + (cbands-1)*width +xmin);
  }



  flag_file = fopen(flagFilename,"r+"); 
  if (flag_file == NULL){
      fprintf(stdout, "flag file does not exist, creating file: %s\n",flagFilename);
      flag_file = fopen(flagFilename,"w"); 
      for (i=0; i< width*yh; i++)fbf[i]=LSNR;		/* initialize all points to LSNR */
  }
  else{
    fprintf(stdout, "reading flag file: %s\n",flagFilename);
    fseek(flag_file, offs*width, REL_BEGIN); 		/*seek start line of flag file  */
    fread((char *)fbf, sizeof(char), yh*width, flag_file); 
    rewind(flag_file);
    for (i=0; i < width*yh; i++){fbf[i] |= LSNR; fbf[i] &= ~LAWN;};	/* clear LAWN flag, logical OR with LSNR */
  }
  
/**************** Read in correlation data *********************/
   
  fprintf(stdout,"reading correlation data file...\n");

  fseek(c_file, offs*width*cbands*sizeof(float), REL_BEGIN); 
  fread((char *)ccb, sizeof(float), cbands*yh*width, c_file); 
 
  fprintf(stdout,"setting low SNR flag...\n");
 
  for (i=0; i < yh; i++) {
    for (j=0; j < xw; j++) { 
      if (cc[i][j] > thr)fb[i][j] &= ~LSNR;	/* unset LSNR flag */;
    }
  }
       
/************** write out flag array *************/

  fprintf(stdout,"writing output file...\n");
  if (ymin > 0){
    for (i=0; i < ymin; i++) fwrite((char *)bz , sizeof(unsigned char), width, flag_file);
  }
  fwrite((char *)fbf, sizeof(unsigned char), yh*width, flag_file); 

  fclose(c_file);
  fclose(flag_file);
  return 0;
}
