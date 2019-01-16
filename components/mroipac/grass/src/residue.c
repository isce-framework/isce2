#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "defines.h"

void mk_res(float *, unsigned char *, int, int, int, int, int, int*, int*) ;

/**
 * @param intFilename input interferogram
 * @param flagFilename name of the flag file
 * @param width number of samples per row
 * @param xmin offset to starting range pixel
 * @param xmax offset to last range pixel
 * @param ymin offset to starting azimuth row
 * @param ymax offset to last azimuth row
 */
int
residues(char *intFilename, char *flagFilename, int width, int xmin, int xmax, int ymin, int ymax)
{
    float *buf, *phase;		/* interferogram line buffer, phase array */	
    double frac;		/* fraction of image which is residues */	
    int nlines;			/* number of lines in the file */
    int offs;			/* offset number of lines to read from start of file*/
    int xw,yh;			/* width, height of processed region */
    int i,j;
    int ff;			/* file flag (NEW or OLD ) */
    int np_res, nm_res;		/* number of positive and negative residues */
    unsigned char *fbf, *bufz;	/* flag array, buffer with zeroes*/
    FILE *int_file, *flag_file;

  int_file = fopen(intFilename,"r"); 
  if (int_file == NULL){fprintf(stderr,"cannot open interferogram file: %s\n",intFilename); exit(-1);}
 
  fseek(int_file, 0L, REL_EOF);
  nlines=(int)ftell(int_file)/(width*2*sizeof(float));
  fprintf(stdout,"#lines in the file: %d\n",nlines); 
  rewind(int_file);

  flag_file = fopen(flagFilename,"r+"); 
  if (flag_file != NULL) ff=OLD_FILE;
  else {
    fprintf(stderr,"cannot open output flag file, creating new file: %s\n",flagFilename);
    flag_file = fopen(flagFilename,"w");
    if(flag_file == NULL){fprintf(stderr,"cannot create new flag file: %s\n",flagFilename); exit(-1);}
    ff=NEW_FILE;         
  }
 
  if (ymax <= 0) {
      ymax = nlines-1;
  } else if (ymax > nlines-1){
    ymax = nlines-1; 
    fprintf(stderr,"insufficient #lines in the file, resetting length: %d\n",ymax);
  }

  if (xmax <= 0) {
      xmax = width-1;
  } else if(xmax > width-1){
    xmax=width-1;
    fprintf(stderr,"file has insufficient width, resetting width: %d\n",xmax);
  }

  yh=ymax-ymin+1;		/* height of array */ 
  xw=xmax-xmin+1;		/* width of array */ 
  offs=ymin;			/* first line of file to start reading/writing */

  fprintf(stdout,"flag array width, height: %d %d\n",xw,yh);

/************** memory allocation ***************/ 
  buf = (float *)malloc(2*sizeof(float)*width);
  if(buf == NULL){fprintf(stderr,"failure to allocate space for input line buffer\n"); exit(-1);}

  bufz = (unsigned char *)malloc(width);
  if(bufz == NULL){fprintf(stderr,"failure to allocate space for null output line\n"); exit(-1);}
  for (j=0; j < width; j++)bufz[j]=0;		/* initialize buffer row array */

  phase = (float *) malloc(sizeof(float)*width*yh);
  if(phase == NULL){fprintf(stderr,"failure to allocate space for phase data\n"); exit(-1);}

  fbf = (unsigned char *) malloc (sizeof(unsigned char)*width*yh);
  if(fbf == NULL){fprintf(stderr,"failure to allocate space for flag array\n"); exit(-1);}

  if(ff == NEW_FILE){
    fprintf(stderr,"initializing flag array...\n");
    for (i=0; i < yh; i++){				/* initialize flag array */
      for (j=0; j < width; j++){
        fbf[i*width+j]=0;
      }
    }
  }
  else {
    fprintf(stderr,"reading flag array...\n");
    fseek(flag_file, offs*width*sizeof(unsigned char), REL_BEGIN);  
    fread((char *)fbf, sizeof(unsigned char), width*yh, flag_file); 
  }

/**************** Read in data, convert to phase *********************/
   
  fseek(int_file, offs*width*2*sizeof(float), REL_BEGIN); 	/*seek start line */

  for (i=0; i < yh; i++){
    if (i%100 == 0){fprintf(stdout,"\rreading input line %d", i); fflush(stdout);}
    fread((char *)buf, sizeof(float), 2*width, int_file);	/* read next line */

    for (j=0; j < width; j++) {
      if((buf[2*j]==0.) && (buf[2*j+1]==0.)) phase[i*width+j]=0.;/* phase undefined */
      else phase[i*width+j] = atan2(buf[2*j+1],buf[2*j]);
    }
  }

/************************ find residues ******************************/

  fprintf(stdout,"\ncalculating residues...\n");
  mk_res(phase, fbf, width, xmin, xmax, ymin, ymax, &np_res, &nm_res);

  frac = (double)(np_res+nm_res)/fabs((double)(xw*yh));

  fprintf(stderr,"\nnumber of positive residues: %d\n",np_res);
  fprintf(stderr,"number of negative residues: %d\n",nm_res);
  fprintf(stderr,"total number of residues:    %d\n",np_res+nm_res);
  fprintf(stderr,"fraction residues:           %8.5f\n",frac);

/********************** write out flag array *************************/

  fprintf(stderr,"writing flag array...\n");
  
  if (ff == OLD_FILE) fseek(flag_file, offs*width*sizeof(unsigned char), REL_BEGIN);
  else for (i=0; i< ymin; i++) fwrite((char *)bufz, sizeof(unsigned char), width, flag_file);	/* write out blank lines */ 
  
  fwrite((char *)fbf, sizeof(unsigned char), yh*width, flag_file);

  fclose(int_file);
  fclose(flag_file);

  return 0;
}  

void mk_res(float *phase, unsigned char *flags, int width, int xmin, 
            int xmax, int ymin, int ymax, int *n_m, int *n_p) 
{

  int i,j,k,l,offs,offt,offb,yh;

  static int ioft[4] = {0, 1, 1, 0} ;   /* row index offset for top of difference */
  static int joft[4] = {1, 1, 0, 0} ;   /* col index offset for top of difference */
  static int iofb[4] = {0, 0, 1, 1} ;   /* row index offset for bottom of difference */
  static int jofb[4] = {0, 1, 1, 0} ;   /* col index offset for bottom of difference */

  *n_m=0;					/* initialize residue counters */
  *n_p=0;
   
  yh=ymax-ymin+1;				/* height of array */ 
  if (xmax >= width-1) xmax = width-1;		 
 
  for (i=0; i < yh-1; i++) {			/* 1 pixel border at the end */
    if (i%100 == 0) fprintf(stderr,"\rprocessing line %d", i);
    offs=i*width;				/* offset for address in array */

    for (j=xmin; j < xmax-1; j++) {		/* 1 pixel border at the left edge */
      for(k=0, l=0; l < 4; l++) {
	offt = offs+ioft[l]*width + j + joft[l];
	offb = offs+iofb[l]*width + j + jofb[l];
	k += nint((double)(phase[offt]-phase[offb])/TWO_PI) ;
      }

      if (k != 0) {				/* residue? */		
	if (k >0) {
          (*n_p)++;				/* increment positive residue counter */
          flags[offs+j]=(flags[offs+j] | PLUS) & ~GUID; /* set flag PLUS, clear GUID */
        }
	else {
          (*n_m)++;				/* increment negative residue counter */
          flags[offs+j]=(flags[offs+j] | MINU) & ~GUID; /* set flag MINU, clear GUID */
        }
      }	 
    }
  }
}
