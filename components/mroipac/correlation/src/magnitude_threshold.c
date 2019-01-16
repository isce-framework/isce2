#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

/**
 * Threshold a phase file using the magnitude values from a coregistered
 * interferogram located in a separate file and output the magnitude of 
 * the interferogram with the thresholded phase.
 *
 * @param intFilename interferogram file name
 * @param phsFilename phase file name
 * @param outFilename output file name
 * @param thresh the magnitude threshold
 * @param width the number of samples per row
 */
int
magnitude_threshold(char *intFilename,char *phsFilename,char *outFilename,double thresh,int width)
{
  long i,j,length,size;
  float *phsRow;
  float complex *intRow,*outRow;
  FILE *intFP,*phsFP,*outFP;
  
  intFP = fopen(intFilename,"r");
  phsFP = fopen(phsFilename,"r");
  outFP = fopen(outFilename,"w");
 
  // Get the file size
  fseek(intFP,0,SEEK_END);
  size = ftell(intFP);
  length = (long)(size/(width*sizeof(float complex)));
  rewind(intFP);

  intRow = (float complex *)malloc(width*sizeof(float complex));
  outRow = (float complex *)malloc(width*sizeof(float complex));
  phsRow = (float *)malloc(width*sizeof(float));
  
  for(i=0;i<length;i++) {
    fread(intRow,sizeof(float complex),width,intFP);
    fread(phsRow,sizeof(float),width,phsFP);
    for(j=0;j<width;j++) {
      float mag = cabsf(intRow[j]);
      float origPhs = cargf(intRow[j]);
      // If the magnitude is below the chosen threshold
      // or if the original phase was 0, set the resulting phase
      // to zero.
      if ((mag < thresh) || (origPhs == 0.0)) {
	outRow[j] = mag + I*0.0;
	} else {
	outRow[j] = mag*cexpf(I*phsRow[j]);
        }
    }
    fwrite(outRow,sizeof(float complex),width,outFP);
  }
  
  free(intRow);
  free(phsRow);
  fclose(intFP);
  fclose(phsFP);
  fclose(outFP);

  return 0;
}
