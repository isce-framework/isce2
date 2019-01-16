#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

/**
 * Take the smoothed phase image from one file and the correctly scaled magnitude image from another and 
 * combine them in to one file
 */
int
rescale_magnitude(char *int_filename,char *sm_filename,int width,int length)
{
  int i,j;
  float complex *original,*smooth;
  FILE *int_file,*sm_file;

  int_file = fopen(int_filename,"rb");
  sm_file = fopen(sm_filename,"rb+");

  original = (float complex *)malloc(width*sizeof(float complex));
  smooth = (float complex *)malloc(width*sizeof(float complex));

  printf("Rescaling magnitude\n");
  for(i=0;i<length;i++)
    {
      fread(original,sizeof(float complex),width,int_file);
      fread(smooth,sizeof(float complex),width,sm_file);
      for(j=0;j<width;j++)
	{
          float mag = cabs(original[j]);
          float phase = carg(smooth[j]);
	  smooth[j] = mag*(cos(phase) + I*sin(phase));
	}
      if (i%1000) {fprintf(stderr,"\rline: %5d",i);}
      fseek(sm_file,-width*sizeof(float complex),SEEK_CUR); // Back up to the begining of the line
      fwrite(smooth,sizeof(float complex),width,sm_file); // Replace the line with the smooth, rescaled value
      fflush(sm_file);
    }

  free(original);
  free(smooth);
  fclose(int_file);
  fclose(sm_file);

  return(EXIT_SUCCESS);
}


