#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tecToPhase.h"

/**
 * Convert from a map of Total Electron Count (TEC) to phase in radians
 *
 * @param tecFilename the file containing the Total Electron Count [electrons/m^2/1e16]
 * @param outFilename the output file name for phase
 * @param width the width of the input and output files in number of samples in range
 * @param fc the carrier frequency of the radar [Hz]
 */
int
convertToPhase(char *tecFilename, char *outFilename, int width, float fc)
{
  int length,i,j;
  float *tec,*phase;
  FILE *tecFile,*outFile;

  tecFile = fopen(tecFilename,"rb");
  outFile = fopen(outFilename,"wb");
 
  fseek(tecFile,0L,SEEK_END);
  length = ftell(tecFile);
  rewind(tecFile);
  if ( (length%width) != 0 )
    {
      printf("File has a non-integer number of lines\n");
      exit(EXIT_FAILURE);
    }
  length = (int)(length/(sizeof(float)*width));

  tec = (float *)malloc(width*sizeof(float));
  phase = (float *)malloc(width*sizeof(float));
  for(i=0;i<length;i++)
    {
      fread(tec,sizeof(float),width,tecFile);
      for(j=0;j<width;j++)
	{
          phase[j] = tecToPhase(tec[j],fc);
	}
      fwrite(phase,sizeof(float),width,outFile);
    }

  free(tec);
  free(phase);
  fclose(tecFile);
  fclose(outFile);

  return 1;
}

/**
 * Convert from Total Electron Count (TEC) in electrons/m^2*1e16 to phase in radians
 *
 * @param tec the Total Electron Count in [electrons/m^2/1e16]
 * @param fc the carrier frequency of the radar [Hz]
 * @return the phase in radians
 */
float
tecToPhase(float tec, float fc)
{
  float phase;
  // Constant (2*pi)/c * e^2/(8*pi^2*\epsilon_0*m_e)
  // e is the elementary charge, \epsilon is the permitivity of free space, m_e is the electron mass, and c is the speed of light
  // [=] coulombs^2/((m/s) (F/m) kg)
  float k3 = 8.45e-7;

  phase = (k3/fc)*tec*1e16;

  return phase;
}
