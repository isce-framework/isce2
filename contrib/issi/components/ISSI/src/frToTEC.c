#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "frToTEC.h"

/**
 * Convert from a map of Faraday rotation to Total Electron Count (TEC)
 * This function assumes a constant value for the magnetic field in the 
 * look direction of the radar (bdotk).
 *
 * @param frFilename the file containing the Faraday rotation map [radians]
 * @param outFilename the output file name
 * @param width the width of the input and output files in number of samples
 * @param bdotk a constant value for the B field in the direction k [gauss]
 * @param fc the carrier frequency of the radar [Hz]
 */
int
convertToTec(char *frFilename, char *outFilename, int width, float bdotk, float fc)
{
  int i,j,length;
  float *fr,*tec;
  FILE *frFile,*outFile;

  frFile = fopen(frFilename,"rb");
  outFile = fopen(outFilename,"wb");
 
  fseek(frFile,0L,SEEK_END);
  length = ftell(frFile);
  rewind(frFile);
  if ( (length%width) != 0 )
    {
      printf("File has a non-integer number of lines\n");
      exit(EXIT_FAILURE);
    }
  length = (int)(length/(sizeof(float)*width));

  fr = (float *)malloc(width*sizeof(float));
  tec = (float *)malloc(width*sizeof(float));
  for(i=0;i<length;i++)
    {
      fread(fr,sizeof(float),width,frFile);
      for(j=0;j<width;j++)
	{
          tec[j] = frToTec(fr[j],fc,bdotk);
	}
      fwrite(tec,sizeof(float),width,outFile);
    }

  free(fr);
  free(tec);
  fclose(frFile);
  fclose(outFile);

  return 1;
}

/**
 * Convert from a map of Faraday rotation to Total Electron Count (TEC)
 * This function assumes that a file containing the values of the magnetic
 * B-vector in units of gauss in the look direction of the radar (bdotk) at 
 * each point is provided.
 *
 * @param frFilename the file containing the Faraday rotation map [radians]
 * @param outFilename the output file name
 * @param bdotkFilename the file containing the values of bdotk [gauss]
 * @param width the width of the input and output files in number of samples
 * @param fc the carrier frequency of the radar [Hz]
 */
int
convertToTecWBdotK(char *frFilename, char *outFilename, char *bdotkFilename,int width, float fc)
{
  int i,j,length;
  float *tec,*fr,*bdotk;
  FILE *frFile,*bdotkFile,*outfile;

  frFile = fopen(frFilename,"rb");
  bdotkFile = fopen(bdotkFilename,"rb");
  outfile = fopen(outFilename,"wb");
 
  fseek(frFile,0L,SEEK_END);
  length = ftell(frFile);
  rewind(frFile);
  if ( (length%width) != 0 )
    {
      printf("File has a non-integer number of lines\n");
      exit(EXIT_FAILURE);
    }
  length = (int)(length/(sizeof(float)*width));

  fr = (float *)malloc(width*sizeof(float));
  bdotk = (float *)malloc(width*sizeof(float));
  tec = (float *)malloc(width*sizeof(float));
  for(i=0;i<length;i++)
    {
      fread(fr,sizeof(float),width,frFile);
      fread(bdotk,sizeof(float),width,bdotkFile);
      for(j=0;j<width;j++)
	{
          tec[j] = frToTec(fr[j],fc,bdotk[j]);
	}
      fwrite(tec,sizeof(float),width,outfile);
    }

  free(fr);
  free(bdotk);
  free(tec);
  fclose(frFile);
  fclose(bdotkFile);
  fclose(outfile);

  return 1;
}

/**
 * Convert from Faraday rotation in radians, to Total Electron Count (TEC) in electrons/m^2*1e16
 *
 * @param fr the Faraday rotation [radians]
 * @param fc the carrier frequency of the radar [Hz]
 * @param bdotk the value of the B field in the direction k [gauss]
 * @return The Total Electron Count (TEC) [electrons/m^2/1e16]
 */
float
frToTec(float fr,float fc, float bdotk)
{
  float tec;
  // Constant |e|^3/(8pi^2 c \epsilon_0 m^2_e)
  // e is the elementary charge, \epsilon is the permitivity of free space, m_e is the electron mass, and c is the speed of light
  // [=] coulombs^3/((m/s)^2 (F/m)  kg^2)
  // [=] coulomb m/kg (? maybe)
  float k1 = 2.365e4; 

  tec = (fr*powf(fc,(float)2.0)/(k1*bdotk*1e-4))/1e16;

  return tec;
}
