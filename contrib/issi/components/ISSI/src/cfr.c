#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include "cfr.h"
#include "issibyteswap.h"

/**
 * Given quad-pol SAR data, calculate the Faraday Rotation at each sample
 * location and write out its value as a complex number.  The Faraday Rotation
 * is calculated using the method of Bickle nad Bates (1965).
 *
 * @note Bickel, S. H., and R. H. T. Bates (1965), Effects of magneto-ionic propagation 
 * on the polarization scattering matrix, pp. 1089,1091.
 *
 * @param hhFile the data file containing the HH polarity
 * @param hvFile the data file containing the HH polarity
 * @param vhFile the data file containing the HH polarity
 * @param vvFile the data file containing the HH polarity
 * @param output the file in which to place the Faraday Rotation
 * @param numberOfSamples the number of samples in range in the input files
 * @param numberOfLines the number of samples in azimuth in the input files
 * @param swap flag for byte swapping
 */
int
cfr(char *hhFile,char *hvFile,char *vhFile,char *vvFile,char *output,int numberOfSamples,int numberOfLines,int swap)
{
	int i,j;
	float complex *hhData,*hvData,*vhData,*vvData,*ans;
	FILE *hh,*hv,*vh,*vv,*out;

	// Open input and output fi les
	hh = fopen(hhFile,"rb");
	hv = fopen(hvFile,"rb");
	vh = fopen(vhFile,"rb");
	vv = fopen(vvFile,"rb");
	out = fopen(output,"wb");

	ans = (float complex *)malloc(numberOfSamples*sizeof(float complex));
	for(i=0;i<numberOfLines;i++)
	{
		if ((i%1000) == 0)
		{
			printf("Line %d\n",i);
		}
		// Read in a line of data from each file
		hhData = readComplexLine(hh,numberOfSamples,swap);
		hvData = readComplexLine(hv,numberOfSamples,swap);
		vhData = readComplexLine(vh,numberOfSamples,swap);
		vvData = readComplexLine(vv,numberOfSamples,swap);

		// Calculate the Faraday rotation using the method of ...
		for(j=0;j<numberOfSamples;j++)
		{
			float complex tmp1,tmp2,z12,z21;
			tmp1 = (hhData[j] + vvData[j])*(0.0 + 1.0*I);
			tmp2 = (vhData[j] - hvData[j]);
			z12 = tmp1 + tmp2;
			z21 = tmp1 - tmp2;
			ans[j] = z12*conjf(z21);
		}
		// Write out a line of Faraday rotation calculations
		writeComplexLine(ans,numberOfSamples,out);
	}

	free(ans);
	free(hhData);
	free(hvData);
	free(vhData);
	free(vvData);
	fclose(hh);
	fclose(hv);
	fclose(vh);
	fclose(vv);
	fclose(out);

	return EXIT_SUCCESS;
}

inline static float bswap_flt(float x) {
  union {uint32_t i; float f;} u;
  u.f = x;
  u.i = bswap_32(u.i);
  return u.f;
}


void
printComplex(float complex f)
{
	printf("%f %fI\n",crealf(f),cimagf(f));
}

/**
 * Read an array of complex float point values from an open file pointer,
 * possibly swapping the byte-order.
 * @param fp an open file pointer
 * @param numberOfSamples the number of complex floating point values to read
 * @param swapBytes if 0, swap the byte-order of the data, else, do not
 * @return an array of complex floating point values
 */
float complex *
readComplexLine(FILE *fp, int numberOfSamples,int swapBytes)
{
    int i;
    float real,imag;
    float complex *z;

    z = (float complex *)malloc(numberOfSamples*sizeof(float complex));

	for(i=0;i<numberOfSamples;i++)
	{
		fread(&real,sizeof(float),1,fp);
		fread(&imag,sizeof(float),1,fp);

		if (swapBytes == CFR_SWAP)
		{
			real = bswap_flt(real);
			imag = bswap_flt(imag);
		}

		z[i] = real + imag*I;
	}

	return z;
}

/**
 * Given an array of known size containing complex floating point values, write
 * it out to an open file pointer.
 *
 * @param z an array of complex values
 * @param numberOfSamples the length of the array z
 * @param output the file pointer in which to write out the array z
 */
int
writeComplexLine(float complex *z,int numberOfSamples,FILE *output)
{
	int i,j;
	float *vals;

	vals = (float *)malloc(2*numberOfSamples*sizeof(float));

	// Unpack complex numbers
	for(i=0,j=0;i<numberOfSamples;i++,j+=2)
	{
		vals[j] = crealf(z[i]);
		vals[j+1] = cimagf(z[i]);
	}

	fwrite(vals,sizeof(float),2*numberOfSamples,output);

	free(vals);

	return EXIT_SUCCESS;
}
