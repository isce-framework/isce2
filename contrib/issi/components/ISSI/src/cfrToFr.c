#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "cfr.h"

/**
 * Given a file containing the complex valued Faraday Rotation, extract
 * the angle of the Faraday Rotation and write it out to a separate file.
 * As a byproduct, calculate the mean Faraday Rotation angle.
 *
 * @param cfrFile a file containing the complex valued Faraday Rotation
 * @param output the file in which to save the Faraday Rotation angle
 * @param numberOfSamples the number of samples in range in the input file
 * @param numberOfLines the number of samples in azimuth in the input file
 * @return the mean Faraday Rotation angle
 */
float
cfrToFr(char *cfrFile, char *output,int numberOfSamples, int numberOfLines)
{
	int i,j;
	float aveFr;
	float *fr;
	float complex *cfrData;
	FILE *cfr,*out;

	cfr = fopen(cfrFile,"rb");
	out = fopen(output,"wb");

	fr = (float *)malloc(numberOfSamples*sizeof(float));
	aveFr = 0.0;

	for(i=0;i<numberOfLines;i++)
	{
		if ((i%1000) == 0)
		{
			printf("Line %d\n",i);
		}

		cfrData = readComplexLine(cfr,numberOfSamples,CFR_NOSWAP);
		for(j=0;j<numberOfSamples;j++)
		{
			fr[j] = cargf(cfrData[j])/4.0;
		        aveFr += fr[j];
		}
		fwrite(fr,sizeof(float),numberOfSamples,out);
	}

	aveFr = aveFr/(numberOfSamples*numberOfLines);

	free(fr);
	fclose(cfr);
	fclose(out);

	return aveFr;
}
