#ifndef CFR_H
#define CFR_H 1

#include <stdio.h>
#include <complex.h>

#define CFR_SWAP 0
#define CFR_NOSWAP 1

int
cfr(char *hhFile,char *hvFile,char *vhFile,char *vvFile,char *output,int numberOfSamples,int numberOfLines,int swap);

float
cfrToFr(char *cfrFile, char *output, int numberOfSamples, int numberOfLines);

float complex *
readComplexLine(FILE *fp, int numberOfSamples,int byteSwap);

int
writeComplexLine(float complex *z,int numberOfSamples,FILE *output);

#endif
