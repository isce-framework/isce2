//Experimental struct definition
//Author: Piyush Agram
//

#ifndef crossmul_h
#define crossmul_h

#include <stdlib.h>
#include <stdio.h>

struct crossmulState{
    int na;         //Width
    int nd;         //Length
    double scale;   //Scale
    int looksac;    //Range looks
    int looksdn;    //Azimuth looks
    int blocksize;  //Azimuth block size
    double wvl1;    //Master wavelength
    double wvl2;    //Slave wavelength
    double drg1;    //Master spacing
    double drg2;    //Slave spacing
    int flatten;    //Flatten flag
    double wgt;     //Range filter weight
};

#endif
