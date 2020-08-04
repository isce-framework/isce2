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
    double wvl1;    //Reference wavelength
    double wvl2;    //Secondary wavelength
    double drg1;    //Reference spacing
    double drg2;    //Secondary spacing
    int flatten;    //Flatten flag
    double wgt;     //Range filter weight
};

#endif
