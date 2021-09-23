//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef RESAMP_SLC_H
#define RESAMP_SLC_H

#include <cstdint>
#include "Poly2d.h"

struct ResampSlc {

    uint64_t slcInAccessor, slcOutAccessor, residRgAccessor, residAzAccessor;
    double wvl, slr, r0, refwvl, refslr, refr0;
    int outWidth, outLength, inWidth, inLength;
    bool isComplex, flatten, usr_enable_gpu;
    Poly2d *rgCarrier, *azCarrier, *rgOffsetsPoly, *azOffsetsPoly;
    Poly2d *dopplerPoly;

    ResampSlc();
    ResampSlc(const ResampSlc&);
    ~ResampSlc();
    void setRgCarrier(Poly2d*);
    void setAzCarrier(Poly2d*);
    void setRgOffsets(Poly2d*);
    void setAzOffsets(Poly2d*);
    void setDoppler(Poly2d*);
    Poly2d* releaseRgCarrier();
    Poly2d* releaseAzCarrier();
    Poly2d* releaseRgOffsets();
    Poly2d* releaseAzOffsets();
    Poly2d* releaseDoppler();
    void clearPolys();
    void resetPolys();
    void resamp();
    void _resamp_cpu();
    void _resamp_gpu();

};

#endif
