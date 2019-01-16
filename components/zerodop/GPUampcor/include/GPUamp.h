//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef GPU_AMPCOR_H
#define GPU_AMPCOR_H

int nBlocksPossible(int*);
void runGPUAmpcor(float*,int*,void**,void**,int*,int*,int**,float**);

#endif
