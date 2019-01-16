/* 
 * cuAmpcorChunk.h
 * Purpose: a group of chips processed at the same time
*/

#ifndef __CUAMPCORCHUNK_H
#define __CUAMPCORCHUNK_H

#include "SlcImage.h"
#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include "cuOverSampler.h"
#include "cuSincOverSampler.h"
#include "cuCorrFrequency.h"

class cuAmpcorChunk{
private:
    int idxChunkDown;
	int idxChunkAcross;
    int idxChunk;
    int nWindowsDown;
    int nWindowsAcross;

	int devId;
	cudaStream_t stream; 
	
	SlcImage *masterImage;
	SlcImage *slaveImage;
	cuAmpcorParameter *param;
	cuArrays<float2> *offsetImage;
	cuArrays<float> *snrImage;
	
	cuArrays<float2> * c_masterChunkRaw, * c_slaveChunkRaw; 
    cuArrays<float2> * c_masterBatchRaw, * c_slaveBatchRaw, * c_slaveBatchZoomIn;
    cuArrays<float> * r_masterBatchRaw, * r_slaveBatchRaw;
    cuArrays<float2> * c_masterBatchOverSampled, * c_slaveBatchOverSampled; 
    cuArrays<float> * r_masterBatchOverSampled, * r_slaveBatchOverSampled;
    cuArrays<float> * r_corrBatchRaw, * r_corrBatchZoomIn, * r_corrBatchZoomInOverSampled, * r_corrBatchZoomInAdjust;
    
    cuArrays<int> *ChunkOffsetDown, *ChunkOffsetAcross;
	 
	cuOverSamplerC2C *masterBatchOverSampler, *slaveBatchOverSampler;
    
    cuOverSamplerR2R *corrOverSampler;
    cuSincOverSamplerR2R *corrSincOverSampler; 
    
	//for frequency domain
	cuFreqCorrelator *cuCorrFreqDomain, *cuCorrFreqDomain_OverSampled;
    
	cuArrays<int2> *offsetInit;
	cuArrays<int2> *offsetZoomIn;
	cuArrays<float2> *offsetFinal;

        //corr statistics
    cuArrays<int2> *i_maxloc;
    cuArrays<float> *r_maxval;
        
        cuArrays<float> *r_corrBatchSum;
        cuArrays<int> *i_corrBatchZoomInValid, *i_corrBatchValidCount;
        
        cuArrays<float> *corrMaxValue;
        cuArrays<float> *r_snrValue;
	
public:
	cuAmpcorChunk()	{}
	//cuAmpcorChunk(cuAmpcorParameter *param_, SlcImage *master_, SlcImage *slave_);
	
	void setIndex(int idxDown_, int idxAcross_);


	cuAmpcorChunk(cuAmpcorParameter *param_, SlcImage *master_, SlcImage *slave_, cuArrays<float2> *offsetImage_, 
	            cuArrays<float> *snrImage_, cudaStream_t stream_);
    
    void loadMasterChunk();
    void loadSlaveChunk();
    void getRelativeOffset(int *rStartPixel, const int *oStartPixel, int diff);
	
    ~cuAmpcorChunk(); 
	
	void run(int, int); 
};



#endif 
