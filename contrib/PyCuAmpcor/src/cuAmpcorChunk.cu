#include "cuAmpcorChunk.h"
#include "cuAmpcorUtil.h"

/**
 * Run ampcor process for a batch of images (a chunk) 
 * @param[in] idxDown_  index oIDIVUP(i,j) ((i+j-1)/j)f the chunk along Down/Azimuth direction
 * @param[in] idxAcross_ index of the chunk along Across/Range direction
 */ 
void cuAmpcorChunk::run(int idxDown_, int idxAcross_)
{
    // set chunk index
    setIndex(idxDown_, idxAcross_);
    
    // load master image chunk 
    loadMasterChunk(); 
    
    //std::cout << "load master chunk ok\n";
    
    cuArraysAbs(c_masterBatchRaw, r_masterBatchRaw, stream);
    cuArraysSubtractMean(r_masterBatchRaw, stream);
    // load slave image chunk
    loadSlaveChunk();
    cuArraysAbs(c_slaveBatchRaw, r_slaveBatchRaw, stream);
    
    //std::cout << "load slave chunk ok\n";
    
    
    //cross correlation for none-oversampled data
    if(param->algorithm == 0) {
        cuCorrFreqDomain->execute(r_masterBatchRaw, r_slaveBatchRaw, r_corrBatchRaw);
    }
    else {
        cuCorrTimeDomain(r_masterBatchRaw, r_slaveBatchRaw, r_corrBatchRaw, stream); //time domain cross correlation
    }    
    cuCorrNormalize(r_masterBatchRaw, r_slaveBatchRaw, r_corrBatchRaw, stream);
    //find the maximum location of none-oversampled correlation
    cuArraysMaxloc2D(r_corrBatchRaw, offsetInit, stream);

// Estimate SNR (Minyan Zhong)

    //std::cout<< "flag stats 1" <<std::endl; 
    //cuArraysCopyExtractCorr(r_corrBatchRaw, r_corrBatchZoomIn, i_corrBatchZoomInValid, offsetInit, stream);

    //std::cout<< "flag stats 2" <<std::endl;
    //cuArraysSumCorr(r_corrBatchZoomIn, i_corrBatchZoomInValid, r_corrBatchSum, i_corrBatchValidCount, stream);

    //std::cout<< "flag stats 3" <<std::endl;
    //cuEstimateSnr(r_corrBatchSum, i_corrBatchValidCount, r_maxval, r_snrValue, stream);

//
    
    //offsetInit->debuginfo(stream);
    // determine the starting pixel to extract slave images around the max location
    cuDetermineSlaveExtractOffset(offsetInit, 
        param->halfSearchRangeDownRaw, // old range
        param->halfSearchRangeAcrossRaw, 
        param->halfZoomWindowSizeRaw,  // new range
        param->halfZoomWindowSizeRaw,
        stream);
    //offsetInit->debuginfo(stream);
    // oversample master
    // (deramping now included in oversampler)
    masterBatchOverSampler->execute(c_masterBatchRaw, c_masterBatchOverSampled, param->derampMethod);
    cuArraysAbs(c_masterBatchOverSampled, r_masterBatchOverSampled, stream);
    cuArraysSubtractMean(r_masterBatchOverSampled, stream);
    
    // extract slave and oversample
    cuArraysCopyExtract(c_slaveBatchRaw, c_slaveBatchZoomIn, offsetInit, stream);
    slaveBatchOverSampler->execute(c_slaveBatchZoomIn, c_slaveBatchOverSampled, param->derampMethod);
    cuArraysAbs(c_slaveBatchOverSampled, r_slaveBatchOverSampled, stream);
    
    // correlate oversampled images
    if(param->algorithm == 0) {
        cuCorrFreqDomain_OverSampled->execute(r_masterBatchOverSampled, r_slaveBatchOverSampled, r_corrBatchZoomIn);
    }
    else {
        cuCorrTimeDomain(r_masterBatchOverSampled, r_slaveBatchOverSampled, r_corrBatchZoomIn, stream); 
    }      
    cuCorrNormalize(r_masterBatchOverSampled, r_slaveBatchOverSampled, r_corrBatchZoomIn, stream);
    
    //std::cout << "debug correlation oversample\n";
    //std::cout << r_masterBatchOverSampled->height << " " << r_masterBatchOverSampled->width << "\n";
    //std::cout << r_slaveBatchOverSampled->height << " " << r_slaveBatchOverSampled->width << "\n";
    //std::cout << r_corrBatchZoomIn->height << " " << r_corrBatchZoomIn->width << "\n";
    
    // oversample the correlation surface    
    cuArraysCopyExtract(r_corrBatchZoomIn, r_corrBatchZoomInAdjust, make_int2(0,0), stream);
    
    //std::cout << "debug oversampling " << r_corrBatchZoomInAdjust << " " << r_corrBatchZoomInOverSampled << "\n";
    
    if(param->oversamplingMethod) {
        corrSincOverSampler->execute(r_corrBatchZoomInAdjust, r_corrBatchZoomInOverSampled);
    }
    else {
        corrOverSampler->execute(r_corrBatchZoomInAdjust, r_corrBatchZoomInOverSampled); 
    }
    
    //find the max again
    
    cuArraysMaxloc2D(r_corrBatchZoomInOverSampled, offsetZoomIn, corrMaxValue, stream);
    
    // determine the final offset from non-oversampled (pixel) and oversampled (sub-pixel)    
    cuSubPixelOffset(offsetInit, offsetZoomIn, offsetFinal, 
        param->oversamplingFactor, param->rawDataOversamplingFactor, 
        param->halfSearchRangeDownRaw, param->halfSearchRangeAcrossRaw,
        param->halfZoomWindowSizeRaw, param->halfZoomWindowSizeRaw,
        stream);
    //offsetInit->debuginfo(stream);
    //offsetZoomIn->debuginfo(stream);
    //offsetFinal->debuginfo(stream);    
        
    cuArraysCopyInsert(offsetFinal, offsetImage, idxDown_*param->numberWindowDownInChunk, idxAcross_*param->numberWindowAcrossInChunk,stream);

    // Minyan Zhong
    //cuArraysCopyInsert(corrMaxValue, snrImage, idxDown_*param->numberWindowDownInChunk, idxAcross_*param->numberWindowAcrossInChunk,stream);    
    //cuArraysCopyInsert(r_snrValue, snrImage, idxDown_*param->numberWindowDownInChunk, idxAcross_*param->numberWindowAcrossInChunk,stream);    

}

void cuAmpcorChunk::setIndex(int idxDown_, int idxAcross_)
{
    idxChunkDown = idxDown_;
    idxChunkAcross = idxAcross_;
	idxChunk = idxChunkAcross + idxChunkDown*param->numberChunkAcross;
    
    if(idxChunkDown == param->numberChunkDown -1) {
		nWindowsDown = param->numberWindowDown - param->numberWindowDownInChunk*(param->numberChunkDown -1);
	}
	else {
		nWindowsDown = param->numberWindowDownInChunk;
	}
	
	if(idxChunkAcross == param->numberChunkAcross -1) {
		nWindowsAcross = param->numberWindowAcross - param->numberWindowAcrossInChunk*(param->numberChunkAcross -1);
	}
	else {
		nWindowsAcross = param->numberWindowAcrossInChunk;
	}
	//std::cout << "DEBUG setIndex" << idxChunk << " " << nWindowsDown << " " << nWindowsAcross << "\n";
	
}

/// obtain the starting pixels for each chip
/// @param[in] oStartPixel 
///
void cuAmpcorChunk::getRelativeOffset(int *rStartPixel, const int *oStartPixel, int diff)
{
    for(int i=0; i<param->numberWindowDownInChunk; ++i) {
		int iDown = i;
		if(i>=nWindowsDown) iDown = nWindowsDown-1; 
        for(int j=0; j<param->numberWindowAcrossInChunk; ++j){
			int iAcross = j;
			if(j>=nWindowsAcross) iAcross = nWindowsAcross-1; 
            int idxInChunk = iDown*param->numberWindowAcrossInChunk+iAcross;
            int idxInAll = (iDown+idxChunkDown*param->numberWindowDownInChunk)*param->numberWindowAcross
				+ idxChunkAcross*param->numberWindowAcrossInChunk+iAcross;
            rStartPixel[idxInChunk] = oStartPixel[idxInAll] - diff;
            //fprintf(stderr, "relative offset %d %d %d %d\n", i, j, rStartPixel[idxInChunk], diff);
        }
    }
} 

void cuAmpcorChunk::loadMasterChunk()
{
    //load a chunk from mmap to gpu
    int startD = param->masterChunkStartPixelDown[idxChunk];
    int startA = param->masterChunkStartPixelAcross[idxChunk];
    int height =  param->masterChunkHeight[idxChunk];
    int width = param->masterChunkWidth[idxChunk];
    masterImage->loadToDevice(c_masterChunkRaw->devData, startD, startA, height, width, stream);
    std::cout << "debug load master: " << startD << " " <<  startA << " " <<  height << " "  << width << "\n";
    //copy the chunk to a batch of images format (nImages, height, width) 
    //use cpu for some simple math  
    getRelativeOffset(ChunkOffsetDown->hostData, param->masterStartPixelDown, param->masterChunkStartPixelDown[idxChunk]);
    ChunkOffsetDown->copyToDevice(stream);
    getRelativeOffset(ChunkOffsetAcross->hostData, param->masterStartPixelAcross, param->masterChunkStartPixelAcross[idxChunk]);
    ChunkOffsetAcross->copyToDevice(stream);
    // if derampMethod = 0 (no deramp), take amplitudes; otherwise, copy complex data 
    if(param->derampMethod == 0) {
        cuArraysCopyToBatchAbsWithOffset(c_masterChunkRaw, param->masterChunkWidth[idxChunk], 
            c_masterBatchRaw, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
    }
    else {
        cuArraysCopyToBatchWithOffset(c_masterChunkRaw, param->masterChunkWidth[idxChunk], 
            c_masterBatchRaw, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
    }
}

void cuAmpcorChunk::loadSlaveChunk()
{
         //load a chunk from mmap to gpu
    slaveImage->loadToDevice(c_slaveChunkRaw->devData, 
        param->slaveChunkStartPixelDown[idxChunk], 
        param->slaveChunkStartPixelAcross[idxChunk], 
        param->slaveChunkHeight[idxChunk], 
        param->slaveChunkWidth[idxChunk], 
        stream);
    //copy to a batch format (nImages, height, width)
    getRelativeOffset(ChunkOffsetDown->hostData, param->slaveStartPixelDown, param->slaveChunkStartPixelDown[idxChunk]);
    ChunkOffsetDown->copyToDevice(stream);
    getRelativeOffset(ChunkOffsetAcross->hostData, param->slaveStartPixelAcross, param->slaveChunkStartPixelAcross[idxChunk]);
    ChunkOffsetAcross->copyToDevice(stream);
    if(param->derampMethod == 0) {
    cuArraysCopyToBatchAbsWithOffset(c_slaveChunkRaw, param->slaveChunkWidth[idxChunk], 
        c_slaveBatchRaw, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
    }
    else
    {
       cuArraysCopyToBatchWithOffset(c_slaveChunkRaw, param->slaveChunkWidth[idxChunk], 
        c_slaveBatchRaw, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
    } 
}

cuAmpcorChunk::cuAmpcorChunk(cuAmpcorParameter *param_, SlcImage *master_, SlcImage *slave_, 
    cuArrays<float2> *offsetImage_, cuArrays<float> *snrImage_, cudaStream_t stream_)
{
    param = param_;
    masterImage = master_;
    slaveImage = slave_;	
    offsetImage = offsetImage_;
    snrImage = snrImage_;
    stream = stream_;
   
    std::cout << "debug Chunk creator " << param->maxMasterChunkHeight << " " << param->maxMasterChunkWidth << "\n";
    c_masterChunkRaw = new cuArrays<float2> (param->maxMasterChunkHeight, param->maxMasterChunkWidth); 
    c_masterChunkRaw->allocate();
    
    c_slaveChunkRaw = new cuArrays<float2> (param->maxSlaveChunkHeight, param->maxSlaveChunkWidth); 
    c_slaveChunkRaw->allocate();
    
    ChunkOffsetDown = new cuArrays<int> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    ChunkOffsetDown->allocate();
    ChunkOffsetDown->allocateHost();
    ChunkOffsetAcross = new cuArrays<int> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    ChunkOffsetAcross->allocate();
    ChunkOffsetAcross->allocateHost();
	
    c_masterBatchRaw = new cuArrays<float2> (
        param->windowSizeHeightRaw, param->windowSizeWidthRaw,
        param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    c_masterBatchRaw->allocate();
    
    c_slaveBatchRaw = new cuArrays<float2> (
        param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
        param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    c_slaveBatchRaw->allocate();
    
    r_masterBatchRaw = new cuArrays<float> (
        param->windowSizeHeightRaw, param->windowSizeWidthRaw,
        param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    r_masterBatchRaw->allocate();
    
    r_slaveBatchRaw = new cuArrays<float> (
        param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
        param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    r_slaveBatchRaw->allocate();
    
    c_slaveBatchZoomIn = new cuArrays<float2> (
        param->searchWindowSizeHeightRawZoomIn, param->searchWindowSizeWidthRawZoomIn,
        param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    c_slaveBatchZoomIn->allocate();
	
    c_masterBatchOverSampled = new cuArrays<float2> (
			param->windowSizeHeight, param->windowSizeWidth,
			param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    c_masterBatchOverSampled->allocate();

    c_slaveBatchOverSampled = new cuArrays<float2> (
			param->searchWindowSizeHeight, param->searchWindowSizeWidth,
			param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    c_slaveBatchOverSampled->allocate();
    
    r_masterBatchOverSampled = new cuArrays<float> (
			param->windowSizeHeight, param->windowSizeWidth,
			param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    r_masterBatchOverSampled->allocate();

    r_slaveBatchOverSampled = new cuArrays<float> (
			param->searchWindowSizeHeight, param->searchWindowSizeWidth,
			param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    r_slaveBatchOverSampled->allocate();
    
    masterBatchOverSampler = new cuOverSamplerC2C(
        c_masterBatchRaw->height, c_masterBatchRaw->width, //orignal size
        c_masterBatchOverSampled->height, c_masterBatchOverSampled->width, //oversampled size 
        c_masterBatchRaw->count, stream);
	
    slaveBatchOverSampler = new cuOverSamplerC2C(c_slaveBatchZoomIn->height, c_slaveBatchZoomIn->width, 
            c_slaveBatchOverSampled->height, c_slaveBatchOverSampled->width, c_slaveBatchRaw->count, stream);
    
    r_corrBatchRaw = new cuArrays<float> (
			param->searchWindowSizeHeightRaw-param->windowSizeHeightRaw+1, 
			param->searchWindowSizeWidthRaw-param->windowSizeWidthRaw+1, 
			param->numberWindowDownInChunk, 
			param->numberWindowAcrossInChunk);
    r_corrBatchRaw->allocate();
    
    r_corrBatchZoomIn = new cuArrays<float> (
			param->searchWindowSizeHeight - param->windowSizeHeight+1, 
			param->searchWindowSizeWidth - param->windowSizeWidth+1, 
			param->numberWindowDownInChunk, 
			param->numberWindowAcrossInChunk);
    r_corrBatchZoomIn->allocate();
    
    r_corrBatchZoomInAdjust = new cuArrays<float> (
			param->searchWindowSizeHeight - param->windowSizeHeight, 
			param->searchWindowSizeWidth - param->windowSizeWidth, 
			param->numberWindowDownInChunk, 
			param->numberWindowAcrossInChunk);
    r_corrBatchZoomInAdjust->allocate();
    
    
    r_corrBatchZoomInOverSampled = new cuArrays<float> (
        param->zoomWindowSize * param->oversamplingFactor, 
        param->zoomWindowSize * param->oversamplingFactor,
        param->numberWindowDownInChunk, 
        param->numberWindowAcrossInChunk);
    r_corrBatchZoomInOverSampled->allocate();
        
    offsetInit = new cuArrays<int2> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    offsetInit->allocate();
		
    offsetZoomIn = new cuArrays<int2> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    offsetZoomIn->allocate();
    		
    offsetFinal = new cuArrays<float2> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    offsetFinal->allocate();
		
    corrMaxValue = new cuArrays<float> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    corrMaxValue->allocate();
	
    if(param->oversamplingMethod) {
        corrSincOverSampler = new cuSincOverSamplerR2R(param->zoomWindowSize, param->oversamplingFactor, stream);
    }
    else {	
        corrOverSampler= new cuOverSamplerR2R(param->zoomWindowSize, param->zoomWindowSize,
			(param->zoomWindowSize)*param->oversamplingFactor, 
		    (param->zoomWindowSize)*param->oversamplingFactor,
		    param->numberWindowDownInChunk*param->numberWindowAcrossInChunk, 
		    stream);		
	}	     
    if(param->algorithm == 0) {
        cuCorrFreqDomain = new cuFreqCorrelator(
            param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
            param->numberWindowDownInChunk*param->numberWindowAcrossInChunk,
            stream);
        cuCorrFreqDomain_OverSampled = new cuFreqCorrelator(
            param->searchWindowSizeHeight, param->searchWindowSizeWidth,
            param->numberWindowDownInChunk*param->numberWindowAcrossInChunk,
            stream); 
    }

	

    debugmsg("all objects in chunk are created ...\n");

}
cuAmpcorChunk::~cuAmpcorChunk()
{
    /*
    delete masterChunkRaw;
    delete slaveChunkRaw;
    delete ChunkOffsetDown;
    delete ChunkOffsetAcross;
    delete masterBatchRaw;
    delete slaveBatchRaw;
    delete masterChunkOverSampled;
    delete slaveChunkOverSampled;
    delete masterChunkOverSampler;
    delete slaveChunkOverSampler;
    delete masterChunk;
    delete slaveChunk;
    delete corrChunk;
    delete offsetInit;
    delete zoomInOffset;
    delete offsetFinal;
    delete corrChunkZoomIn;
    delete corrChunkZoomInOverSampled;
    delete corrOverSampler;
    delete corrSincOverSampler;
    delete corrMaxValue;
    if(param->algorithm == 0)
        delete cuCorrFreqDomain;
    */
}
