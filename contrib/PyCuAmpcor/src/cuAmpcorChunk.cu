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

    // load reference image chunk
    loadReferenceChunk();

    //std::cout << "load reference chunk ok\n";

    cuArraysAbs(c_referenceBatchRaw, r_referenceBatchRaw, stream);
    cuArraysSubtractMean(r_referenceBatchRaw, stream);
    // load secondary image chunk
    loadSecondaryChunk();
    cuArraysAbs(c_secondaryBatchRaw, r_secondaryBatchRaw, stream);

    //std::cout << "load secondary chunk ok\n";


    //cross correlation for none-oversampled data
    if(param->algorithm == 0) {
        cuCorrFreqDomain->execute(r_referenceBatchRaw, r_secondaryBatchRaw, r_corrBatchRaw);
    }
    else {
        cuCorrTimeDomain(r_referenceBatchRaw, r_secondaryBatchRaw, r_corrBatchRaw, stream); //time domain cross correlation
    }
    cuCorrNormalize(r_referenceBatchRaw, r_secondaryBatchRaw, r_corrBatchRaw, stream);


    // find the maximum location of none-oversampled correlation
    // 41 x 41, if halfsearchrange=20
    //cuArraysMaxloc2D(r_corrBatchRaw, offsetInit, stream);
    cuArraysMaxloc2D(r_corrBatchRaw, offsetInit, r_maxval, stream);

    offsetInit->outputToFile("offsetInit1", stream);

    // Estimation of statistics
    // Author: Minyan Zhong
    // Extraction of correlation surface around the peak
    cuArraysCopyExtractCorr(r_corrBatchRaw, r_corrBatchRawZoomIn, i_corrBatchZoomInValid, offsetInit, stream);

    cudaDeviceSynchronize();

    // debug: output the intermediate results
    r_maxval->outputToFile("r_maxval",stream);
    r_corrBatchRaw->outputToFile("r_corrBatchRaw",stream);
    r_corrBatchRawZoomIn->outputToFile("r_corrBatchRawZoomIn",stream);
    i_corrBatchZoomInValid->outputToFile("i_corrBatchZoomInValid",stream);

    // Summation of correlation and data point values
    cuArraysSumCorr(r_corrBatchRawZoomIn, i_corrBatchZoomInValid, r_corrBatchSum, i_corrBatchValidCount, stream);

    // SNR
    cuEstimateSnr(r_corrBatchSum, i_corrBatchValidCount, r_maxval, r_snrValue, stream);

    // Variance
    // cuEstimateVariance(r_corrBatchRaw, offsetInit, r_maxval, r_covValue, stream);

    // Using the approximate estimation to adjust secondary image (half search window size becomes only 4 pixels)
    //offsetInit->debuginfo(stream);
    // determine the starting pixel to extract secondary images around the max location
    cuDetermineSecondaryExtractOffset(offsetInit,
        param->halfSearchRangeDownRaw, // old range
        param->halfSearchRangeAcrossRaw,
        param->halfZoomWindowSizeRaw,  // new range
        param->halfZoomWindowSizeRaw,
        stream);
    //offsetInit->debuginfo(stream);
    // oversample reference
    // (deramping now included in oversampler)
    referenceBatchOverSampler->execute(c_referenceBatchRaw, c_referenceBatchOverSampled, param->derampMethod);
    cuArraysAbs(c_referenceBatchOverSampled, r_referenceBatchOverSampled, stream);
    cuArraysSubtractMean(r_referenceBatchOverSampled, stream);

    // extract secondary and oversample
    cuArraysCopyExtract(c_secondaryBatchRaw, c_secondaryBatchZoomIn, offsetInit, stream);
    secondaryBatchOverSampler->execute(c_secondaryBatchZoomIn, c_secondaryBatchOverSampled, param->derampMethod);
    cuArraysAbs(c_secondaryBatchOverSampled, r_secondaryBatchOverSampled, stream);

    // correlate oversampled images
    if(param->algorithm == 0) {
        cuCorrFreqDomain_OverSampled->execute(r_referenceBatchOverSampled, r_secondaryBatchOverSampled, r_corrBatchZoomIn);
    }
    else {
        cuCorrTimeDomain(r_referenceBatchOverSampled, r_secondaryBatchOverSampled, r_corrBatchZoomIn, stream);
    }
    cuCorrNormalize(r_referenceBatchOverSampled, r_secondaryBatchOverSampled, r_corrBatchZoomIn, stream);

    //std::cout << "debug correlation oversample\n";
    //std::cout << r_referenceBatchOverSampled->height << " " << r_referenceBatchOverSampled->width << "\n";
    //std::cout << r_secondaryBatchOverSampled->height << " " << r_secondaryBatchOverSampled->width << "\n";
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

    // Do insertion.
    // Offsetfields.
    cuArraysCopyInsert(offsetFinal, offsetImage, idxDown_*param->numberWindowDownInChunk, idxAcross_*param->numberWindowAcrossInChunk,stream);

    // Debugging matrix.
    cuArraysCopyInsert(r_corrBatchSum, floatImage1, idxDown_*param->numberWindowDownInChunk, idxAcross_*param->numberWindowAcrossInChunk,stream);
    cuArraysCopyInsert(i_corrBatchValidCount, intImage1, idxDown_*param->numberWindowDownInChunk, idxAcross_*param->numberWindowAcrossInChunk,stream);

    // Old: save max correlation coefficients.
    //cuArraysCopyInsert(corrMaxValue, snrImage, idxDown_*param->numberWindowDownInChunk, idxAcross_*param->numberWindowAcrossInChunk,stream);
    // New: save SNR
    cuArraysCopyInsert(r_snrValue, snrImage, idxDown_*param->numberWindowDownInChunk, idxAcross_*param->numberWindowAcrossInChunk,stream);

    // Variance.
    cuArraysCopyInsert(r_covValue, covImage, idxDown_*param->numberWindowDownInChunk, idxAcross_*param->numberWindowAcrossInChunk,stream);
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

void cuAmpcorChunk::loadReferenceChunk()
{

    // we first load the whole chunk of image from cpu to a gpu buffer c(r)_referenceChunkRaw
    // then copy to a batch of windows with (nImages, height, width) (leading dimension on the right)

    // get the chunk size to be loaded to gpu
    int startD = param->referenceChunkStartPixelDown[idxChunk]; //start pixel down (along height)
    int startA = param->referenceChunkStartPixelAcross[idxChunk]; // start pixel across (along width)
    int height =  param->referenceChunkHeight[idxChunk]; // number of pixels along height
    int width = param->referenceChunkWidth[idxChunk];  // number of pixels along width

    //use cpu to compute the starting positions for each window
    getRelativeOffset(ChunkOffsetDown->hostData, param->referenceStartPixelDown, param->referenceChunkStartPixelDown[idxChunk]);
    // copy the positions to gpu
    ChunkOffsetDown->copyToDevice(stream);
    // same for the across direction
    getRelativeOffset(ChunkOffsetAcross->hostData, param->referenceStartPixelAcross, param->referenceChunkStartPixelAcross[idxChunk]);
    ChunkOffsetAcross->copyToDevice(stream);

    // check whether the image is complex (e.g., SLC) or real( e.g. TIFF)
    if(referenceImage->isComplex())
    {
        // allocate a gpu buffer to load data from cpu/file
        // try allocate/deallocate the buffer on the fly to save gpu memory 07/09/19
        c_referenceChunkRaw = new cuArrays<float2> (param->maxReferenceChunkHeight, param->maxReferenceChunkWidth);
        c_referenceChunkRaw->allocate();

        // load the data from cpu
        referenceImage->loadToDevice((void *)c_referenceChunkRaw->devData, startD, startA, height, width, stream);
        //std::cout << "debug load reference: " << startD << " " <<  startA << " " <<  height << " "  << width << "\n";

        //copy the chunk to a batch format (nImages, height, width)
        // if derampMethod = 0 (no deramp), take amplitudes; otherwise, copy complex data
        if(param->derampMethod == 0) {
            cuArraysCopyToBatchAbsWithOffset(c_referenceChunkRaw, param->referenceChunkWidth[idxChunk],
                c_referenceBatchRaw, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
        }
        else {
            cuArraysCopyToBatchWithOffset(c_referenceChunkRaw, param->referenceChunkWidth[idxChunk],
                c_referenceBatchRaw, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
        }
        // deallocate the gpu buffer
        delete c_referenceChunkRaw;
    }
    // if the image is real
    else {
        r_referenceChunkRaw = new cuArrays<float> (param->maxReferenceChunkHeight, param->maxReferenceChunkWidth);
        r_referenceChunkRaw->allocate();

        // load the data from cpu
        referenceImage->loadToDevice((void *)r_referenceChunkRaw->devData, startD, startA, height, width, stream);

        // copy the chunk (real) to a batch format (complex)
        cuArraysCopyToBatchWithOffsetR2C(r_referenceChunkRaw, param->referenceChunkWidth[idxChunk],
                c_referenceBatchRaw, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
        // deallocate the gpu buffer
        delete r_referenceChunkRaw;
    }


}

void cuAmpcorChunk::loadSecondaryChunk()
{

    //copy to a batch format (nImages, height, width)
    getRelativeOffset(ChunkOffsetDown->hostData, param->secondaryStartPixelDown, param->secondaryChunkStartPixelDown[idxChunk]);
    ChunkOffsetDown->copyToDevice(stream);
    getRelativeOffset(ChunkOffsetAcross->hostData, param->secondaryStartPixelAcross, param->secondaryChunkStartPixelAcross[idxChunk]);
    ChunkOffsetAcross->copyToDevice(stream);

    if(secondaryImage->isComplex())
    {
        c_secondaryChunkRaw = new cuArrays<float2> (param->maxSecondaryChunkHeight, param->maxSecondaryChunkWidth);
        c_secondaryChunkRaw->allocate();

        //load a chunk from mmap to gpu
        secondaryImage->loadToDevice(c_secondaryChunkRaw->devData,
            param->secondaryChunkStartPixelDown[idxChunk],
            param->secondaryChunkStartPixelAcross[idxChunk],
            param->secondaryChunkHeight[idxChunk],
            param->secondaryChunkWidth[idxChunk],
            stream);

        if(param->derampMethod == 0) {
            cuArraysCopyToBatchAbsWithOffset(c_secondaryChunkRaw, param->secondaryChunkWidth[idxChunk],
                c_secondaryBatchRaw, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
        }
        else {
           cuArraysCopyToBatchWithOffset(c_secondaryChunkRaw, param->secondaryChunkWidth[idxChunk],
                c_secondaryBatchRaw, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
        }
        delete c_secondaryChunkRaw;
    }
    else { //real image
        //allocate the gpu buffer
        r_secondaryChunkRaw = new cuArrays<float> (param->maxSecondaryChunkHeight, param->maxSecondaryChunkWidth);
        r_secondaryChunkRaw->allocate();

        //load a chunk from mmap to gpu
        secondaryImage->loadToDevice(r_secondaryChunkRaw->devData,
            param->secondaryChunkStartPixelDown[idxChunk],
            param->secondaryChunkStartPixelAcross[idxChunk],
            param->secondaryChunkHeight[idxChunk],
            param->secondaryChunkWidth[idxChunk],
            stream);

        // convert to the batch format
        cuArraysCopyToBatchWithOffsetR2C(r_secondaryChunkRaw, param->secondaryChunkWidth[idxChunk],
                c_secondaryBatchRaw, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
        delete r_secondaryChunkRaw;
    }
}

cuAmpcorChunk::cuAmpcorChunk(cuAmpcorParameter *param_, GDALImage *reference_, GDALImage *secondary_,
    cuArrays<float2> *offsetImage_, cuArrays<float> *snrImage_, cuArrays<float3> *covImage_, cuArrays<int> *intImage1_, cuArrays<float> *floatImage1_, cudaStream_t stream_)

{
    param = param_;
    referenceImage = reference_;
    secondaryImage = secondary_;
    offsetImage = offsetImage_;
    snrImage = snrImage_;
    covImage = covImage_;

    intImage1 = intImage1_;
    floatImage1 = floatImage1_;

    stream = stream_;

    // std::cout << "debug Chunk creator " << param->maxReferenceChunkHeight << " " << param->maxReferenceChunkWidth << "\n";
    // try allocate/deallocate on the fly to save gpu memory 07/09/19
    // c_referenceChunkRaw = new cuArrays<float2> (param->maxReferenceChunkHeight, param->maxReferenceChunkWidth);
    // c_referenceChunkRaw->allocate();

    // c_secondaryChunkRaw = new cuArrays<float2> (param->maxSecondaryChunkHeight, param->maxSecondaryChunkWidth);
    // c_secondaryChunkRaw->allocate();

    ChunkOffsetDown = new cuArrays<int> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    ChunkOffsetDown->allocate();
    ChunkOffsetDown->allocateHost();
    ChunkOffsetAcross = new cuArrays<int> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    ChunkOffsetAcross->allocate();
    ChunkOffsetAcross->allocateHost();

    c_referenceBatchRaw = new cuArrays<float2> (
        param->windowSizeHeightRaw, param->windowSizeWidthRaw,
        param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    c_referenceBatchRaw->allocate();

    c_secondaryBatchRaw = new cuArrays<float2> (
        param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
        param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    c_secondaryBatchRaw->allocate();

    r_referenceBatchRaw = new cuArrays<float> (
        param->windowSizeHeightRaw, param->windowSizeWidthRaw,
        param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    r_referenceBatchRaw->allocate();

    r_secondaryBatchRaw = new cuArrays<float> (
        param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
        param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    r_secondaryBatchRaw->allocate();

    c_secondaryBatchZoomIn = new cuArrays<float2> (
        param->searchWindowSizeHeightRawZoomIn, param->searchWindowSizeWidthRawZoomIn,
        param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    c_secondaryBatchZoomIn->allocate();

    c_referenceBatchOverSampled = new cuArrays<float2> (
			param->windowSizeHeight, param->windowSizeWidth,
			param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    c_referenceBatchOverSampled->allocate();

    c_secondaryBatchOverSampled = new cuArrays<float2> (
			param->searchWindowSizeHeight, param->searchWindowSizeWidth,
			param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    c_secondaryBatchOverSampled->allocate();

    r_referenceBatchOverSampled = new cuArrays<float> (
			param->windowSizeHeight, param->windowSizeWidth,
			param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    r_referenceBatchOverSampled->allocate();

    r_secondaryBatchOverSampled = new cuArrays<float> (
			param->searchWindowSizeHeight, param->searchWindowSizeWidth,
			param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    r_secondaryBatchOverSampled->allocate();

    referenceBatchOverSampler = new cuOverSamplerC2C(
        c_referenceBatchRaw->height, c_referenceBatchRaw->width, //orignal size
        c_referenceBatchOverSampled->height, c_referenceBatchOverSampled->width, //oversampled size
        c_referenceBatchRaw->count, stream);

    secondaryBatchOverSampler = new cuOverSamplerC2C(c_secondaryBatchZoomIn->height, c_secondaryBatchZoomIn->width,
            c_secondaryBatchOverSampled->height, c_secondaryBatchOverSampled->width, c_secondaryBatchRaw->count, stream);

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


    // new arrays due to snr estimation
    std::cout<< "corrRawZoomInHeight: " << param->corrRawZoomInHeight << "\n";
    std::cout<< "corrRawZoomInWidth: " << param->corrRawZoomInWidth << "\n";

    r_corrBatchRawZoomIn = new cuArrays<float> (
			param->corrRawZoomInHeight,
			param->corrRawZoomInWidth,
			param->numberWindowDownInChunk,
			param->numberWindowAcrossInChunk);
    r_corrBatchRawZoomIn->allocate();

    i_corrBatchZoomInValid = new cuArrays<int> (
			param->corrRawZoomInHeight,
			param->corrRawZoomInWidth,
			param->numberWindowDownInChunk,
			param->numberWindowAcrossInChunk);
    i_corrBatchZoomInValid->allocate();


    r_corrBatchSum = new cuArrays<float> (
                    param->numberWindowDownInChunk,
                    param->numberWindowAcrossInChunk);
    r_corrBatchSum->allocate();

    i_corrBatchValidCount = new cuArrays<int> (
                        param->numberWindowDownInChunk,
                        param->numberWindowAcrossInChunk);
    i_corrBatchValidCount->allocate();

    i_maxloc = new cuArrays<int2> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);

    i_maxloc->allocate();

    r_maxval = new cuArrays<float> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);

    r_maxval->allocate();

    r_snrValue = new cuArrays<float> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);

    r_snrValue->allocate();

    r_covValue = new cuArrays<float3> (param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);

    r_covValue->allocate();

    // end of new arrays

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
    delete referenceChunkRaw;
    delete secondaryChunkRaw;
    delete ChunkOffsetDown;
    delete ChunkOffsetAcross;
    delete referenceBatchRaw;
    delete secondaryBatchRaw;
    delete referenceChunkOverSampled;
    delete secondaryChunkOverSampled;
    delete referenceChunkOverSampler;
    delete secondaryChunkOverSampler;
    delete referenceChunk;
    delete secondaryChunk;
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
