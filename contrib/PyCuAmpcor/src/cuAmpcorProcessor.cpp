#include "cuAmpcorProcessor.h"
#include "cuAmpcorProcessorROIPAC.h"
#include "cuAmpcorProcessorGrIMP.h"


// Factory method implementation
// create the batch processor for a given {workflow}
std::unique_ptr<cuAmpcorProcessor> cuAmpcorProcessor::create(int workflow,
    cuAmpcorParameter *param_,
    GDALImage *reference_, GDALImage *secondary_,
    cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_,
    cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
    cudaStream_t stream_)
{
    if (workflow == 0) {
        return std::unique_ptr<cuAmpcorProcessor>(new cuAmpcorProcessorROIPAC(
            param_, reference_, secondary_, offsetImage_,
            snrImage_, covImage_, peakValueImage_, stream_));
    } else if (workflow == 1) {
        return std::unique_ptr<cuAmpcorProcessor>(new cuAmpcorProcessorGrIMP(
            param_, reference_, secondary_, offsetImage_,
            snrImage_, covImage_, peakValueImage_, stream_));
    } else {
        throw std::invalid_argument("Unsupported workflow");
    }
}

// constructor
cuAmpcorProcessor::cuAmpcorProcessor(cuAmpcorParameter *param_,
        GDALImage *reference_, GDALImage *secondary_,
        cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_,
        cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
        cudaStream_t stream_)
    : param(param_), referenceImage(reference_), secondaryImage(secondary_),
    offsetImage(offsetImage_), snrImage(snrImage_), covImage(covImage_),
    peakValueImage(peakValueImage_), stream(stream_)
{
}

/// set chunk index
void cuAmpcorProcessor::setIndex(int idxDown_, int idxAcross_)
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
}

/// obtain the starting pixels for each chip
/// @param[in] oStartPixel start pixel locations for all chips
/// @param[out] rstartPixel  start pixel locations for chips within the chunk
void cuAmpcorProcessor::getRelativeOffset(int *rStartPixel, const int *oStartPixel, int diff)
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
        }
    }
}

