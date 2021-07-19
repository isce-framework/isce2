/*
 * @file cuNormalizer.cu
 * @brief processors to normalize the correlation surface
 *
 */

#include "cuCorrNormalizer.h"
#include "cuAmpcorUtil.h"

cuNormalizer::cuNormalizer(int secondaryNX, int secondaryNY, int count)
{
    // depending on NY, choose different processor
    if(secondaryNY <= 64) {
        processor = new cuNormalizeFixed<64>();
    }
    else if (secondaryNY <= 128) {
        processor = new cuNormalizeFixed<128>();
    }
    else if (secondaryNY <= 256) {
        processor = new cuNormalizeFixed<256>();
    }
    else if (secondaryNY <= 512) {
        processor = new cuNormalizeFixed<512>();
    }
    else if (secondaryNY <= 1024) {
        processor = new cuNormalizeFixed<1024>();
    }
    else {
        processor = new cuNormalizeSAT(secondaryNX, secondaryNY, count);
    }
}

cuNormalizer::~cuNormalizer()
{
    delete processor;
}

void cuNormalizer::execute(cuArrays<float> *correlation,
    cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream)
{
    processor->execute(correlation, reference, secondary, stream);
}

/**
 *
 *
 **/

cuNormalizeSAT::cuNormalizeSAT(int secondaryNX, int secondaryNY, int count)
{
    // allocate the work array
    // reference sum square
    referenceSum2 = new cuArrays<float>(1, 1, count);
    referenceSum2->allocate();

    // secondary sum and sum square
    secondarySAT = new cuArrays<float>(secondaryNX, secondaryNY, count);
    secondarySAT->allocate();
    secondarySAT2 = new cuArrays<float>(secondaryNX, secondaryNY, count);
    secondarySAT2->allocate();
};

cuNormalizeSAT::~cuNormalizeSAT()
{
    delete referenceSum2;
    delete secondarySAT;
    delete secondarySAT2;
}

void cuNormalizeSAT::execute(cuArrays<float> *correlation,
    cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream)
{
    cuCorrNormalizeSAT(correlation, reference, secondary,
        referenceSum2, secondarySAT, secondarySAT2, stream);
}

template<int Size>
void cuNormalizeFixed<Size>::execute(cuArrays<float> *correlation,
    cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream)
{
    cuCorrNormalizeFixed<Size>(correlation, reference, secondary, stream);
}

template class cuNormalizeFixed<64>;
template class cuNormalizeFixed<128>;
template class cuNormalizeFixed<256>;
template class cuNormalizeFixed<512>;
template class cuNormalizeFixed<1024>;

// end of file
