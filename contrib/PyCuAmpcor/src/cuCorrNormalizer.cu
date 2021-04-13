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
        processor = new cuNormalize64();
    }
    else if (secondaryNY <= 128) {
        processor = new cuNormalize128();
    }
    else if (secondaryNY <= 256) {
        processor = new cuNormalize256();
    }
    else if (secondaryNY <= 512) {
        processor = new cuNormalize512();
    }
    else if (secondaryNY <= 1024) {
        processor = new cuNormalize1024();
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

void cuNormalize64::execute(cuArrays<float> *correlation,
    cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream)
{
    cuCorrNormalize64(correlation, reference, secondary, stream);
}

void cuNormalize128::execute(cuArrays<float> *correlation,
    cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream)
{
    cuCorrNormalize128(correlation, reference, secondary, stream);
}

void cuNormalize256::execute(cuArrays<float> *correlation,
    cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream)
{
    cuCorrNormalize256(correlation, reference, secondary, stream);
}

void cuNormalize512::execute(cuArrays<float> *correlation,
    cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream)
{
    cuCorrNormalize512(correlation, reference, secondary, stream);
}

void cuNormalize1024::execute(cuArrays<float> *correlation,
    cuArrays<float> *reference, cuArrays<float> *secondary, cudaStream_t stream)
{
    cuCorrNormalize1024(correlation, reference, secondary, stream);
}

// end of file