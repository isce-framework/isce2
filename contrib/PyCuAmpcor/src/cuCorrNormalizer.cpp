/*
 * @file cuNormalizer.cu
 * @brief processors to normalize the correlation surface
 *
 */

#include "cuCorrNormalizer.h"
#include "cuAmpcorUtil.h"

cuNormalizeProcessor*
newCuNormalizer(int secondaryNX, int secondaryNY, int count)
{
    // depending on NY, choose different processor
    if(secondaryNY <= 64) {
        return new cuNormalizeFixed<64>();
    }
    else if (secondaryNY <= 128) {
        return new cuNormalizeFixed<128>();
    }
    else if (secondaryNY <= 256) {
        return new cuNormalizeFixed<256>();
    }
    else if (secondaryNY <= 512) {
        return new cuNormalizeFixed<512>();
    }
    else if (secondaryNY <= 1024) {
        return new cuNormalizeFixed<1024>();
    }
    else {
        return new cuNormalizeSAT(secondaryNX, secondaryNY, count);
    }
}

cuNormalizeSAT::cuNormalizeSAT(int secondaryNX, int secondaryNY, int count)
{
    // allocate the work array
    // reference sum square
    referenceSum2 = new cuArrays<real_type>(1, 1, count);
    referenceSum2->allocate();

    // secondary sum and sum square
    secondarySAT = new cuArrays<real_type>(secondaryNX, secondaryNY, count);
    secondarySAT->allocate();
    secondarySAT2 = new cuArrays<real_type>(secondaryNX, secondaryNY, count);
    secondarySAT2->allocate();
};

cuNormalizeSAT::~cuNormalizeSAT()
{
    delete referenceSum2;
    delete secondarySAT;
    delete secondarySAT2;
}

void cuNormalizeSAT::execute(cuArrays<real_type> *correlation,
    cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream)
{
    cuCorrNormalizeSAT(correlation, reference, secondary,
        referenceSum2, secondarySAT, secondarySAT2, stream);
}

template<int Size>
void cuNormalizeFixed<Size>::execute(cuArrays<real_type> *correlation,
    cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream)
{
    cuCorrNormalizeFixed<Size>(correlation, reference, secondary, stream);
}

template class cuNormalizeFixed<64>;
template class cuNormalizeFixed<128>;
template class cuNormalizeFixed<256>;
template class cuNormalizeFixed<512>;
template class cuNormalizeFixed<1024>;

// end of file
