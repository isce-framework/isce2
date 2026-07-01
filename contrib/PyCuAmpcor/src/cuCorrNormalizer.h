/*
 * @file cuNormalizer.h
 * @brief normalize the correlation surface
 *
 * cuNormalizeProcessor is an abstract class for processors to normalize the correlation surface.
 * It has different implementations wrt different image sizes.
 * cuNormalizeFixed<64/128/.../1024> use a shared memory accelerated algorithm, which are limited by the number of cuda threads in a block.
 * cuNormalizeSAT uses the sum area table based algorithm, which applies to any size (used for >1024).
 * cuNormalizer is a wrapper class which determines which processor to use.
 */

#ifndef __CUNORMALIZER_H
#define __CUNORMALIZER_H

#include "cuArrays.h"
#include "data_types.h"

/**
 * Abstract class interface for correlation surface normalization processor
 * with different implementations
 */
class cuNormalizeProcessor {
public:
    // default constructor and destructor
    cuNormalizeProcessor() = default;
    virtual ~cuNormalizeProcessor() = default;
    // execute interface
    virtual void execute(cuArrays<real_type> * correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream) = 0;
};

// factory with the secondary dimension
cuNormalizeProcessor* newCuNormalizer(int NX, int NY, int count);


template<int Size>
class cuNormalizeFixed : public cuNormalizeProcessor
{
public:
    void execute(cuArrays<real_type> * correlation, cuArrays<real_type> *reference, cuArrays<real_type> *search, cudaStream_t stream) override;
};

class cuNormalizeSAT : public cuNormalizeProcessor
{
private:
    cuArrays<real_type> *referenceSum2;
    cuArrays<real_type> *secondarySAT;
    cuArrays<real_type> *secondarySAT2;

public:
    cuNormalizeSAT(int secondaryNX, int secondaryNY, int count);
    ~cuNormalizeSAT();
    void execute(cuArrays<real_type> * correlation, cuArrays<real_type> *reference, cuArrays<real_type> *search, cudaStream_t stream) override;
};

#endif
// end of file
