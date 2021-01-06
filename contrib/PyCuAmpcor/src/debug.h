/**
 * @file debug.h
 * @brief Define flags to control the debugging
 *
 * CUAMPCOR_DEBUG is used to output debugging information and intermediate results,
 *    disabled when NDEBUG macro is defined.
 * CUDA_ERROR_CHECK is always enabled, to check CUDA routine errors
 *
 */

// code guard
#ifndef __CUAMPCOR_DEBUG_H
#define __CUAMPCOR_DEBUG_H

#ifndef NDEBUG
#define CUAMPCOR_DEBUG
#endif //NDEBUG

#define CUDA_ERROR_CHECK

#endif //__CUAMPCOR_DEBUG_H
//end of file