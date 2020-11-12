#ifndef __DEBUG_H
#define __DEBUG_H

#include <iostream>
#include <assert.h>
#include <stdio.h>

#ifndef NDEBUG
#define CUAMPCOR_DEBUG
#define debugmsg(msg) fprintf(stderr, msg)
#else
#define debugmsg(msg)
#endif //NDEBUG

#define CUDA_ERROR_CHECK

#endif //__DEBUG_H
