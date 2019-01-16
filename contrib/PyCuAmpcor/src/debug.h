#ifndef __DEBUG_H
#define __DEBUG_H

#pragma once

#include <iostream>
#include <assert.h>

#define _DEBUG_ 1

#define CUDA_ERROR_CHECK

#define debugmsg(msg) if(_DEBUG_) fprintf(stderr, msg)

//__CUDA_ARCH__

#endif 
