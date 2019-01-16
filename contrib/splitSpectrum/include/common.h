#ifndef SPLITSPECTRUM_COMMON_H
#define SPLITSPECTRUM_COMMON_H

#include <time.h>
#include <sys/time.h>
#include <omp.h>


double getWallTime()
{
    struct timeval time;
    if (gettimeofday(&time,NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


int numberOfThreads()
{
    int nthreads = 0;

    #pragma omp parallel
    if (omp_get_thread_num() == 1)
    {
        nthreads = omp_get_num_threads();
    }

    std::cout << "Processing with " << nthreads << " threads \n";

    if (nthreads == 0)
    {
        std::cout << "Looks like the code was not linked with openmp. \n";
        std::cout << "Recompile with the right linker flags. \n";
        throw;
    }
    
    if (nthreads == 1)
    {
        std::cout << "This code has been designed to work with multiple threads. \n";
        std::cout << "Looks like sequential execution due to compilation or environment settings. \n";
        std::cout << "Check your settings for optimal performace. Continuing ... \n";
    }

    return nthreads;
}

#endif //SPLITSPECTRUM_COMMON_H

