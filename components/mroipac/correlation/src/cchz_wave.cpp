#include <vector>
#include <cmath>
#include <iostream>
#include <complex>
#include "DataAccessor.h"

int cchz_wave(int flag, DataAccessor* intAcc, DataAccessor* ampAcc, DataAccessor* corAcc, int bx)
{
    /* interferogram line buffer, complex input data, row pointers */
    std::complex<float> *bufcz;
    std::complex<float> *cmpb;
    std::complex<float> *tc;
    std::complex<float> **cmp;

    double a1,ai1,ai2,ar,ai;
    double *rw;              /* range correlation weights */
    double *azw;             /* azimuth correlation weights */

    float *output;           /* output arrays - BIP amp and coh*/
    float cor;
    std::complex<float> *ib1,*t1;       /* image intensities buffers */
    std::complex<float> **i1;           /* pointers to 2 image intensities lines */
    float wt;                /* product of range and azimuth weights */
    std::complex<float> dt1, dt2;       /* temp variables to go from slcs to int-amp */

    int width, nlines;
    int xw,yh;               /* width, height of processed region */
    int i,j,k,n;             /* loop counters */
    int icnt;                /* line counter */
    int nrw,nazw;            /* size of filter windows in range, azimuth */

    int xmin, xmax, ymin, ymax;


    if (flag)
        std::cout <<  "Operating on coregistered SLCs. \n";
    else
        std::cout << "Operating on ROI_PAC style int and amp files. \n";


    nlines = min(intAcc->getNumberOfLines(), ampAcc->getNumberOfLines());
    width = intAcc->getWidth();
    if(ampAcc->getWidth() != width)
    {
        std::cout << "Input image width's dont match. Exiting ... \n";
        exit(-1);
    }


    std::cout << "Number of lines : " << nlines << "\n";
    std::cout << "Number of pixels: " << width << "\n";

    //Just checking and setting default if a bad value is provided.
    if (bx <=0 ) {
        std::cout << "No default box size provided.. Setting to default value of 3 \n";
        bx = 3;
    }
    else {
        std::cout << "Processing with box size of " << bx << "\n";
    }

    xmin = 0;
    xmax = width-1;
    ymin = 0;
    ymax = nlines-1;

    if (xmax <= 0) {
        xmax=width-1;
    }

    xw = width;
    yh = nlines;

    bufcz = new std::complex<float>[width];
    cmpb  = new std::complex<float>[width*bx];
    cmp = new std::complex<float>*[bx];


    if (bufcz==NULL || cmpb==NULL ||  cmp==NULL){
        std::cout << "failure to allocate space for complex data buffers!\n";
        exit(-1);
    }

    ib1   = new std::complex<float>[width*bx];
    i1    = new std::complex<float>*[bx];
    output = new float[2*width];

    if (ib1==NULL ||  i1==NULL || output ==NULL){
        std::cout << "failure to allocate space for memory buffers!\n";
        exit(-1);
    }

    nrw=bx;
    nazw=bx;
    std::cout << "# correlation weights (range,azimuth): " << nrw << " " << nazw << "\n";

    rw = new double[nrw];
    azw = new double[nazw];

    if(rw == NULL || azw == NULL) {
        std::cout<< "ERROR: memory allocation for correlation weights failed!\n";
        exit(-1);
    }

    std::cout << "\nrange correlation weights:\n";
    for(j=0; j < nrw; j++){
        rw[j]=1.0-std::fabs(2.0*(double)(j-nrw/2)/(bx+1));
        std::cout << "index,coefficient: " << j-nrw/2 <<"  " << rw[j] << "\n";
    }

    std::cout << "\nazimuth correlation weights:\n";
    for(j=0; j < nazw; j++){
        azw[j]=1.0-std::fabs(2.0*(double)(j-nazw/2)/(bx+1));
        std::cout << "index,coefficient:  " << j-nazw/2 <<"  " << azw[j] << "\n";
    }

    for(j=0; j < width; j++)
    {
        bufcz[j] = 0.0;
    }

    for(j=0; j < width*bx; j++)
    {
        ib1[j]=0.0;
    }

    for(i=0; i < bx; i++)
    {        /* initialize array pointers */
        cmp[i] = &(cmpb[i*width]);
        i1[i]  = &(ib1[i*width]);
    }


    for(i=0;i<(2*width);i++)
    {
        output[i] = 0.0;
    }


    icnt = 0;
    for(i=0;i<(bx/2);i++)
    {
        corAcc->setLineSequential((char*) output);
        icnt++;
    }

    /* Read bx-1 lines of each image */
    for(i=0; i<bx-1;i++)
    {
        intAcc->getLineSequential((char*)(&(cmpb[i*width])));
    }

    for(i=0;i<bx-1;i++)
    {
        ampAcc->getLineSequential((char*)(&(ib1[i*width])));
    }


    /* PSA - Quick fix for coregistered SLC */
    if (flag)  /* To convert from slcs to int-amp */
    {
        for(i=0; i < (bx-1); i++)
        {
            for(j=0; j < width; j++)
            {
                dt1 = cmp[i][j];
                dt2 = i1[i][j];

                cmp[i][j] = conj(dt1) * dt2;
       
                i1[i][j] = abs(dt1) + 1i * abs(dt2);
            }
        }
    }

    for (i=bx/2; i < (yh-bx/2); i++)
    {
        if(i%10 == 0)
            std::cout << "\rprocessing line: " <<  i;


        intAcc->getLineSequential((char*) (cmp[bx-1]));
        ampAcc->getLineSequential((char*) (i1[bx-1]));

        if (flag)  /* To convert from slcs to int-amp */
        {
            for(j=0; j < width; j++)
            {
                dt1 = cmp[bx-1][j];
                dt2 = i1[bx-1][j];

                cmp[bx-1][j] = conj(dt1) * dt2;
                i1[bx-1][j] = abs(dt1) + 1i * abs(dt2);
            } 
        }

        /* move across the image j=xmin+bx/2 to j=width-bx/2-1 (xmin=0, xw=width)*/
        for (j=xmin+bx/2; j < xw-bx/2; j++)
        {
            ai1=0.0;
            ai2=0.0;
            ar=0.0;
            ai=0.0;

            /* average over the box */
            for (k=0; k < bx; k++)
            {
                for (n=j-bx/2; n < j-bx/2+bx; n++)
                {
                    wt=azw[k]*rw[n-j+bx/2];
                    ai1 += pow(i1[k][n].real(),2)*wt;
                    ai2 += pow(i1[k][n].imag(),2)*wt;
                    ar  += cmp[k][n].real()*wt;
                    ai  += cmp[k][n].imag()*wt;
                }
            }

            a1=sqrt(ai1*ai2);
            output[2*j]=sqrt((double)i1[bx/2][j].real()*(double)i1[bx/2][j].imag()) ;
            /* renormalized correlation coefficient */
            if (a1 > 0.0)
                cor = (float)hypot(ar,ai)/a1;
            else 
                cor=0.0;
            output[2*j+1]=min(cor,1.0f);

        }

        
        corAcc->setLineSequential((char*) output);
        icnt++;

        /* buffer circular shift */
        /* save pointer addresses of the oldest line */
        t1=i1[0]; tc=cmp[0];
        /* shift addresses */
        for (k=1; k < bx; k++)
        {
            i1[k-1]=i1[k];
            cmp[k-1]=cmp[k];
        }
        /* new data will overwrite the oldest */
        i1[bx-1]=t1; cmp[bx-1]=tc;
    }

    for(j=0; j<(2*xw);j++)
    {
        output[j] = 0.0;
    }

    for(j=0; j < (bx/2); j++)
    {
        corAcc->setLineSequential((char*) output);
        icnt++;
    }

    std::cout << "\noutput lines:" << icnt << "\n";
  
    delete [] rw;
    delete [] azw;
    delete [] ib1;
    delete [] i1;
    delete [] output;
    delete [] bufcz;
    delete [] cmp;
    delete [] cmpb;
    return(0);
}

