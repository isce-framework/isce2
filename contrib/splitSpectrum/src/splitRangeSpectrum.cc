/**\file splitRangeSpectrum.cc
 * \author Heresh Fattahi.
 *  */
#include "splitRangeSpectrum.h"
#include <gdal.h>
#include <gdal_priv.h>
#include <iostream>
#include <complex>
#include <vector>
#include <fftw3.h>
#include <math.h>
#include "common.h"

using namespace std;

const float PI = std::acos(-1);
const std::complex<float> I(0, 1);

using splitSpectrum::splitRangeSpectrum;


typedef std::vector< std::complex<float> > cpxVec;
typedef std::vector< float > fltVec;

int bandPass(GDALDataset* slcSubDataset, cpxVec &spectrum, cpxVec &spectrumSub, cpxVec &slcSub, int ind1, int ind2, int width, int inysize, int cols, int yoff, fltVec rangeTime, float subBandFrequency)
{

    // Band-pass filetr
    // Copy the spectrum of the High band and center the High Band to center 
    // frequency of the sub-band (demodulation is applied)
    // First copy the right half of the sub-band spectrum to elements [0:n/2] 
    // where n = ind2 - ind1
    // Since the spectrum obtained with FFTW3 is not normalized, we normalize 
    // it by dividing the real and imaginary parts by the length of the signal 
    // for each line, which is cols.

    //int kk,jj,ii;
    //#pragma omp parallel for\
    //    default(shared)
    //for (kk = 0; kk< inysize * (ind2 - ind1 - width); kk++)
    //{
    //    jj = kk/(ind2 -ind1 - width);
    //    ii = kk % (ind2-ind1-width) + ind1 + width;

    //    spectrumSub[jj*cols + ii - ind1 - width] = spectrum[ii+jj*cols]/(1.0f*cols);
    //}

    // Then copy the left part part to elements [N-n/2:N] where N is the length 
    // of the signal (N=cols)
    //#pragma omp parallel for\
    //    default(shared)
    //for (kk=0; kk<inysize*width; kk++)
    //{
    //    jj = kk/width;
    //    ii = kk%width + ind1;
    //    spectrumSub[jj*cols + ii + cols - width - ind1] = spectrum[ii+jj*cols]/(1.0f*cols);
    //}
    
    //Extracting the sub-band spectrum
    int kk,jj,ii;
    #pragma omp parallel for\
        default(shared)
    for (kk = 0; kk < inysize*(ind2-ind1); kk++)
    {
        jj = kk/(ind2-ind1);
        ii = kk % (ind2-ind1) + ind1;
        spectrumSub[ii+jj*cols] = spectrum[ii+jj*cols]/(1.0f*cols);
    }


    // A plan for inverse fft of the sub-band spectrum
    fftwf_plan planInverse = fftwf_plan_many_dft(1, &cols, inysize,
                                        (fftwf_complex *) (&spectrumSub[0]), &cols,
                                        1, cols,
                                        (fftwf_complex *) (&slcSub[0]), &cols,
                                        1, cols,
                                        FFTW_BACKWARD, FFTW_ESTIMATE);

    fftwf_execute(planInverse);
    fftwf_destroy_plan(planInverse);

    // demodulate the sub band slc to center the sub band spectrum
    #pragma omp parallel for\
        default(shared)
    for (kk = 0; kk < inysize*cols; kk++)
    {
        jj = kk/cols;
        ii = kk % cols;
        slcSub[ii+jj*cols] = slcSub[ii+jj*cols]*(std::exp(-1.0f*I*2.0f*PI*subBandFrequency*rangeTime[ii]));
    }

        // writing the HB SLC to file using GDAL
    int status;
    status = slcSubDataset->GetRasterBand(1)->RasterIO( GF_Write, 0, yoff,
                                cols, inysize,
                                (void*) (&slcSub[0]),
                                cols, inysize, GDT_CFloat32,
                                sizeof(std::complex<float>),
                                sizeof(std::complex<float>)*cols, NULL);
    return(0);

}

int index_frequency(double B, int N, double f)
// deterrmine the index (n) of a given frequency f
// B: bandwidth, N: length of a signal
// Assumption: for indices 0 to (N-1)/2, frequency is positive 
//              and for indices larger than (N-1)/2 frequency is negative
{
    // index of a given frequency f
    int n;
    // frequency interval
    double df = B/N;

    if (f < 0)
        n = round(f/df + N);
    else
        n = round(f/df);
    return n;
}

float frequency (double B, int N, int n)
// calculates frequency at a given index.
// Assumption: for indices 0 to (N-1)/2, frequency is positive 
// and for indices larger than (N-1)/2 frequency is negative
{
    //frequency interval given B as the total bandwidth
    double f, df = B/N;
    int middleIndex = ((N-1)/2);

    if (n > middleIndex)
        f = (n-N)*df;
    else
        f = n*df;

    return f;
}

float adjustCenterFrequency(double B, int N, double dc)
{
    
    // because of quantization, there may not be an index representing dc. We 
    // therefore adjust dc to make sure that there is an index to represent it. 
    // We find the index that is closest to nominal dc and then adjust dc to the 
    // frequency of that index.
    // B = full band-width
    // N = length of signal
    // dc = center frequency of the sub-band

    int ind;
    double df = B/N;
    if (dc < 0){
        ind = N+round(dc/df);
    }
    else{
        ind = round(dc/df);
    }
    dc = frequency (B, N, ind);

    return dc;
}


void splitRangeSpectrum::setInputDataset(std::string inDataset)
{
    // set input dataset which is the full-band SLC
    inputDS = inDataset;
}

void splitRangeSpectrum::setLowbandDataset(std::string inLbDataset, std::string inHbDataset)
{
    // set output datasets which are two SLCs at low-band and high-band
    // low-band dataset
    lbDS = inLbDataset;

    // high-band dataset
    hbDS = inHbDataset;
}



void splitRangeSpectrum::setMemorySize(int inMemSize)
{
    // set memory size
    memsize = inMemSize;
}

void splitRangeSpectrum::setBlockSize(int inBlockSize)
{
    // set block size (number of lines to be read as one block)
    blocksize = inBlockSize; 
}

void splitRangeSpectrum::setBandwidth(double fs, double lBW, double hBW)
{
    // set the range sampling rate and the band-width of low-band and high-band SLC
    rangeSamplingRate = fs;
    lowBandWidth = lBW;
    highBandWidth = hBW;
}



void splitRangeSpectrum::setSubBandCenterFrequencies(double fl, double fh)
{
    // set center frequencies of low-band and high-band SLCs
    lowCenterFrequency = fl;
    highCenterFrequency = fh;
}



//int split_spectrum_process(splitOptions *opts)
//{

int splitRangeSpectrum::split_spectrum_process()
{
    // Print user options to screen
    //opts->print();
    
    // cols: number of columns of the SLC
    // rows: number of lines of the SLC
    int cols, rows;
    int blockysize;
    int nbands;

    // Define NULL GDAL datasets for input full-band SLC and out put sub-band SLCs
    GDALDataset* slcDataset = NULL;
    GDALDataset* slcLBDataset = NULL;
    GDALDataset* slcHBDataset = NULL;

    // Clock variables
    double t_start, t_end;

    // Register GDAL drivers
    GDALAllRegister();
    slcDataset = reinterpret_cast<GDALDataset *>( GDALOpenShared( inputDS.c_str(), GA_ReadOnly));
    
    if (slcDataset == NULL)
    {
        std::cout << "Cannot open SLC file { " << inputDS << "}" << endl;
        std::cout << "GDALOpen failed - " << inputDS << endl;
        std::cout << "Exiting with error code .... (102) \n";
        GDALDestroyDriverManager();
        return 102;
    }

    cols = slcDataset->GetRasterXSize();
    rows = slcDataset->GetRasterYSize();
    nbands = slcDataset->GetRasterCount();


    // Determine blocksizes
    // Number of vectors = 6
    // Memory for one pixel CFloat32 = 8 byte
    // cols=number of columns in one line
    blockysize = int((memsize * 1.0e6)/(cols * 8 * 6) );
    std::cout << "Computed block size based on memory size = " << blockysize << " lines \n";

    // if (blockysize < opts->blocksize)
    //    blockysize = opts->blocksize;


    std::cout << "Block size = " << blockysize << " lines \n";
    int totalblocks = ceil( rows / (1.0 * blockysize));
    std::cout << "Total number of blocks to process: " << totalblocks << "\n";
    // Start the clock
    t_start = getWallTime();

    // Array for reading complex SLC data
    std::vector< std::complex<float> > slc(cols*blockysize);
    // Array for spectrum of full band SLC
    std::vector< std::complex<float> > spectrum(cols*blockysize);
    // Array for spectrum of low-band SLC  
    std::vector< std::complex<float> > spectrumLB(cols*blockysize);
    // Array for spectrum of high-band SLC
    std::vector< std::complex<float> > spectrumHB(cols*blockysize);
    // Array for low-band SLC
    std::vector< std::complex<float> > slcLB(cols*blockysize);
    //Array for high-band SLC
    std::vector< std::complex<float> > slcHB(cols*blockysize);

    //vector for range time
    std::vector< float > rangeTime(cols);
   
    // populating vector of range time for one line
    for (int ii = 0; ii < cols; ii++)
    {
        rangeTime[ii] = ii/rangeSamplingRate;
    }

    // Start block-by-block processing
    int blockcount = 0;
    int status;
    // number of threads
    int nthreads;
    nthreads = numberOfThreads();
    
    // creating output datasets
    GDALDriver *poDriver = (GDALDriver*) GDALGetDriverByName("ENVI");
    char **mOptions = NULL;
    mOptions = CSLSetNameValue(mOptions, "INTERLEAVE", "BIL");
    mOptions = CSLSetNameValue(mOptions, "SUFFIX", "ADD");

    // creating output datasets for low-band SLC
    slcLBDataset  = (GDALDataset*) poDriver->Create(lbDS.c_str(), cols, rows, 1, GDT_CFloat32, mOptions);

    if (slcLBDataset == NULL)
    {
        std::cout << "Could not create meanamp dataset {" << lbDS << "} \n";
        std::cout << "Exiting with non-zero error code ... 104 \n";

        GDALClose(slcDataset);
        GDALDestroyDriverManager();
        return 104;
    } 

    // creating output datasets for high-band SLC
    slcHBDataset  = (GDALDataset*) poDriver->Create(hbDS.c_str(), cols, rows, 1, GDT_CFloat32, mOptions);
    if (slcHBDataset == NULL)
    {
        std::cout << "Could not create meanamp dataset {" << lbDS << "} \n";
        std::cout << "Exiting with non-zero error code ... 104 \n";

        GDALClose(slcDataset);
        GDALDestroyDriverManager();
        return 104;
    }  
    
    CSLDestroy(mOptions);

    float highBand [2];
    float lowBand [2];

    cout << "sub-band center frequencies after adjustment: " << endl;
    cout << "low-band: " << lowCenterFrequency << endl;
    cout << "high-band: " << highCenterFrequency << endl;

    lowBand[0] = lowCenterFrequency - lowBandWidth/2.0;
    lowBand[1] = lowBand[0] + lowBandWidth;

    highBand[0] = highCenterFrequency - highBandWidth/2.0;
    highBand[1] = highBand[0] + highBandWidth;

    // defining the pixel number of the high-band
    int indH1, indH2, widthHB;

    // index of the lower bound of the high-band
    indH1 = index_frequency(rangeSamplingRate, cols, highBand[0]);

    // index of the upper bound of the High Band
    indH2 = index_frequency(rangeSamplingRate, cols, highBand[1]);

    // width of the subband (unit pixels)
    widthHB = (indH2 - indH1)/2;


    // defining the pixel number of the low-band
    int indL1, indL2, widthLB;

    // index of the lower bound of the low-band
    indL1 = index_frequency(rangeSamplingRate, cols, lowBand[0]);

    // index of the upper bound of the low-band
    indL2 = index_frequency(rangeSamplingRate, cols, lowBand[1]);

    // width of the subband (unit pixels)
    widthLB = (indL2 - indL1)/2;

    // Block-by-block processing
    int fftw_status;
    fftw_status = fftwf_init_threads();
    cout << "fftw_status: " << fftw_status;

    for (int yoff=0; yoff < rows; yoff += blockysize)
    {
        // Increment block counter
        blockcount++;

        // Determine number of rows to read
        int inysize = blockysize;
        if ((yoff+inysize) > rows)
            inysize = rows - yoff;

        // Read a block of the SLC data to cpxdata array
        status = slcDataset->GetRasterBand(1)->RasterIO( GF_Read, 0, yoff,
                                cols, inysize,
                                (void*) (&slc[0]),
                                cols, inysize, GDT_CFloat32,
                                sizeof(std::complex<float>),
                                sizeof(std::complex<float>)*cols, NULL);

        
        // creating the forward 1D fft plan for inysize lines of SLC data. 
        // Each fft is applied on multiple lines of SLC data.
        fftwf_plan_with_nthreads(nthreads);
        fftwf_plan plan = fftwf_plan_many_dft(1, &cols, inysize,
                                        (fftwf_complex *) (&slc[0]), &cols,
                                        1, cols,
                                        (fftwf_complex *) (&spectrum[0]), &cols,
                                        1, cols,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
        
        // execute the fft plan
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);

        // bandpass the spectrum for low-band, demodulate to center the sub-band 
        // spectrum, and normalize the sub-band spectrum 
        bandPass(slcLBDataset, spectrum, spectrumLB, slcLB, indL1, indL2, widthLB, inysize, cols, yoff, rangeTime, lowCenterFrequency);
        // bandpass the spectrum for high-band, demodulate to center the sub-band 
        // spectrum, and normalize the sub-band spectrum
        bandPass(slcHBDataset, spectrum, spectrumHB, slcHB, indH1, indH2, widthHB, inysize, cols, yoff, rangeTime, highCenterFrequency);
        
                
    }

    t_end = getWallTime();
    
    std::cout << "splitSpectrum processing time: " << (t_end-t_start)/60.0 << " mins \n";
    //close the datasets
    GDALClose(slcDataset);
    GDALClose(slcLBDataset);
    GDALClose(slcHBDataset);

    return (0);
};

