#include <iostream>
#include <fstream>
#include "byteswap.h"
#include "Burst.hh"

Burst::Burst(int rangeSamples,int azimuthSamples,bool isBigEndian) 
{
    this->isBigEndian = isBigEndian;
    this->azimuthSamples = azimuthSamples;
    this->rangeSamples = rangeSamples;
    this->asri = new int[this->rangeSamples];
    this->asfv = new int[this->rangeSamples];
    this->aslv = new int[this->rangeSamples];
}

Burst::~Burst() 
{
    delete [] this->asri;
    delete [] this->asfv;
    delete [] this->aslv;
}

    void
Burst::parse(std::istream &fin,std::ostream &fout)
{
    this->parseAzimuthHeader(fin);
    for(int i=0;i<this->azimuthSamples;i++)
    {
        if ((i % 1000) == 0)
        {
            std::cout << "Parsing Line " << i << std::endl;
        }
        this->parseRangeLine(fin,fout,i);
    }
}

    void
Burst::parseAzimuthHeader(std::istream &fin)
{
    // For each of the three azimuth header lines, skip the first two 4-byte samples
    // Read 'Range Samples' number of 4 byte integers 
    fin.seekg(8, std::ios_base::cur);
    fin.read((char *)(this->asri),this->rangeSamples*sizeof(int));
    // again
    fin.seekg(8, std::ios_base::cur);
    fin.read((char *)(this->asfv),this->rangeSamples*sizeof(int));
    // and again
    fin.seekg(8, std::ios_base::cur);
    fin.read((char *)(this->aslv),this->rangeSamples*sizeof(int));

    if (!this->isBigEndian)
    {
        // Byte swap
        for(int i=0;i<this->rangeSamples;i++)
        {
            this->asri[i] = bswap_32(this->asri[i]);
            this->asfv[i] = bswap_32(this->asfv[i])-1;
            this->aslv[i] = bswap_32(this->aslv[i])-1;
        }
    }
}
    void
Burst::parseRangeLine(std::istream &fin,std::ostream &fout,int lineNumber)
{
    short *data;
    int rsfv,rslv;
    int asfv,aslv;
    float *floatData;

    data = new short[2*this->rangeSamples];
    floatData = new float[2*this->rangeSamples];

    // Read line header
    fin.read((char*)(&rsfv),sizeof(int));
    fin.read((char*)(&rslv),sizeof(int));
    if (!this->isBigEndian)
    {
        // Byte swap
        rsfv = bswap_32(rsfv)-1;
        rslv = bswap_32(rslv)-1;
    }
    // Read data
    fin.read((char*)(data),2*this->rangeSamples*sizeof(short));
    // Byte swap data and mask out invalid points
    for(int rangeBin=0,j=0;rangeBin<this->rangeSamples;rangeBin++,j+=2)
    {
        asfv = this->asfv[rangeBin];
        aslv = this->aslv[rangeBin];
        // gdal_translate
        if ((lineNumber < asfv) || (lineNumber > aslv) || (rangeBin < rsfv) || (rangeBin >= rslv))
        {
            floatData[j] = 0.0;
            floatData[j+1] = 0.0;
        }
        else
        {
            if (!this->isBigEndian)
            {
                data[j] = bswap_16(data[j]);
                data[j+1] = bswap_16(data[j+1]);
            }
            floatData[j] = (float)data[j];
            floatData[j+1] = (float)data[j+1];
        }
    }
    fout.write((char*)floatData,2*this->rangeSamples*sizeof(float));

    delete [] data;
    delete [] floatData;
}
