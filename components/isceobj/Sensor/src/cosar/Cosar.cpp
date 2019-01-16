#include <iostream>
#include <fstream>
#include "byteswap.h"
#include "Cosar.hh"
#include <stdlib.h>

// Cosar files are big-endian
// thus, we need to determine the endianness of the machine we are on
// and decide whether we need to swap bytes
Cosar::Cosar(std::string input, std::string output)
{
    // Check the endianness
    if (is_big_endian() == 1) 
    {
        std::cout << "Machine is Big Endian" << std::endl;
        this->isBigEndian = true;
    }
    else
    {
        std::cout << "Machine is Little Endian" << std::endl;
        this->isBigEndian = false;
    }
    this->fin.open(input.c_str(), std::ios::binary | std::ios::in);
    if (fin.fail())
    { 
        std::cout << "Error in file " << __FILE__ << " at line " << __LINE__ << std::endl;
        std::cout << "Cannot open file " << input << std::endl ;
        exit(1);
    }   
    this->fout.open(output.c_str(), std::ios::binary | std::ios::out);
    if (fout.fail())
    {
        std::cout << "Error in file " << __FILE__ << " at line " << __LINE__ << std::endl;
        std::cout << "Cannot open file " << input << std::endl ;
        exit(1);
    }
    try {
        this->header = new Header(this->isBigEndian);
    } catch(const char *ex) {
        throw;
    }
}

Cosar::~Cosar()
{
    this->fin.close();
    this->fout.close();
}

    void
Cosar::parse()
{
    this->header->parse(this->fin);
    this->header->print();
    int byteTotal = this->header->getRangelineTotalNumberOfBytes();
    int numLines = this->header->getTotalNumberOfLines();
    int burstSize = this->header->getBytesInBurst();
    int rangeSamples = this->header->getRangeSamples();
    int azimuthSamples = this->header->getAzimuthSamples();

    std::cout << "Image is " << azimuthSamples << " x " << rangeSamples << std::endl;
    this->numberOfBursts = (int)(byteTotal*numLines)/burstSize;
    this->bursts = new Burst*[this->numberOfBursts];
    for(int i=0;i<this->numberOfBursts;i++)
    {
        std::cout << "Extracting Burst " << (i+1) << " of " << this->numberOfBursts << std::endl;
        this->bursts[i] = new Burst(rangeSamples,azimuthSamples,this->isBigEndian);
        this->bursts[i]->parse(this->fin,this->fout);
    }
}
