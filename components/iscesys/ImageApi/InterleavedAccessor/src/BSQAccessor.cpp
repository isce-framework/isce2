#include <iostream>
#include <fstream>
#include "BSQAccessor.h"

void BSQAccessor::init(void * poly)
{
  return;
}
void BSQAccessor::setData(char * buf, int row, int col, int numEl)
{
    char * dataLine = new char[numEl*Bands*SizeV];
    for(int i = 0; i < numEl; ++i)
    {
        for(int j = 0; j < Bands; ++j)
        {
            for(int k = 0; k < SizeV; ++k)
            {
                dataLine[i*SizeV + j*SizeV*numEl + k] = buf[i*Bands*SizeV + j*SizeV + k];
            }

        }
    }
    for(int i = 0; i < Bands; ++i)
    {
      streampos posNow = (streampos) NumberOfLines*LineWidth*SizeV*i +  (streampos) row*LineWidth*SizeV + (streampos) col*SizeV;
        FileObject.seekp(posNow);
        FileObject.write(&dataLine[i*numEl*SizeV],numEl*SizeV);
        //the good flag gets set but not the eof for some reason, so assume eof when good is set false
        if(!FileObject.good())
        {
            NoGoodFlag = 1;
            EofFlag = -1;
        }

    }
    
}
void BSQAccessor::setDataBand(char * buf, int row, int col, int numEl, int band)
{
    streampos posNow = ((streampos)NumberOfLines*LineWidth*SizeV*band) + ((streampos) row*LineWidth*SizeV) + ((streampos) col*SizeV);
    FileObject.seekp(posNow);
    FileObject.write(buf, numEl*SizeV);

    if(!FileObject.good())
    {
        NoGoodFlag = 1;
        EofFlag = -1;
    }
}

void BSQAccessor::getData(char * buf, int row, int col, int & numEl)
{
     
    char * dataLine = new char[numEl*Bands*SizeV];
    int actualRead = 0;
    for(int i = 0; i < Bands; ++i)
    {
      streampos posNow = ((streampos)NumberOfLines*LineWidth*SizeV*i) +  ((streampos)row*LineWidth*SizeV + (streampos) col*SizeV);
        FileObject.seekg(posNow);
        FileObject.read(&dataLine[i*numEl*SizeV],numEl*SizeV);
        actualRead = FileObject.gcount()/(SizeV);

        //the good flag gets set but not the eof for some reason, so assume eof when good is set false
        if(!FileObject.good())
        {
            NoGoodFlag = 1;
            EofFlag = -1;
        }
    }
    numEl = actualRead; //if one of the reads is different, then something went wrong
    for(int i = 0; i < numEl; ++i)
    {
        for(int j = 0; j < Bands; ++j)
        {
            for(int k = 0; k < SizeV; ++k)
            {
                buf[i*Bands*SizeV + j*SizeV + k] = dataLine[i*SizeV + j*SizeV*numEl + k];
            }

        }
    }
}
void BSQAccessor::getDataBand(char * buf, int row, int col, int &numEl, int band)
{
    streampos posNow = ((streampos)NumberOfLines*LineWidth*SizeV*band) + ((streampos)row*LineWidth*SizeV) + ((streampos) col*SizeV);
    FileObject.seekg(posNow);
    FileObject.read(buf, numEl*SizeV);
    numEl = FileObject.gcount()/(SizeV);

    if(!FileObject.good())
    {
        NoGoodFlag = 1;
        EofFlag = -1;
    }
}

