#include <iostream>
#include "DataAccessorCaster.h"

using namespace std;
int DataAccessorCaster::getLine(char * buf, int pos)
{
   ////// REMEMBER THAT getData might change the forth argument ////////////
    int width = LineWidth;
    char  * dataLine  = new char[DataSizeIn*Bands*LineWidth]; 
    Accessor->getData(dataLine,pos,LineOffset,width);
    Caster->convert(dataLine,buf,LineWidth*Bands);
    delete [] dataLine;
    return  Accessor->getEofFlag();
}
int DataAccessorCaster::getLineBand(char * buf, int pos, int band)
{
    int width = LineWidth;
    char *dataLine = new char[DataSizeIn*LineWidth];
    Accessor->getDataBand(dataLine,pos,0,width, band);
    Caster->convert(dataLine,buf,LineWidth);
    delete [] dataLine;
    return Accessor->getEofFlag();
}
void DataAccessorCaster::setLine(char * buf, int pos)
{
    char  * dataLine  = new char[DataSizeOut*Bands*LineWidth]; 
    Caster->convert(buf,dataLine,LineWidth*Bands);
    Accessor->setData(dataLine,pos,LineOffset,LineWidth);
    delete [] dataLine;
}
void DataAccessorCaster::setLineBand(char * buf, int pos, int band)
{
    char * dataLine = new char[DataSizeOut*LineWidth];
    Caster->convert(buf, dataLine, LineWidth);
    Accessor->setDataBand(dataLine, pos, 0, LineWidth, band);
    delete [] dataLine;
}

void DataAccessorCaster::getSequentialElements(char * buf, int row, int col, int & numEl)
{
    char  * dataLine  = new char[DataSizeIn*Bands*numEl]; 
    Accessor->getData(dataLine,row,col,numEl);
    Caster->convert(dataLine,buf,numEl*Bands);
    delete [] dataLine;
}
void DataAccessorCaster::setSequentialElements(char * buf, int row, int col, int numEl)
{
    char  * dataLine  = new char[DataSizeOut*Bands*numEl]; 
    Caster->convert(buf,dataLine,numEl*Bands);
    Accessor->setData(dataLine,row,col,numEl);
    delete [] dataLine;
}
void DataAccessorCaster::setLineSequential(char * buf)
{
    char  * dataLine  = new char[DataSizeOut*Bands*LineWidth]; 
    Caster->convert(buf,dataLine,LineWidth*Bands);
    Accessor->setData(dataLine,LineCounter,0,LineWidth);
    ++LineCounter;
    delete [] dataLine;

}
void DataAccessorCaster::setLineSequentialBand(char * buf, int band)
{
    char * dataLine = new char[DataSizeOut*LineWidth];
    Caster->convert(buf, dataLine, LineWidth);
    Accessor->setDataBand(dataLine, LineCounter, 0, LineWidth, band);
    ++LineCounter;
    delete [] dataLine;
}
int DataAccessorCaster::getLineSequential(char * buf)
{
    int width = LineWidth;
    char  * dataLine  = new char[DataSizeIn*Bands*LineWidth]; 
    Accessor->getData(dataLine,LineCounter,LineOffset,width);
    Caster->convert(dataLine,buf,LineWidth*Bands);
    ++LineCounter;
    delete [] dataLine;
    return  Accessor->getEofFlag();
}
int DataAccessorCaster::getLineSequentialBand(char * buf, int band)
{
    int width = LineWidth;
    char * dataLine = new char[DataSizeIn*LineWidth];
    Accessor->getDataBand(dataLine, LineCounter, 0, width, band);
    Caster->convert(dataLine, buf, LineWidth);
    ++LineCounter;
    delete [] dataLine;
    return Accessor->getEofFlag();
}
void DataAccessorCaster::getStream(char * buf,  int  & numEl)
{
    char  * dataLine  = new char[DataSizeIn*numEl]; 
    Accessor->getStream(dataLine,numEl);
    Caster->convert(dataLine,buf,numEl);
    delete [] dataLine;
}
void DataAccessorCaster::getStreamAtPos(char * buf, int & pos,  int & numEl)
{
    char  * dataLine  = new char[DataSizeIn*numEl]; 
    Accessor->getStreamAtPos(dataLine,pos,numEl);
    Caster->convert(dataLine,buf,numEl);
    delete [] dataLine;

}
void DataAccessorCaster::setStream(char * buf,  int & numEl)
{
    char  * dataLine  = new char[DataSizeOut*numEl]; 
    Caster->convert(buf,dataLine,numEl);
    Accessor->setStream(dataLine,numEl);
    delete [] dataLine;

}
void DataAccessorCaster::setStreamAtPos(char * buf, int &  pos,  int & numEl)
{
    char  * dataLine  = new char[DataSizeOut*numEl]; 
    Caster->convert(buf,dataLine,numEl);
    Accessor->setStreamAtPos(dataLine,pos,numEl);
    delete [] dataLine;

}
void DataAccessorCaster::finalize()
{
    Accessor->finalize();
    delete Accessor;
    delete Caster;

}
double DataAccessorCaster::getPx2d(int row, int col)
{
  return 0.;
}
double DataAccessorCaster::getPx1d(int pos)
{
  return 0.;
}
