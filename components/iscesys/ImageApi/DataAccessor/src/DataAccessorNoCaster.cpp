#include <iostream>
#include "DataAccessorNoCaster.h"

using namespace std;
int DataAccessorNoCaster::getLine(char * buf, int pos)
{
   ////// REMEMBER THAT getData might change the forth argument ////////////
    int width = LineWidth;
    Accessor->getData(buf,pos,0,width);
    return  Accessor->getEofFlag();
}

int DataAccessorNoCaster::getLineBand(char * buf, int pos, int band)
{
    int width = LineWidth;
    Accessor->getDataBand(buf, pos, 0, width, band);
    return Accessor->getEofFlag();
}
void DataAccessorNoCaster::setLine(char * buf, int pos)
{
    Accessor->setData(buf,pos,0,LineWidth);
}
void DataAccessorNoCaster::setLineBand(char * buf, int pos, int band)
{
    Accessor->setDataBand(buf, pos, 0, LineWidth, band);
}
void DataAccessorNoCaster::getSequentialElements(char * buf, int row, int col, int & numEl)
{
    Accessor->getData(buf,row,col,numEl);
}
void DataAccessorNoCaster::setSequentialElements(char * buf, int row, int col, int numEl)
{
    Accessor->setData(buf,row,col,numEl);
}
void DataAccessorNoCaster::setLineSequential(char * buf)
{
    Accessor->setData(buf,LineCounter,0,LineWidth);
    ++LineCounter;

}
void DataAccessorNoCaster::setLineSequentialBand(char * buf, int band)
{
    Accessor->setDataBand(buf, LineCounter,0,LineWidth,band);
    ++LineCounter;
}
int DataAccessorNoCaster::getLineSequential(char * buf)
{
    int width = LineWidth;
    Accessor->getData(buf,LineCounter,0,width);
    ++LineCounter;
    return  Accessor->getEofFlag();
}
int DataAccessorNoCaster::getLineSequentialBand(char * buf,int band)
{
    int width = LineWidth;
    Accessor->getDataBand(buf, LineCounter,0,width, band);
    ++LineCounter;
    return Accessor->getEofFlag();
}
void DataAccessorNoCaster::getStream(char * buf,  int  & numEl)
{
    Accessor->getStream(buf,numEl);
}
void DataAccessorNoCaster::getStreamAtPos(char * buf, int & pos,  int & numEl)
{
    Accessor->getStreamAtPos(buf,pos,numEl);

}
void DataAccessorNoCaster::setStream(char * buf,  int & numEl)
{
    Accessor->setStream(buf,numEl);

}
void DataAccessorNoCaster::setStreamAtPos(char * buf, int &  pos,  int & numEl)
{
    Accessor->setStreamAtPos(buf,pos,numEl);

}
void DataAccessorNoCaster::finalize()
{
    Accessor->finalize();
    delete Accessor;

}
double DataAccessorNoCaster::getPx2d(int row, int col)
{
    double ret = 0;
    int numEl = 1;
    //NOTE: the forth arg is a reference so we cannot put just the number 1
    Accessor->getData((char *)&ret,row,col,numEl);

    return ret;
}
double DataAccessorNoCaster::getPx1d(int pos)
{
    double ret = 0;
    int numEl = 1;
    //NOTE: the forth arg is a reference so we cannot put just the number 1
    Accessor->getData((char *)&ret,0,pos,numEl);

    return ret;
}
