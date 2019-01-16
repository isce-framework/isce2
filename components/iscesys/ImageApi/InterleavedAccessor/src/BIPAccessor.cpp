#include <iostream>
#include <fstream>
#include "BIPAccessor.h"

using namespace std;
void
BIPAccessor::init(void * poly)
{
  return;
}
void
BIPAccessor::setData(char * buf, int row, int col, int numEl)
{
  streampos posNow = ((streampos) row * LineWidth * Bands * SizeV) + ((streampos) col * Bands * SizeV);
  FileObject.seekp(posNow);
  FileObject.write(buf, numEl * Bands * SizeV);
  //the good flag gets set but not the eof for some reason, so assume eof when good is set false
  if (!FileObject.good())
  {
    NoGoodFlag = 1;
    EofFlag = -1;
  }
}
int cnt = 0;
void
BIPAccessor::setDataBand(char* buf, int row, int col, int numEl, int band)
{
  streampos posNow = ((streampos) row * LineWidth * Bands * SizeV)
      + ((streampos) col * Bands * SizeV); //+ (streampos) band * SizeV;

  if (Bands > 1)
  {


    char * dataLine = new char[numEl * Bands * SizeV];
    FileObject.seekg(posNow);
    FileObject.read(dataLine, numEl * Bands * SizeV);
    FileObject.seekp(posNow);
    for (int i = 0; i < numEl; ++i)
    {

      for (int j = 0; j < SizeV; ++j)
      {
        dataLine[i * SizeV * Bands + band * SizeV + j] = buf[i * SizeV + j];
      }

    }
    FileObject.write(dataLine, numEl * Bands * SizeV);

  }
  else
  {
    setData(buf, row, col, numEl);
  }
}

void
BIPAccessor::getData(char * buf, int row, int col, int & numEl)
{
  streampos posNow = ((streampos) row * LineWidth * Bands * SizeV)
      + ((streampos) col * Bands * SizeV);
  FileObject.seekg(posNow);
  FileObject.read(buf, numEl * Bands * SizeV);
  numEl = FileObject.gcount() / (SizeV * Bands);

  //the good flag gets set but not the eof for some reason, so assume eof when good is set false
  if (!FileObject.good())
  {
    NoGoodFlag = 1;
    EofFlag = -1;
  }
}
void
BIPAccessor::getDataBand(char * buf, int row, int col, int &numEl, int band)
{

  int actualRead = 0;
  streampos posNow = ((streampos) row * LineWidth * Bands * SizeV)
      + ((streampos) col * Bands * SizeV);

  if (Bands > 1)
  {
    char * dataLine = new char[numEl * Bands * SizeV];
    FileObject.seekg(posNow);
    FileObject.read(dataLine, numEl * Bands * SizeV);
    actualRead = FileObject.gcount() / (Bands * SizeV);
    for (int i = 0; i < numEl; ++i)
    {
      for (int j = 0; j < SizeV; ++j)
      {
        buf[i * SizeV + j] = dataLine[i * SizeV * Bands + band * SizeV + j];
      }
    }

    numEl = actualRead;

    if (!FileObject.good())
    {
      NoGoodFlag = 1;
      EofFlag = -1;
    }
  }
  else
  {
//        std::cout << "Line = " << row << " Offset = " << posNow << std::endl;
    actualRead = numEl;
    getData(buf, row, col, actualRead);
    numEl = actualRead;
  }
}
