#include "ImageAccessor.h"
#include <exception>
#include <cmath>
#include <complex>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
void ImageAccessor::createFile(int  fileLength)
{
    LineAccessor::createFile(&fileLength);
}
void ImageAccessor::finalizeImageAccessor()
{
    LineAccessor::finalizeLineAccessor();
}

void ImageAccessor::getElements(char * dataLine, int * row, int * col, int  numEl)
{
    LineAccessor::getElements(dataLine, row,  col,  &numEl);
}
	
void ImageAccessor::getLineSequential(char * dataLine, int  & eof)
{
    LineAccessor::getLineSequential(dataLine, &eof);
}
void ImageAccessor::getSequentialElements(char * dataLine, int   row, int   col, int & numEl)
{
    LineAccessor::getSequentialElements(dataLine, &row, &col, &numEl);
}
void ImageAccessor::getLine(char * dataLine, int & row)
{
    LineAccessor::getLine(dataLine, &row);
}
void ImageAccessor::initImageAccessor(string filename, string filemode, char endianFile, string type, int col, int row)
{
    LineAccessor::initLineAccessor(filename,filemode,endianFile,type,row,col);
}

void ImageAccessor::initSequentialAccessor(int  begLine)
{

    LineAccessor::initSequentialAccessor(&begLine);
}
void ImageAccessor::setElements(char * dataLine, int * row, int * col, int  numEl)
{
    LineAccessor::setElements(dataLine, row, col, &numEl);

}
void ImageAccessor::setLine(char * dataLine, int  row)
{
    LineAccessor::setLine(dataLine, &row);
}
void ImageAccessor::setLineSequential(char * dataLine)
{
    LineAccessor::setLineSequential(dataLine);
}
void ImageAccessor::setSequentialElements(char * dataLine, int  row, int  col, int  numEl)
{
    LineAccessor::setSequentialElements(dataLine,&row,&col,&numEl);
}
