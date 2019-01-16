#include "LineAccessorF.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

// these functions allow the fortran code to use the member functions of the LineAccessor objects
void getLineAccessorObject_f(uint64_t * ptLineAccessor)
{
	LineAccessor * tmp =  new LineAccessor;
	(* ptLineAccessor) = (uint64_t ) tmp;
}
void  getMachineEndianness_f(uint64_t * ptLineAccessor, char * endian)
{
	endian[0] = ((LineAccessor * )(* ptLineAccessor))->getMachineEndianness();
}
void initLineAccessor_f(uint64_t * ptLineAccessor, char * filename, char * filemode, char * endianFile, char * type, int * row, int * col, long int filenameLength, long int filemodeLength, long int pass, long int typeLength)
{

	string filenameStr = getString(filename,filenameLength);
	string filemodeStr = getString(filemode,filemodeLength);
	string typeStr = getString(type,typeLength);
	((LineAccessor * )(* ptLineAccessor))->initLineAccessor(filenameStr,filemodeStr,(*endianFile),typeStr,(*row),(*col));
	
}
void changeBandScheme_f(uint64_t * ptLineAccessor, char * filein, char * fileout, char * type, int * width, int * numBands, int * bandIn, int * bandOut, long int fileinLength, long int fileoutLength, long int typeLength)
{
	string fileinStr = getString(filein,fileinLength);
	string fileoutStr = getString(fileout,fileoutLength);
	string typeStr = getString(type,typeLength);
	BandSchemeType bIn = convertIntToBandSchemeType((*bandIn));	
	BandSchemeType bOut = convertIntToBandSchemeType((*bandOut));	
	((LineAccessor * )(* ptLineAccessor))->changeBandScheme(fileinStr, fileoutStr, typeStr, (*width),(*numBands), bIn, bOut);

}
void convertFileEndianness_f(uint64_t * ptLineAccessor, char * filein, char * fileout, char * type, long int fileinLength, long int fileoutLength, long int typeLength)
{
    string fileinStr = getString(filein,fileinLength);
    string fileoutStr = getString(fileout,fileoutLength);
    string typeStr = getString(type,typeLength);
    ((LineAccessor * )(* ptLineAccessor))->convertFileEndianness(fileinStr, fileoutStr, typeStr);
}
void finalizeLineAccessor_f(uint64_t * ptLineAccessor)
{	
	((LineAccessor * )(* ptLineAccessor))->finalizeLineAccessor();
	LineAccessor * tmp = (LineAccessor *) (* ptLineAccessor);
	delete tmp;
}
void createFile_f(uint64_t * ptLineAccessor, int * length)
{
	((LineAccessor * ) (* ptLineAccessor))->createFile(length);
}

void rewindImage_f(uint64_t * ptLineAccessor)
{
	((LineAccessor * ) (* ptLineAccessor))->rewindImage();
}

void getTypeSize_f(uint64_t *  ptLineAccessor, char * type, int * size, long int len)
{
	string typeStr = getString(type,len);
	(*size) = ((LineAccessor * ) (* ptLineAccessor))->getTypeSize(typeStr);
}
void getFileLength_f(uint64_t *  ptLineAccessor, int * length)
{
	((LineAccessor * ) (* ptLineAccessor))->getFileLength(length);
}
void getFileWidth_f(uint64_t *  ptLineAccessor, int * lineWidth)
{
	((LineAccessor * ) (* ptLineAccessor))->getFileWidth(lineWidth);
}
void printObjectInfo_f(uint64_t *  ptLineAccessor)
{
	((LineAccessor * ) (* ptLineAccessor))->printObjectInfo();
}
void printAvailableDataTypesAndSizes_f(uint64_t *  ptLineAccessor)
{
	((LineAccessor * ) (* ptLineAccessor))->printAvailableDataTypesAndSizes();
}
void initSequentialAccessor_f(uint64_t * ptLineAccessor, int * begLine)
{
	((LineAccessor * ) (* ptLineAccessor))->initSequentialAccessor(begLine);
}

void getLine_f(uint64_t * ptLineAccessor,  char * dataLine, int * ptLine) 
{
    ((LineAccessor * ) (* ptLineAccessor))->getLine(dataLine, ptLine);
}

void getLineSequential_f(uint64_t * ptLineAccessor,  char * dataLine, int * ptLine) 
{
    ((LineAccessor * ) (* ptLineAccessor))->getLineSequential(dataLine, ptLine);
}
void setLine_f(uint64_t * ptLineAccessor,  char * dataLine, int * ptLine) 
{
    ((LineAccessor * ) (* ptLineAccessor))->setLine(dataLine, ptLine);
}
void setLineSequential_f(uint64_t * ptLineAccessor,  char * dataLine) 
{
    ((LineAccessor * ) (* ptLineAccessor))->setLineSequential(dataLine);
}

void setStream_f(uint64_t * ptLineAccessor,  char * dataLine, int * numEl)
{
	((LineAccessor * ) (* ptLineAccessor))->setStream(dataLine, numEl);
}
void setStreamAtPos_f(uint64_t * ptLineAccessor,  char * dataLine, int * pos, int * numEl)
{
	((LineAccessor * ) (* ptLineAccessor))->setStreamAtPos(dataLine, pos, numEl);
}
void getStream_f(uint64_t * ptLineAccessor,  char * dataLine, int * numEl)
{
	((LineAccessor * ) (* ptLineAccessor))->getStream(dataLine, numEl);
}
void getStreamAtPos_f(uint64_t * ptLineAccessor,  char * dataLine, int * pos, int * numEl)
{
	((LineAccessor * ) (* ptLineAccessor))->getStreamAtPos(dataLine, pos, numEl);
}
void getElements_f(uint64_t * ptLineAccessor,  char * dataLine, int * row, int * col, int * numEl)
{
	((LineAccessor * ) (* ptLineAccessor))->getElements(dataLine, row,  col,  numEl);
}
void setElements_f(uint64_t * ptLineAccessor,  char * dataLine, int * row, int * col, int * numEl)
{
	((LineAccessor * ) (* ptLineAccessor))->setElements(dataLine, row,  col,  numEl);
}
void getSequentialElements_f(uint64_t * ptLineAccessor,  char * dataLine, int * row, int * col, int * numEl)
{
	((LineAccessor * ) (* ptLineAccessor))->getSequentialElements(dataLine, row,  col,  numEl);
}
void setSequentialElements_f(uint64_t * ptLineAccessor,  char * dataLine, int * row, int * col, int * numEl)
{
	((LineAccessor * ) (* ptLineAccessor))->setSequentialElements(dataLine, row,  col,  numEl);
}

string getString(char * word, long int len)
{
    int i = len - 1;
    string retStr;
    while(word[i] == ' ')
    {
	--i;
    }
    int count = i;
    while(i >= 0)
    {
	retStr += word[count - i];
	--i;
    }
    return retStr;
}

BandSchemeType convertIntToBandSchemeType(int band)
{
    BandSchemeType ret = BNULL;
    switch (band)
    {
	case 0:
	{
	    break;
	}
	case 1:
	{
	    ret = BSQ;
	    break;
	}
	case 2:
	{
	    ret = BIP;
	    break;
	}
	case 3:
	{
	    ret = BIL;
	    break;
	}
	default:
	{
	    
	    cout << "Error. Band scheme is an integer number between 0 and 3." << endl;
	    ERR_MESSAGE;
	}
    }
    return ret;
}

