#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>
#include "GDALAccessor.h"

using namespace std;
void
GDALAccessor::init (void * poly)
{
    return;
}
void
GDALAccessor::finalize ()
{
    if (FileObject != NULL)
    {
        std::cout << "GDAL close: " << Filename << std::endl;
	GDALClose ((GDALDatasetH) FileObject);
    }
    if (!(Data == NULL))
    {
	delete[] Data;
    }
}
void
GDALAccessor::init (string filename, string accessMode, int sizeV, int Bands,
		    int LineWidth)
{
    init (filename, accessMode,sizeV);
}
void
GDALAccessor::init (string filename, string accessMode, int sizeV)
{
    setAccessMode (accessMode);

    Filename = filename;
    openFile (Filename, AccessMode, &FileObject);
    LineWidth = FileObject->GetRasterXSize ();
    NumberOfLines = FileObject->GetRasterYSize ();
    DataType = (GDALDataType) sizeV;
    Bands = FileObject->GetRasterCount ();
    SizeV = GDALGetDataTypeSize (DataType) / 8; //the function returns # bits
    FileSize = LineWidth * NumberOfLines * Bands * SizeV;
}

void
GDALAccessor::rewindAccessor ()
{
    LastPosition = 0;
    EofFlag = 0;
}
int
GDALAccessor::getFileLength ()
{
    return NumberOfLines;
}
void
GDALAccessor::createFile (int numberOfLines)
{
    //TODO
}

void
GDALAccessor::openFile (string filename, string accessMode, GDALDataset ** fd)
{

    if (accessMode == "read" || accessMode == "READ")
    {
        std::cout << "GDAL open (R): " << filename << std::endl;
	(*fd) = (GDALDataset *) GDALOpenShared (filename.c_str (), GA_ReadOnly);
	if ((*fd) == NULL)
	{
	    cout << "Error. Cannot open the file " << filename << " in "
		    << accessMode << " mode." << endl;
	    ERR_MESSAGE
	    ;
	}
    }
    else
    {
	cout << "Error. Only read mode is available and not " << accessMode
		<< " mode." << endl;
	ERR_MESSAGE
	;
    }

}
//The IORaster can read all the bands at once but the data is read one band at the time.
//This means that one has to know the interleaved scheme and reassemble the data into a stream.
//Just assume that is one band image or that the user created the appropriate vrt file to read all at once
void
GDALAccessor::getStream (char * dataLine, int & numEl)
{
    //NOTE: arguments 4 and 5 (nXSize and nYSize) are one based

    int ypos0 = LastPosition / LineWidth;
    int xpos0 = LastPosition % LineWidth;
    LastPosition += numEl;
    int ypos1 = (LastPosition - std::streampos(1)) / LineWidth;
    if (LastPosition * SizeV >= FileSize)
    {
	numEl -= LastPosition % LineWidth;
	LastPosition = 0;
	ypos1 = NumberOfLines - 1;

    }

    char buf[SizeV * (ypos1 - ypos0 + 1) * LineWidth];
    FileObject->RasterIO (GF_Read, 0, ypos0, LineWidth, ypos1 - ypos0 + 1, buf,
			  LineWidth, ypos1 - ypos0 + 1, DataType, 1, NULL, 0, 0,
			  0, NULL);
    for (int i = 0; i < numEl; ++i)
    {
	for (int j = 0; j < SizeV; ++j)
	{
	    dataLine[i * SizeV + j] = buf[xpos0 * SizeV + i * SizeV + j];
	}
    }
}

void
GDALAccessor::getStreamAtPos (char * dataLine, int & pos, int & numEl)
{
    if (pos * SizeV >= FileSize)
    {
	numEl = 0;
    }
    else
    {
	//put pos in  npos since it changes and pos is by reference.
	//should not have passed by reference since it is not modified
	int npos = pos;
	int ypos0 = npos / LineWidth;
	int xpos0 = npos % LineWidth;
	npos += numEl;
	int ypos1 = (npos - 1) / LineWidth;
	if (npos * SizeV >= FileSize)
	{
	    numEl -= npos % LineWidth;
	    ypos1 = NumberOfLines - 1;
	}
	char buf[SizeV * (ypos1 - ypos0 + 1) * LineWidth];
	FileObject->RasterIO (GF_Read, 0, ypos0, LineWidth, ypos1 - ypos0 + 1,
			      buf, LineWidth, ypos1 - ypos0 + 1, DataType, 1,
			      NULL, 0, 0, 0, NULL);
	for (int i = 0; i < numEl; ++i)
	{
	    for (int j = 0; j < SizeV; ++j)
	    {
		dataLine[i * SizeV + j] = buf[xpos0 * SizeV + i * SizeV + j];
	    }
	}
    }
}
void
GDALAccessor::setData (char * buf, int row, int col, int numEl)
{
//TO DO once we start with new formats
    return;
}

void
GDALAccessor::setDataBand (char* buf, int row, int col, int numEl, int band)
{
//TO DO once we start with new formats
    return;
}

//Since GDAL RasterIO returns the data in BSQ (band sequential) no matter what the underlying scheme is
//we don't need a reader for each interleaved scheme
void
GDALAccessor::getData (char * buf, int row, int col, int & numEl)
{
    int ypos0 = row;
    int xpos0 = col;
    int ypos1 = ypos0 + (xpos0 + numEl - 1) / LineWidth;
    
//make sure we don't go over
    if (ypos1 >= NumberOfLines)
    {
	ypos1 = NumberOfLines - 1;
	//adjust number of elements read
	numEl -= (xpos0 + numEl - 1) % LineWidth;
    EofFlag = -1;
    }

//B. Riel: 05/19/17: additional check for ypos0 to prevent negative allocation size
    if (ypos0 > ypos1)
    {
    ypos0 = ypos1;
    }

//get every band at once. Read enough line to fit all the data. GDAL read one band after the other
//i.e. band sequential scheme
    char dataLine[SizeV * (ypos1 - ypos0 + 1) * LineWidth * Bands];
    CPLErr err = FileObject->RasterIO (GF_Read, 0, ypos0, LineWidth, ypos1 - ypos0 + 1,
			  dataLine, LineWidth, ypos1 - ypos0 + 1, DataType,
			  Bands, NULL, 0, 0, 0, NULL);

    for (int i = 0; i < numEl; ++i)
    {
	for (int j = 0; j < Bands; ++j)
	{

	    for (int k = 0; k < SizeV; ++k)
	    {

		buf[i * Bands * SizeV + j * SizeV + k] = dataLine[xpos0 * SizeV
			+ i * SizeV
			+ j * SizeV * (ypos1 - ypos0 + 1) * LineWidth + k];
	    }
	}
    }
}
//Similarly as above the RasterIO already returns the band. Just put it into the buffer
void
GDALAccessor::getDataBand (char * buf, int row, int col, int &numEl, int band)
{
    GDALRasterBand *poBand;
//NOTE GDAL band counting is 1 based
    poBand = FileObject->GetRasterBand (band + 1);
    int ypos0 = row;
    int xpos0 = col;
    int ypos1 = ypos0 + (xpos0 + numEl - 1) / LineWidth;
//make sure we don't go over
    if (ypos1 >= NumberOfLines)
    {
	ypos1 = NumberOfLines - 1;
	//adjust number of elements read
	numEl -= (xpos0 + numEl - 1) % LineWidth;
        EofFlag = -1;
    }
//get every band at once. Read enough line to fit all the data. GDAL read one band after the other
//i.e. band sequential scheme
    char dataLine[SizeV * (ypos1 - ypos0 + 1) * LineWidth];
    poBand->RasterIO (GF_Read, 0, ypos0, LineWidth, ypos1 - ypos0 + 1, dataLine,
		      LineWidth, ypos1 - ypos0 + 1, DataType, 0, 0);

    for (int i = 0; i < numEl; ++i)
    {
	for (int k = 0; k < SizeV; ++k)
	{
	    buf[i * SizeV + k] = dataLine[xpos0 * SizeV + i * SizeV + k];
	}

    }
}
void
GDALAccessor::setStream (char * dataLine, int numEl)
{

}
void
GDALAccessor::setStreamAtPos (char * dataLine, int & pos, int & numEl)
{

}

