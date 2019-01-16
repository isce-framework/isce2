#include "LineAccessor.h"
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
// PUBLIC

void LineAccessor::changeBandScheme(string filein, string fileout, string type, int width, int numBands, BandSchemeType bandIn, BandSchemeType bandOut)
{

    try
    {
        fstream fin(filein.c_str(), ios::in);
        if(!fin)
        {
            cout << "Cannot open file " << filein << endl;
            ERR_MESSAGE;
        }
        int sizeV = getTypeSize(type);
        int length = ((int)getFileSize(fin))/(width*numBands*sizeV);
        char * totFile = new char[sizeV*width*length*numBands];
        char * line = new char[sizeV*width];
        ofstream fout(fileout.c_str());
        if(!fout)
        {
            cout << "Cannot open file " << fileout << endl;
            ERR_MESSAGE;
        }
        fin.read(totFile,sizeV*width*length*numBands);

        if((bandIn == BIP && bandOut == BIL) || (bandIn == BSQ && bandOut == BIL))
        {
            for(int i = 0; i < length; ++i)
            {
                for(int k = 0; k < numBands; ++k)
                {
                    int cnt = 0;
                    for(int j = 0; j < width; ++j)
                    {

                        for(int p = 0; p < sizeV; ++p)
                        {
                            if((bandIn == BIP))
                            {
                                line[cnt] = totFile[p + k*sizeV + j*sizeV*numBands + i*sizeV*numBands*width];
                            }
                            else
                            {
                                line[cnt] = totFile[p + j*sizeV + i*sizeV*width + k*sizeV*length*width];

                            }
                            ++cnt;
                        }


                    }
                    fout.write(line,cnt);

                }

            }

        }
        else if((bandIn == BIP && bandOut == BSQ) || (bandIn == BIL && bandOut == BSQ))
        {
            for(int k = 0; k < numBands; ++k)
            {
                for(int i = 0; i < length; ++i)
                {
                    int cnt = 0;
                    for(int j = 0; j < width; ++j)
                    {

                        for(int p = 0; p < sizeV; ++p)
                        {
                            if((bandIn == BIP))
                            {
                                line[cnt] = totFile[p + k*sizeV + j*sizeV*numBands + i*sizeV*numBands*width];
                            }
                            else
                            {
                                line[cnt] = totFile[p + j*sizeV + k*sizeV*width + i*sizeV*numBands*width];

                            }
                            ++cnt;
                        }

                    }
                    fout.write(line,cnt);

                }

            }

        }
        else if((bandIn == BSQ && bandOut == BIP) || (bandIn == BIL && bandOut == BIP))
        {
            for(int i = 0; i < length; ++i)
            {
                for(int j = 0; j < width; ++j)
                {
                    int cnt = 0;
                    for(int k = 0; k < numBands; ++k)
                    {

                        for(int p = 0; p < sizeV; ++p)
                        {
                            if((bandIn == BSQ))
                            {
                                line[cnt] = totFile[p + j*sizeV + i*sizeV*width + k*sizeV*length*width];
                            }
                            else
                            {
                                line[cnt] = totFile[p + j*sizeV + k*sizeV*width + i*sizeV*numBands*width];

                            }
                            ++cnt;
                        }

                    }
                    fout.write(line,cnt);

                }

            }

        }
        else
        {
            cout << "Error. Type of input and/or output interleaving band scheme must be BIL,BSQ or BIP." << endl;
            ERR_MESSAGE;
        }
        delete [] totFile;
        delete [] line;
        fout.close();
        fin.close();
    }
    catch(bad_alloc&)//cannot read the full size in memory, try something else
    {
        //for BIP <-> BIL can read one  "line" (width and number bands) and rearrange the elements i.e. the
        //file can be read one "line"  at the time

        fstream fin(filein.c_str(), ios::in);
        if(!fin)
        {
            cout << "Cannot open file " << filein << endl;
            ERR_MESSAGE;
        }
        int sizeV = getTypeSize(type);
        int length = ((int)getFileSize(fin))/(width*numBands*sizeV);
        int lineSize = sizeV*width*numBands;
        char * lineIn = new char[lineSize];
        char * line = new char[lineSize];
        ofstream fout(fileout.c_str());
        if(!fout)
        {
            cout << "Cannot open file " << fileout << endl;
            ERR_MESSAGE;
        }
        if((bandOut == BSQ))
        {
            vector<char> lineL(lineSize,0);
            for(int i = 0; i < length; ++i)
            {
                fout.write((char *) &lineL[0], lineSize);
            }
        }
        for(int i = 0; i < length; ++i)
        {

            if(((bandIn == BIL) &&  (bandOut == BIP)) || ((bandOut == BIL) &&  (bandIn == BIP)) )
            {
                fin.read(lineIn,lineSize);
                for(int k = 0; k < numBands; ++k)
                {
                    for(int j = 0; j < width; ++j)
                    {

                        for(int p = 0; p < sizeV; ++p)
                        {
                            if((bandIn == BIL))
                            {
                                line[p + k*sizeV + j*sizeV*numBands] = lineIn[p + j*sizeV + k*sizeV*width];
                            }
                            else
                            {
                                line[p + j*sizeV + k*sizeV*width] = lineIn[p + k*sizeV + j*sizeV*numBands];
                            }
                        }

                    }
                }

                fout.write(line,lineSize);
            }
            else if(((bandIn == BIL) &&  (bandOut == BSQ)) || ((bandIn == BIP) &&  (bandOut == BSQ)) )
            {
                fin.read(lineIn,lineSize);
                for(int k = 0; k < numBands; ++k)
                {
                    streampos pos = sizeV*((width*i) + (k*length*width));
                    fout.seekp(pos);
                    if(bandIn == BIL)
                    {
                        fout.write(&lineIn[sizeV*width*k],sizeV*width);
                    }
                    else
                    {
                        for(int j = 0; j < width; ++j)
                        {
                            for(int p = 0; p < sizeV; ++p)
                            {
                                line[p + sizeV*(j + k*width)] =  lineIn[p + k*sizeV + j*sizeV*numBands];
                            }
                        }

                        fout.write(&line[sizeV*width*k],sizeV*width);

                    }
                }
            }
            else if(((bandIn == BSQ) &&  (bandOut == BIL)) || ((bandIn == BSQ) &&  (bandOut == BIP)))
            {
                for(int k = 0; k < numBands; ++k)
                {
                    streampos pos = sizeV*((width*i) + (k*length*width));
                    fin.seekg(pos);
                    fin.read(&lineIn[sizeV*width*k],sizeV*width);
                }
                if(bandOut == BIL)
                {
                    fout.write(lineIn,sizeV*width*numBands);
                }
                else
                {
                    for(int k = 0; k < numBands; ++k)
                    {
                        for(int j = 0; j < width; ++j)
                        {
                            for(int p = 0; p < sizeV; ++p)
                            {
                                line[p + sizeV*(k + j*numBands)] =  lineIn[p + j*sizeV + k*sizeV*width];
                            }
                        }
                    }
                    fout.write(line,lineSize);
                }

            }
            else
            {
                cout << "Error. Type of input and/or output interleaving band scheme must be BIL,BSQ or BIP." << endl;
                ERR_MESSAGE;

            }

        }


        delete [] lineIn;
        delete [] line;
        fout.close();
        fin.close();
    }
}

void LineAccessor::convertFileEndianness(string fileIn, string fileOut, string type)
{

    fstream fin(fileIn.c_str(), ios::in);
    if(!fin)
    {
        cout << "Error. Cannot open file " << fileIn << endl;
        ERR_MESSAGE;
    }
    ofstream fout(fileOut.c_str());
    if(!fin)
    {
        cout << "Error. Cannot open file " << fileOut << endl;
        ERR_MESSAGE;
    }
    bool memoryNotAllocated = true;
    char * fileBuffer = NULL;
    int divisor = 1;
    int sizeV = getSizeForSwap(type);
    if(sizeV == 1)
    {
        cout << "No need to convert endianness if the type size is one." << endl;
    }
    else
    {
        streampos  fileSize = 0;
        streampos memorySize = 0;
        while(memoryNotAllocated)
        {
            try
            {
                fileSize = getFileSize(fin);
                memorySize = (fileSize/(divisor*sizeV))*sizeV;//make sure that an integer number of sizeV is read
                fileBuffer = new char[memorySize];
                memoryNotAllocated = false;

            }
            catch(bad_alloc&)
            {
                divisor *= 2;
            }
        }
        while(!fin.eof())
        {
            fin.read(fileBuffer,memorySize);
            streampos bytesRead = fin.gcount();
            streampos numElements = bytesRead/sizeV;
            swapBytes(fileBuffer,numElements,sizeV);
            fout.write(fileBuffer,bytesRead);

        }


        delete [] fileBuffer;
    }
    fin.close();
    fout.close();

}
void LineAccessor::createFile(int * fileLength)
{
    //Checked other ways of doing it using "truncate" function, but it's not portable
    vector<char> line(LineSize,0);
    for(int i = 0; i < (*fileLength); ++i)
    {
        FileObject.write((char *) &line[0], LineSize);
    }
    FileLength = (*fileLength);
    FileSize = LineSize*FileLength;
    FileObject.seekp(0, ios_base::beg);
    FileObject.clear();

}

void LineAccessor::rewindImage()
{
    ColumnPosition = 1;
    LineCounter = 1;
    LinePosition = 1;
    FileObject.seekp(0, ios_base::beg);
    FileObject.seekg(0, ios_base::beg);
    FileObject.clear();
}
char LineAccessor::getMachineEndianness()
{
    unsigned short int intV = 49;//ascii code for 1
    char *  ptChar = (char *) &intV;
    char retVal = 'b';
    if(ptChar[0] == '1')
    {
        retVal = 'l';
    }

    return retVal;
}



char * LineAccessor::getTileArray()
{
    return PtArray;
}

int LineAccessor::getSizeForSwap(string type)
{
    int size = getTypeSize(type);
    if(type == "CFLOAT" || type == "CDOUBLE" || type == "CLONGDOUBLE" || type == "cfloat" || type == "cdouble" || type == "clongdouble")
    {
        size /=2;

    }
    return size;
}

int LineAccessor::getTypeSize(string type)
{
    int retVal = -1;
    if(type == "byte" || type == "BYTE" || type == "char" || type == "CHAR")
    {
        retVal = sizeof(char);
    }
    else if(type == "short" || type == "SHORT")
    {
        retVal = sizeof(short);
    }
    else if(type == "int" || type == "INT")
    {
        retVal = sizeof(int);
    }
    else if(type == "long" || type == "LONG")
    {
        retVal = sizeof(long);
    }
    else if(type == "longlong" || type == "LONGLONG")
    {
        retVal = sizeof(long long);
    }
    else if(type == "float" || type == "FLOAT")
    {
        retVal = sizeof(float);
    }
    else if(type == "double" || type == "DOUBLE")
    {
        retVal = sizeof(double);
    }
    else if(type == "longdouble" || type == "LONGDOUBLE")
    {
        retVal = sizeof(long double);
    }
    else if(type == "cfloat" || type == "CFLOAT")
    {
        retVal = sizeof(complex<float>);
    }
    else if(type == "cdouble" || type == "CDOUBLE")
    {
        retVal = sizeof(complex<double>);
    }
    else if(type == "clongdouble" || type == "CLONGDOUBLE")
    {
        retVal = sizeof(complex<long double>);
    }
    else
    {
        vector<string> data = getAvailableDataTypes();
        cout << "Error. Unrecognized data type " << type << ". Available types are: "<< endl;
        for(int i = 0; i < (int)data.size(); ++i)
        {
            cout << data[i] << endl;
        }
        ERR_MESSAGE;
    }
    return retVal;
}

vector<string> LineAccessor::getAvailableDataTypes()
{
    vector<string> dataType;
    dataType.push_back("BYTE");
    dataType.push_back("CHAR");
    dataType.push_back("SHORT");
    dataType.push_back("INT");
    dataType.push_back("LONG");
    dataType.push_back("LONGLONG");
    dataType.push_back("FLOAT");
    dataType.push_back("DOUBLE");
    dataType.push_back("LONGDOUBLE");
    dataType.push_back("CFLOAT");
    dataType.push_back("CDOUBLE");
    dataType.push_back("CLONGDOUBLE");
    return dataType;
}
void LineAccessor::printAvailableDataTypesAndSizes()
{
    vector<string> dataType;
    vector<int>  size;

    getAvailableDataTypesAndSizes(dataType, size);
    for(int i = 0; i < (int)size.size(); ++i)
    {
        cout << dataType[i] << "\t" << size[i] << endl;
    }
}
void LineAccessor::getAvailableDataTypesAndSizes(vector<string> & dataType, vector<int> & size)
{
    dataType.clear();
    size.clear();
    dataType.push_back("BYTE");
    size.push_back(getTypeSize("BYTE"));
    dataType.push_back("CHAR");
    size.push_back(getTypeSize("CHAR"));
    dataType.push_back("SHORT");
    size.push_back(getTypeSize("SHORT"));
    dataType.push_back("INT");
    size.push_back(getTypeSize("INT"));
    dataType.push_back("LONG");
    size.push_back(getTypeSize("LONG"));
    dataType.push_back("LONGLONG");
    size.push_back(getTypeSize("LONGLONG"));
    dataType.push_back("FLOAT");
    size.push_back(getTypeSize("FLOAT"));
    dataType.push_back("DOUBLE");
    size.push_back(getTypeSize("DOUBLE"));
    dataType.push_back("LONGDOUBLE");
    size.push_back(getTypeSize("LONGDOUBLE"));
    dataType.push_back("CFLOAT");
    size.push_back(getTypeSize("CFLOAT"));
    dataType.push_back("CDOUBLE");
    size.push_back(getTypeSize("CDOUBLE"));
    dataType.push_back("CLONGDOUBLE");
    size.push_back(getTypeSize("CLONGDOUBLE"));
}
void LineAccessor::finalizeLineAccessor()
{
    if(NeedToFlush)
    {
        FileObject.write(PtArray,(LineCounter - 1)*SizeV*FileWidth);
    }
    FileObject.close();
    delete [] PtArray;
}

void LineAccessor::getStream(char * dataLine,  int * numEl)
{
    FileObject.read(dataLine,(*numEl)*SizeV);
    (*numEl) = FileObject.gcount()/SizeV;

}
void LineAccessor::getStreamAtPos(char * dataLine, int * pos,  int * numEl)
{
    streampos off = (streampos) ((*pos) - 1)*SizeV;
    FileObject.seekg(off, ios_base::beg);
    FileObject.read(dataLine,(*numEl)*SizeV);
    (*numEl) = FileObject.gcount()/SizeV;

}
void LineAccessor::getElements(char * dataLine, int * row, int * col, int * numEl)
{
    vector<int> indx((*numEl),0);
    vector<int> colCp((*numEl),0);
    vector<int> rowCp((*numEl),0);
    for(int i = 0; i < (*numEl); ++i)
    {
        checkRowRange(row[i]);
        checkColumnRange(col[i]);
        indx[i] = i;
        colCp[i] = col[i];
        rowCp[i] = row[i];
    }
    quickSort(&rowCp[0],&colCp[0],&indx[0],0,(*numEl) - 1);//so could check if some elements are close by and load
    // a tile that might contain some.
    int elementsRead = 0;
    int rowPos = rowCp[0];

    char * buffer = new char[LineSize*ReadBufferSize];
    while(true)
    {
        streampos off = (streampos) (rowPos - 1)*LineSize;
        FileObject.seekg(off, ios_base::beg);
        FileObject.read(buffer,LineSize*ReadBufferSize);
        int lineIndx = elementsRead;
        int startIndx = elementsRead;
        while(true)
        {
            if(rowCp[lineIndx] < (rowPos) + ReadBufferSize)
            {
                ++lineIndx;
                if(lineIndx == (*numEl))
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }
        for(int i = startIndx; i < lineIndx; ++i)
        {
            for(int j = 0; j < SizeV; ++j)
            {

                dataLine[j + indx[i]*SizeV] = buffer[j + (colCp[i] - 1)*SizeV + (rowCp[i] - rowPos)*LineSize];
            }
        }
        if(lineIndx == (*numEl))
        {
            break;
        }
        elementsRead = lineIndx;
        rowPos = rowCp[lineIndx];

    }

    delete [] buffer;

    int numElForSwap = (*numEl)*SizeV/SizeForSwap;
    if(EndianMachine != EndianFile)
    {
        swapBytes(dataLine,numElForSwap,SizeForSwap);
    }

}

void LineAccessor::getSequentialElements(char * dataLine, int * row, int * col, int * numEl)
{
    checkRowRange((*row));
    checkColumnRange((*col));
    streampos off = (streampos) ((*row) - 1)*LineSize + ((*col) - 1)*SizeV;
    FileObject.seekg(off, ios_base::beg);
    FileObject.read(dataLine,(*numEl)*SizeV);
    (*numEl) = FileObject.gcount()/SizeV;
    int numElForSwap = FileObject.gcount()/SizeForSwap;
    if(EndianMachine != EndianFile)
    {
        swapBytes(dataLine,numElForSwap,SizeForSwap);
    }
}
void LineAccessor::getLine(char * dataLine, int * row)
{
    if((*row) > FileLength || (*row) < 1)
    {
        (*row) = -1;

    }
    else
    {
        streampos off = (streampos) ((*row) - 1)*LineSize;
        FileObject.seekg(off, ios_base::beg);
        FileObject.read(dataLine,LineSize);
        int numElForSwap = FileObject.gcount()/SizeForSwap;
        if(EndianMachine != EndianFile)
        {
            swapBytes(dataLine,numElForSwap,SizeForSwap);
        }
    }
}
void LineAccessor::getLineSequential(char * dataLine, int * eof)
{
    if(LinePosition > FileLength)// return negative val to signify the eof
    {
        (*eof) = -1;
    }
    else
    {
        if(SizeYTile == 1)
        {
            FileObject.read(dataLine,LineSize);
            int numElForSwap = LineSize/SizeForSwap;
            if(EndianMachine != EndianFile)
            {
                swapBytes(dataLine,numElForSwap,SizeForSwap);
            }
            (*eof) = LinePosition;

        }
        else
        {
            if( ((LineCounter))%(SizeYTile + 1) == 0)
            {
                FileObject.read(PtArray,TileSize);
                int numElForSwap = FileObject.gcount()/SizeForSwap;
                if(EndianMachine != EndianFile)
                {
                    swapBytes(PtArray,numElForSwap,SizeForSwap);
                }
                LineCounter = 1;
            }
            for(int i = 0; i < FileWidth*SizeV; ++i)
            {
                dataLine[i] = PtArray[i + (LineCounter - 1)*SizeV*FileWidth];
            }
            (*eof) = LinePosition;
            ++LineCounter;
        }
    }
    ++LinePosition;

}

bool LineAccessor::isInit()
{
    return IsInit;
}

void LineAccessor::initLineAccessor(string filename, string filemode, char endianFile, string type,int row,int col)
{
    IsInit = true;
    DataType  = getAvailableDataTypes();
    SizeV = getTypeSize(type);
    SizeForSwap = getSizeForSwap(type);
    FileDataType = type;

    Filename = filename;
    if(col <= 0)
    {

        SizeXTile = 1;
    }
    else
    {

        SizeXTile = col;
    }

    SizeYTile = row;
    FileWidth = SizeXTile;
    LineSize = SizeXTile*SizeV;
    TileSize = SizeXTile*SizeYTile*SizeV;
    PtArray = new char[SizeXTile*SizeYTile*SizeV];
    EndianMachine = getMachineEndianness();
    if(endianFile == 'l' || endianFile == 'L' || endianFile == 'b' || endianFile == 'B')
    {
        EndianFile = endianFile;
    }
    else
    {
        cout << "Error. Endianness must be \"l,L,b,B\"" << endl;
        ERR_MESSAGE;
    }
    setAccessMode(filemode);
    openFile(Filename,AccessMode, FileObject);
    MachineSize = getMachineSize();
}
// move the fstream pointer to the begLine
void LineAccessor::initSequentialAccessor(int * begLine)
{
    if((AccessMode == "write") && ((*begLine) > FileLength))
    {
        cout << "Error. Cannot position the file pointer at line " << (*begLine) << endl;
        ERR_MESSAGE;
    }
    LinePosition = (* begLine);
    checkRowRange(LinePosition);
    streampos off = (streampos) ((*begLine) - 1)*LineSize;
    FileObject.seekg(off, ios_base::beg);
    FileObject.seekp(off, ios_base::beg);
    //the first check is due to avoid that the file pointer move past the first line. see getLineSequential where it checks for SizeYTile == 1.
    if((SizeYTile > 1) && (AccessMode == "read"))
    {
        FileObject.read(PtArray,TileSize);
        int numElForSwap = FileObject.gcount()/SizeForSwap;
        if(EndianMachine != EndianFile)
        {
            swapBytes(PtArray,numElForSwap,SizeForSwap);
        }
    }
}

void LineAccessor::printObjectInfo()
{
    cout << "File name: " << Filename << endl;
    cout << "File access mode: " << AccessMode << endl;
    cout << "File datatype: " << FileDataType << endl;
    cout << "File datatype size: " << SizeV << endl;
    cout << "File endiannes: " << (EndianFile == 'b' ? "big endian": "little endian") << endl;
    cout << "Machine endiannes: " << (EndianMachine == 'b' ? "big endian": "little endian") << endl;
    cout << "File size: " << FileSize << " bytes" << endl;
    cout << "File width (number of columns): " << FileWidth << endl;
    cout << "File length (number of rows): " << FileLength << endl;
    cout << "Tile size: " << SizeYTile << (SizeYTile == 1 ? " row, ": " rows, ") << SizeXTile << (SizeXTile == 1 ? " column" : " columns") << endl;

}

void LineAccessor::setStream(char * dataLine,  int * numEl)
{
    FileObject.write(dataLine,(*numEl)*SizeV);

}
void LineAccessor::setStreamAtPos(char * dataLine, int * pos,  int * numEl)
{
    streampos off = (streampos) ((*pos) - 1)*SizeV;
    FileObject.seekp(off, ios_base::beg);
    FileObject.write(dataLine,(*numEl)*SizeV);

}
void LineAccessor::setElements(char * dataLine, int * row, int * col, int * numEl)
{
    //make sure rows and colums are in range
    for(int i = 0; i < (*numEl); ++i)
    {
        checkRowRange(row[i]);
        checkColumnRange(col[i]);
    }
    int elementsRead = 0;//how many elements were in a given tile
    int rowPos = row[0];//beginning of buffer read
    //allocate  a tile
    char * buffer = new char[LineSize*ReadBufferSize];
    while(true)
    {
        int lineIndx = elementsRead;//last line (relative to posRow) were data are
        int startIndx = elementsRead;//first line (relative to rowPos) were data are
        //count how many lines are in a tile
        while(true)
        {
            if(row[lineIndx] < (rowPos) + ReadBufferSize)
            {
                ++lineIndx;
                if(lineIndx == (*numEl))
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }
        if(lineIndx == startIndx + 1)//there is only one element contained in the tile, so don't load it and just write the element
        {
            streampos off = (streampos) (rowPos - 1)*LineSize + (col[startIndx] - 1)*SizeV;
            FileObject.seekp(off, ios_base::beg);
            for(int j = 0; j < SizeV; ++j)
            {
                buffer[j] = dataLine[j + startIndx*SizeV];
            }
            FileObject.write(buffer,SizeV);

        }
        else
        {
            streampos off = (streampos) (rowPos - 1)*LineSize;
            FileObject.seekg(off, ios_base::beg);
            FileObject.read(buffer,LineSize*ReadBufferSize);
            if(FileObject.eof())
            {
                FileObject.clear();
            }
            int countG =  FileObject.gcount();

            FileObject.seekp(off, ios_base::beg);
            //copy elements in the tile and write back
            for(int i = startIndx; i < lineIndx; ++i)
            {
                for(int j = 0; j < SizeV; ++j)
                {

                    buffer[j + (col[i] - 1)*SizeV + (row[i] - rowPos)*LineSize] = dataLine[j + i*SizeV];
                }
            }
            FileObject.write(buffer,countG);
        }
        //wrote all elements, break
        if(lineIndx == (*numEl))
        {
            break;
        }
        elementsRead = lineIndx;
        rowPos = row[lineIndx];

    }

    delete [] buffer;

}

void LineAccessor::setLineSequential(char * dataLine)
{
    if(SizeYTile == 1)
    {
        FileObject.write(dataLine,LineSize);

    }
    else
    {
        for(int i = 0; i < FileWidth*SizeV; ++i)
        {
            PtArray[i + (LineCounter - 1)*FileWidth*SizeV] = dataLine[i];
        }
        if( ((LineCounter))%(SizeYTile) == 0)
        {

            FileObject.write(PtArray,TileSize);
            NeedToFlush = false;
            LineCounter = 1;
        }
        else // just increase the counter
        {
            NeedToFlush = true;
            ++LineCounter;
        }
    }
    ++LinePosition;
}

void LineAccessor::setLine(char * dataLine, int * row)
{
    if(((*row) > FileLength) || ((*row) < 1))
    {
        cout << "Error. The line to be set is out of range. Total number of line in the file = " << FileLength << ", line requested = " << (*row) << endl;
        ERR_MESSAGE;
    }
    LinePosition = (*row);
    streampos off = (streampos) ((*row) - 1)*LineSize;
    FileObject.seekp(off, ios_base::beg);
    FileObject.write(dataLine,LineSize);
}
void LineAccessor::setSequentialElements(char * dataLine, int * row, int * col, int * numEl)
{
    if(!((*row) == LinePosition) && ((*col) == ColumnPosition))
    {
        //is not sequential w/respect to previous write, check if it's in range.
        // in this case the file needs to be already allocated.

        checkRowRange((*row));
        checkColumnRange((*col));
    }
    LinePosition = (*row) + ((*col) + (*numEl))/FileWidth;
    ColumnPosition = ((*col) + (*numEl))%FileWidth;//next column where to write
    if(FileLength > 0 && (LinePosition > FileLength) && ColumnPosition > 1)
    {
        cout << "Error. Writing outside file bounds." << endl;
        ERR_MESSAGE;
    }
    streampos off = (streampos) ((*row) - 1)*LineSize + ((*col) - 1)*SizeV;
    FileObject.seekp(off, ios_base::beg);
    FileObject.write(dataLine,(*numEl)*SizeV);
}

//PRIVATE


void LineAccessor::checkColumnRange(int col)
{
    if(( col) > FileWidth)
    {
        cout << "Error. Trying to access the column " << col  <<" that is larger than the file width " << FileWidth << " ." << endl;
        ERR_MESSAGE;
    }
    if(( col) < 1)
    {
        cout << "Error. The column number has to be a positive." << endl;
        ERR_MESSAGE;
    }

}
void LineAccessor::checkRowRange(int row)
{
    if(( row) > FileLength)
    {

        cout << "Error. Trying to access the line "<< row << " that is larger than the number of lines in the file " << FileLength << "." << endl;
        ERR_MESSAGE;
    }
    if(( row) < 1)
    {
        cout << "Error. The line number has to be a positive" << endl;
        ERR_MESSAGE;
    }

}

streampos  LineAccessor::getFileSize(fstream & fin)
{
    if(!fin.is_open())
    {
        cout << "File must be open" << endl;
        ERR_MESSAGE;
    }
    streampos savePos = fin.tellg();
    fin.seekg(0,ios::end);
    streampos retPos = fin.tellg();
    fin.seekg(savePos,ios::beg);
    return retPos;



}

streampos  LineAccessor::getFileSize(string filename)
{
    ifstream fin;
    fin.open(filename.c_str());
    if(!fin)
    {
        cout << "Cannot open file " << filename << endl;
        ERR_MESSAGE;
    }
    streampos savePos = fin.tellg();
    fin.seekg(0,ios::end);
    streampos retPos = fin.tellg();
    fin.seekg(savePos,ios::beg);
    return retPos;



}
void  LineAccessor::openFile(string filename, string accessMode, fstream & fd)
{
    if(accessMode == "read" || accessMode == "READ")
    {

        fd.open(filename.c_str(), ios_base::in);
        if(fd.fail())
        {
            cout << "Error. Cannot open the file " << filename << " in " << accessMode << " mode." <<endl;
            ERR_MESSAGE;
        }
        if(SizeYTile > 1)// only in this case tiling and prebuffering is used
        {
            fd.read(PtArray,TileSize);// read first tile
        }
        int numElForSwap = fd.gcount()/SizeForSwap;
        if(EndianMachine != EndianFile)
        {
            swapBytes(PtArray,numElForSwap,SizeForSwap);
        }
        FileSize = getFileSize(fd);
        FileLength = FileSize/(SizeV*FileWidth);// number of lines
        if(FileSize%(SizeV*FileWidth))
        {
            //better be divisable by sizeV*FileWidth
            cout << "Error. The number of lines in the file " << Filename << " computed as file_size/(line_size) is not integer. Filesize = " << FileSize << " number element per line = " << FileWidth << " size of one element = " << SizeV << endl;
            ERR_MESSAGE;
        }

    }
    else if(accessMode == "write" || accessMode == "WRITE")
    {
        fd.open(filename.c_str(), ios_base::out);
    }
    else if(accessMode == "append" || accessMode == "APPEND")
    {
        fd.open(filename.c_str(), ios_base::app);
    }
    else if(accessMode == "writeread" || accessMode == "WRITEREAD")
    {
        fd.open(filename.c_str(), ios_base::trunc | ios_base::in | ios_base::out);
    }
    else if(accessMode == "readwrite" || accessMode == "READWRITE")
    {
        fd.open(filename.c_str(), ios_base::in | ios_base::out);
        if(SizeYTile > 1)// only in this case tiling and prebuffering is used
        {
            fd.read(PtArray,TileSize);
        }
        int numElForSwap = fd.gcount()/SizeForSwap;
        if(EndianMachine != EndianFile)
        {
            swapBytes(PtArray,numElForSwap,SizeForSwap);
        }
        FileSize = getFileSize(fd);
        FileLength = FileSize/(SizeV*FileWidth);// number of lines
        if(FileSize%(SizeV*FileWidth))
        {
            //better be divisable by sizeV*FileWidth
            cout << "Error. The number of lines in the file computed as file_size/(line_size) is not integer. Filesize = " << FileSize << " number element per line = " << FileWidth << " size of one element = " << SizeV << endl;
            ERR_MESSAGE;
        }
    }
    else
    {
        cout << "Error. Unrecognized open mode " << accessMode << " for file " << filename << endl;
        ERR_MESSAGE;
    }
    if(!fd)
    {
        cout << "Cannot open file " << filename << endl;
        ERR_MESSAGE;
    }




}
void LineAccessor::setAccessMode(string accessMode)
{
    if(accessMode == "read" || accessMode == "READ")
    {
        AccessMode = "read";
    }
    else if(accessMode == "write" || accessMode == "WRITE")
    {
        AccessMode = "write";
    }
    else if(accessMode == "append" || accessMode == "APPEND")
    {
        AccessMode = "append";
    }
    else if(accessMode == "writeread" || accessMode == "WRITEREAD")
    {
        AccessMode = "writeread";
    }
    else if(accessMode == "readwrite" || accessMode == "READWRITE")
    {
        AccessMode = "readwrite";
    }
    else
    {
        cout << "Error. Unrecognized open mode " << accessMode  << endl;
        ERR_MESSAGE;
    }

}

void LineAccessor::quickSort(int * row, int * col ,int * indx, int lo, int hi)
{
    int i =  lo;
    int j = hi;
    int tmpIndxR = 0;
    int tmpIndxC = 0;
    int tmpIndx = 0;
    int half = row[(lo + hi)/2];
    do
    {
        while (row[i] < half) ++i;
        while (row[j] > half) --j;
        if(i <= j )
        {
            tmpIndxR = row[i];
            tmpIndxC = col[i];
            tmpIndx = indx[i];
            row[i] = row[j];
            col[i] = col[j];
            indx[i] = indx[j];
            row[j] = tmpIndxR;
            col[j] = tmpIndxC;
            indx[j] = tmpIndx;
            ++i;
            --j;
        }
    }
    while(i <= j);
    if(lo < j) quickSort(row,col,indx,lo,j);
    if(hi > i) quickSort(row,col,indx,i,hi);
}
void LineAccessor::swapBytes(char * buffer, int numElements, int sizeV)
{
    switch(sizeV)
    {
        case 2:
            {
                for(int i = 0; i < numElements; ++i)
                {

                    (* (uint16_t *) &buffer[i*sizeV]) = swap2Bytes((uint16_t *) &buffer[i*sizeV]);
                }
                break;
            }
        case 4:
            {

                for(int i = 0; i < numElements; ++i)
                {
                    (* ((uint32_t *) &buffer[i*sizeV])) = swap4Bytes((uint32_t *)&buffer[i*sizeV]);
                }
                break;

            }
        case 8:
            {

#ifndef MACHINE_64
                for(int i = 0; i < numElements; ++i)
                {
                    swap8BytesSlow(&buffer[i*sizeV]);
                }
#else
                for(int i = 0; i < numElements; ++i)
                {

                    (* (uint64_t *) &buffer[i*sizeV]) = swap8BytesFast((uint64_t *) &buffer[i*sizeV]);
                }
#endif
                break;
            }
        case 12:
            {

                for(int i = 0; i < numElements; ++i)
                {
                    swap12Bytes(&buffer[i*sizeV]);
                }
                break;
            }
        case 16:
            {

                for(int i = 0; i < numElements; ++i)
                {
                    swap16Bytes(&buffer[i*sizeV]);
                }
                break;
            }
        default:
            {
                cout << "Unexpected variable size" << endl;
                ERR_MESSAGE;
            }

    }


}

uint16_t LineAccessor::swap2Bytes(uint16_t * x)
{
    return ((*x) & 0xFF00) >> 8 |
        ((*x) & 0x00FF) << 8;
}
uint32_t LineAccessor::swap4Bytes(uint32_t * x)
{
    return ((*x) & 0xFF000000) >> 24 |
        ((*x) & 0x00FF0000) >> 8 |
        ((*x) & 0x0000FF00) << 8 |
        ((*x) & 0x000000FF) << 24;
}

// had to do it since some g++ compiler give a warning if the number is larger then the register, others give an error
#ifdef MACHINE_64
// if the machine is not 64 bit this cannot be used since the registers are too small (>> and  << is done into register, not memory => fast)
uint64_t LineAccessor::swap8BytesFast(uint64_t * x)
{
    return ((*x) & 0xFF00000000000000) >> 56 |
        ((*x) & 0x00FF000000000000) >> 40 |
        ((*x) & 0x0000FF0000000000) >> 24 |
        ((*x) & 0x000000FF00000000)  >> 8  |
        ((*x) & 0x00000000FF000000)  << 8  |
        ((*x) & 0x0000000000FF0000) << 24 |
        ((*x) & 0x000000000000FF00) << 40 |
        ((*x) & 0x00000000000000FF) << 56;
}
#endif
void LineAccessor::swap8BytesSlow(char * x)
{
    char  tmp;
    int size = 8;
    int half = 4;
    for(int i = 0; i < half; ++i)
    {
        tmp = x[i];
        x[i] = x[size-1-i];
        x[size-1-i] = tmp;
    }

}
void LineAccessor::swap12Bytes(char * x) //for some architecture size(long double) = 12
{
    char  tmp;
    int size = 12;
    int half = 6;
    for(int i = 0; i < half; ++i)
    {
        tmp = x[i];
        x[i] = x[size-1-i];
        x[size-1-i] = tmp;
    }
}
void LineAccessor::swap16Bytes(char * x) //for some architecture size(long double) = 12
{
    char  tmp;
    int size = 16;
    int half = 8;
    for(int i = 0; i < half; ++i)
    {
        tmp = x[i];
        x[i] = x[size-1-i];
        x[size-1-i] = tmp;
    }
}

//end-of-file
