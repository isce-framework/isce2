#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>
#include "InterleavedAccessor.h"

using namespace std;
void InterleavedAccessor::finalize()
{
    if(FileObject && FileObject.is_open())
    {
        std::cout << "API close:  " << Filename << std::endl;
        FileObject.close();
    }
    if(!(Data == NULL))
    {
        delete [] Data;
    }
}


void InterleavedAccessor::init(string filename, string accessMode, int sizeV,int bands,int width)
{
    LineWidth = width;
    SizeV = sizeV;
    Bands = bands;
    setAccessMode(accessMode);

    Filename = filename;
    openFile(Filename,AccessMode, FileObject);
    //if(AccessMode != "write")// if file is readable so can use tellg
    //{
        streampos save = FileObject.tellg();
        FileObject.seekg(0,ios::end);
        streampos size = FileObject.tellg();
        if(size != 0)
        {
            NumberOfLines = size/(SizeV*LineWidth*Bands);
        }
        else
        {
            NumberOfLines = -1;
        }
        if(!FileObject.good())
        {
            FileObject.clear();
        }
        FileObject.seekg(save); // put back original position
    //}

}

void InterleavedAccessor::rewindAccessor()
{
    
    FileObject.clear();
    if(FileObject && AccessMode != "write")// if file is readable 
    {
        FileObject.seekg(0,ios::end);
    } 
    EofFlag = 0;
}
int InterleavedAccessor::getFileLength()
{
    int length = 0;
    if(AccessMode == "write" || AccessMode == "writeread")
    {
        streampos save = FileObject.tellp();
        FileObject.seekp(0,ios::end);
        streampos size = FileObject.tellp();
        if(size != 0)
        {
            length = size/(SizeV*LineWidth*Bands);
        }
        if(!FileObject.good())
        {
            FileObject.clear();
        }
        FileObject.seekp(save); // put back original position
    }
    else
    {
        length = NumberOfLines;
    }
    return length;

}
void InterleavedAccessor::createFile(int numberOfLines)
{
    int lineSize = LineWidth*Bands*SizeV;
    vector<char> line(lineSize,0);
    for(int i = 0; i < numberOfLines; ++i)
    {
        
        FileObject.write((char *) &line[0], lineSize);
    }
    //rewind
    FileObject.seekp(0, ios_base::beg);
    if(!FileObject.good())
    {
        FileObject.clear();
    }
    NumberOfLines = numberOfLines;
}

void  InterleavedAccessor::openFile(string filename, string accessMode, fstream & fd)
{
    if(accessMode == "read" || accessMode == "READ")
    {
        std::cout << "API open (R): " << filename << std::endl;
        fd.open(filename.c_str(), ios_base::in);
        if(fd.fail())
        {
            string errMsg = "Cannot open the file " + filename + " in " + accessMode + " mode.";
            throw runtime_error(errMsg);
        }

    }
    else if(accessMode == "write" || accessMode == "WRITE")
    {
        std::cout << "API open (W): " << filename << std::endl;
        fd.open(filename.c_str(), ios_base::out);
    }
    else if(accessMode == "append" || accessMode == "APPEND")
    {
        std::cout << "API open (A): "<< filename << std::endl;
        fd.open(filename.c_str(), ios_base::app);
    }
    else if(accessMode == "writeread" || accessMode == "WRITEREAD")
    {
        std::cout << "API open (WR): " << filename << std::endl;
        fd.open(filename.c_str(), ios_base::trunc | ios_base::in | ios_base::out);
    }
    else if(accessMode == "readwrite" || accessMode == "READWRITE")
    {
        std::cout << "API open (RW): " <<filename << std::endl;
        fd.open(filename.c_str(), ios_base::in | ios_base::out);
    }
    else
    {
        string errMsg = "Unrecognized open mode " + accessMode + " for file " + filename;
        throw runtime_error(errMsg);
    }
    if(!fd.good())
    {
        string errMsg = "Cannot open file " + filename;
        throw runtime_error(errMsg);
    }
}
void InterleavedAccessor::getStream(char * dataLine,  int  & numEl)
{
    FileObject.read(dataLine,numEl*SizeV);
    numEl = FileObject.gcount()/SizeV;
}
void InterleavedAccessor::getStreamAtPos(char * dataLine, int & pos,  int & numEl)
{
    streampos off = (streampos) (pos*SizeV);
    FileObject.seekg(off, ios_base::beg);
    FileObject.read(dataLine,numEl*SizeV);
    numEl = FileObject.gcount()/SizeV;

}
void InterleavedAccessor::setStream(char * dataLine,  int  numEl)
{
    FileObject.write(dataLine,numEl*SizeV);

}
void InterleavedAccessor::setStreamAtPos(char * dataLine, int &  pos,  int & numEl)
{
    streampos off = (streampos) (pos*SizeV);
    FileObject.seekp(off, ios_base::beg);
    FileObject.write(dataLine,numEl*SizeV);

}
