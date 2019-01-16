#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>
#include "InterleavedBase.h"

using namespace std;
/*void InterleavedBase::finalize()
{
    std::cout << "Base finalize: " << Filename << std::endl;
    if(!(Data == NULL))
    {
        delete [] Data;
    }
}*/

//assume that the init has been already called
void InterleavedBase::alloc(int numLines)
{
    Data = new char[LineWidth*SizeV*Bands*numLines];
    NumberOfLines = numLines;
}



void InterleavedBase::setAccessMode(string accessMode)
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

