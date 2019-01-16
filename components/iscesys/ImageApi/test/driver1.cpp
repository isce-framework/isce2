#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "AccessorFactory.h"
using namespace std;
int main(int argc, char ** argv)
{
    string filein = "IMG-HH-ALPSRP059980680-P1.0__A.raw";
    string fileout = "ouput.raw";
    
    
    string schemeIn = "BIP";
    string schemeOut = "BIP";
    string accessIn = "read";
    string accessOut = "write";
    int numEl = 3600;
    int Bands = 1;
    int sizeIn = 1; 
    int sizeOut = 1;
    
    /*
    int lineSize = numEl*Bands*sizeOut;
    vector<char> line(lineSize,0);
    for(int i = 0; i < numLines; ++i)
    {
        fout.write((char *) &line[0], lineSize);
    }
    fout.close();
    */
    /////
    /////
    ////
    //
    // find out problem with write
    AccessorFactory AFI;
    DataAccessor * DANCI = AFI.createAccessor(filein,accessIn,sizeIn,Bands,numEl,schemeIn);
    AccessorFactory AFO;
    DataAccessor * DACO = AFO.createAccessor(fileout,accessOut,sizeOut,Bands,numEl,schemeOut);
    //DACO->createFile(numLines);
    char *  line = new char[numEl*Bands];
    int totEl = numEl;
    int lc = 0;
    for(int i = 0; ; ++i)
    {
        /*
        int numElNow = totEl;
        DANCI->getSequentialElements((char *) line,i,0,numElNow);
        if(numElNow != totEl) break;
        DACO->setSequentialElements((char *) line,i,0,numElNow);
        */
        /*
        int flag = DANCI->getLineSequential((char *) line);
        if(flag < 0) break;
        DACO->setLineSequential((char *) line);
        */
        /*
        int flag = DANCI->getLine((char *) line,lc);
        if(flag < 0) break;
        DACO->setLine((char *) line,lc);
        ++lc;
        */
        /*
        int numElNow = totEl;
        DANCI->getStream((char *) line,numElNow);
        if(numElNow != totEl) break;
        DACO->setStream((char *) line,numElNow);
        */
        int numElNow = totEl;
        int pos = lc*numElNow;
        DANCI->getStreamAtPos((char *) line,pos,numElNow);
        if(numElNow != totEl) break;
        DACO->setStreamAtPos((char *) line,pos,numElNow);
        ++lc;
    }
    AFI.finalize(DANCI);
    AFO.finalize(DACO);
    delete [] line;
}
