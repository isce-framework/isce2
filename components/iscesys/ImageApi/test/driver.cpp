#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "AccessorFactory.h"
using namespace std;
int main(int argc, char ** argv)
{
    ofstream fout("input.bip");
    int numEl = 4;
    int numLines = 6;
    int Bands = 2;
    int sizeIn = 4; 
    int sizeOut = 8;

    vector<float> in(numEl*numLines*Bands,0);
    ofstream foutTxt("inputBil.txt");
    int pos = 0;
    for(int k = 0; k < numLines; ++k)
    {
        for(int i = 0; i < numEl; ++i)
        {
            for(int j = 0; j < Bands; ++j)
            {
                in[j + Bands*i + numEl*Bands*k] = pos++;
                foutTxt <<  in[j + Bands*i + numEl*Bands*k] << " ";
            }

        }
        foutTxt << endl;
    }
    foutTxt.close();
    fout.write((char *)&in[0],numEl*numLines*Bands*sizeof(float));
    fout.close();
    string filein = "input.bip";
    string fileout = "ouput.bil";
    
    
    string caster = "FloatToDouble";
    string schemeIn = "BIP";
    string schemeOut = "BSQ";
    string accessIn = "read";
    string accessOut = "write";
    
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
    DataAccessor * DACO = AFO.createAccessor(fileout,accessOut,sizeOut,Bands,numEl,schemeOut,caster);
    DACO->createFile(numLines);
    float *  line = new float[numEl*Bands];
    for(int i = 0; i < numLines; ++i)
    {
        DANCI->getLine((char *) line,i);
        DACO->setLine((char *) line,i);
    }
    AFI.finalize(DANCI);
    AFO.finalize(DACO);
    ifstream fin;
    fin.open(fileout.c_str());
    for(int k = 0; k < numLines; ++k)
    {
        for(int j = 0; j < Bands; ++j)
        {
            for(int i = 0; i < numEl; ++i)
            {
                double tmp = 0;
                fin.read((char *)&tmp,8);
                if(fin.eof()) break;
                cout <<  tmp << " ";
            }
            cout << endl;

        }
    }
    fin.close();
    delete [] line;
}
