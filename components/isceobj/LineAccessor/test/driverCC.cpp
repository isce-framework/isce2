#include "driverCC.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <complex>
#include "LineAccessor.h"
using namespace std;

int main(int argc, char ** argv)
{

	stringstream ss(stringstream::in | stringstream::out);
	int choice = 1;
	if(argc > 1)
	{
	    ss << argv[1];
	    ss >> choice;
	}
	switch(choice)
	{
	    case 1:
	    case 2:
		{
		    LineAccessor LAGet;
		    string  infile = "testFile";
		    int fileSize = 4*4*10;
		    char buffer[fileSize];
		    //create test file 160 bytes with integer from 0 to 159
		    for(int i = 0; i < fileSize; ++i)
		    {
			buffer[i] = i;
		    }
		    ofstream fout(infile.c_str());
		    fout.write(buffer,fileSize);
		    fout.close();
		    string filemode = "read";
		    char endian = 'l';
		    char machineEnd = LAGet.getMachineEndianness();
		    //choice = 1 test machine and file endianness being the same
		    if((choice == 1))
		    {
			if((machineEnd == 'b'))
			{
			    endian = 'b';
			}
		    }
		    else if((choice == 2))
		    {
			if((machineEnd == 'l'))
			{
			    endian = 'b';
			}
		    }
		    string  outfile = (choice == 1 ? "testOutC1": "testOutC2" );
		    string type = "FLOAT";// using numpy nomenclature for variable type
		    int col = 4;// width of the tile. this means 10 lines in total. each line is col*sizeof(float) = 4*4 = 16 bytes.
		    int row = 3;// height of the tile. 
		    // create image object to read from
		    LAGet.initLineAccessor(infile, filemode, endian, type, row,col);

		    LineAccessor LASet;
		    string filemode1 = "writeread";
		    // create image objet to write into
		    LASet.initLineAccessor(outfile, filemode1, endian, type, row,col);
		    uint64_t addressGet =(uint64_t) &LAGet;
		    uint64_t  addressSet =(uint64_t)&LASet;
		    testImageSetGet_f(&addressGet,&addressSet,&choice);
		    //need to do flushing and free memory
		    LASet.finalizeLineAccessor();
		    LAGet.finalizeLineAccessor();
		    break;
		}
	    case 3:
		{
		    int fileSize = 4*4*10;
		    char buffer[fileSize];
		    for(int i = 0; i < fileSize; ++i)
		    {
			buffer[i] = i;
		    }
		    string infile = "testSwap";
		    string outfile = "testSwapOutC";
		    string type = "FLOAT";
		    ofstream fout(infile.c_str());
		    fout.write(buffer,fileSize);
		    fout.close();
		    LineAccessor LAGet;
		    LAGet.convertFileEndianness(infile,outfile,type);
		    break;
		}
	    case 4:
		{
		    LineAccessor LAGet;
		    int numb = 2;

		    string  infile = "testFileBand";
		    string  outfile = "testFileBandOutC";
		    int fileSize = 4*4*10;
		    char buffer[fileSize];
		    for(int i = 0; i < fileSize; ++i)
		    {
			buffer[i] = i;
		    }
		    ofstream fout(infile.c_str());
		    fout.write(buffer,fileSize);
		    fout.close();
		    string type = "FLOAT";// using numpy nomenclature for variable type
		    int col = 4;// width of the tile

		    LAGet.changeBandScheme(infile, outfile, type, col,numb,BSQ, BIL);
		    break;
		}
	    default:
		{
		    cout << "Error. Wrong selection" << endl;
		    ERR_MESSAGE;
		}

	}
	
#if(0)
	
#endif
}


