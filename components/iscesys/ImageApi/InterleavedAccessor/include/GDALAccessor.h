#ifndef GDALAccessor_h
#define GDALAccessor_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <iostream>
#include "InterleavedBase.h"
#include <stdint.h>
#include "gdal_priv.h"
#include <string.h>
using namespace std;

/*
 * NOTE:
 * Since GDAL RasterIO always returns data in BSQ regardless of the underlying scheme there is no need
 * to have separate accessor for each interleaving scheme when reading.
*/
class GDALAccessor : public InterleavedBase
{
public:
    GDALAccessor () :
	    InterleavedBase ()
    {
	FileObject = NULL;
	Driver = NULL;
	//For the moment let's use VRT
	DriverName = "VRT";
	GDALAllRegister ();
	LastPosition = 0;
    }
    virtual
    ~GDALAccessor ()
    {
    }

    //NOTE sizeV here is identify the enum value corresponding to a specific GDALDataType
    //NOTE the filename is the one used in GDALOpen
    void
    init (string filename, string accessMode, int sizeV);
    void
    init (string filename, string accessMode, int sizeV, int Bands,
	  int LineWidth);
    void
    openFile (string filename, string accessMode, GDALDataset ** fd);
    GDALDataset *
    getFileObject ()
    {
	return FileObject;
    }
    void
    init (void * poly);

    void
    getStreamAtPos (char * buf, int & pos, int & numEl);
    void
    setStreamAtPos (char * buf, int & pos, int & numEl);
    void
    getStream (char * buf, int & numEl);
    void
    setStream (char * buf, int numEl);
    void
    getData (char * buf, int row, int col, int & numEl);
    void
    getDataBand (char * buf, int row, int col, int & numEl, int band);
    void
    setData (char * buf, int row, int col, int numEl);
    void
    setDataBand (char * buf, int row, int col, int numEl, int band);
    void
    rewindAccessor ();
    void
    createFile (int numberOfLine);
    int
    getFileLength ();
    void
    finalize ();
protected:
    GDALDataset * FileObject;
    GDALDriver * Driver;
    //Use default vrt but it into a variable in case we want to use different ones in the future
    string DriverName;
    streampos LastPosition;
    //gdal enum datasettype
    GDALDataType DataType;
    streampos FileSize;
};

#endif //GDALAccessor_h
