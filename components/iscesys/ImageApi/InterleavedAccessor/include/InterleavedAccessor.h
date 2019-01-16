#ifndef InterleavedAccessor_h
#define InterleavedAccessor_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include "InterleavedBase.h"
using namespace std;

class InterleavedAccessor : public InterleavedBase
{
public:
    InterleavedAccessor () :
	    InterleavedBase ()
    {
    }
    virtual
    ~InterleavedAccessor ()
    {
    }
    void
    init (string filename, string accessMode, int sizeV, int Bands,
	  int LineWidth);
    void
    openFile (string filename, string accessMode, fstream & fd);
    fstream &
    getFileObject ()
    {
	return FileObject;
    }
    virtual void
    init (void * poly) = 0;

    void
    getStreamAtPos (char * buf, int & pos, int & numEl);
    void
    setStreamAtPos (char * buf, int & pos, int & numEl);
    void
    getStream (char * buf, int & numEl);
    void
    setStream (char * buf, int numEl);
    void
    rewindAccessor ();
    void
    createFile (int numberOfLine);
    int
    getFileLength ();
    void
    finalize ();

protected:

    /**
     * Stream associated with the image file.
     *
     **/
    fstream FileObject;

};

#endif //InterleavedAccessor_h
