#ifndef AccessorFactory_h
#define AccessorFactory_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <iostream>
#include <stdlib.h>
#include "DataAccessor.h"
#include <string>
using namespace std;

class AccessorFactory
{
public:
  AccessorFactory()
  {
  }
  ~AccessorFactory()
  {
  }
  DataAccessor *
  createAccessor(string filename, string accessMode, int size, int bands,
      int width, string interleved); //used for no caster
  DataAccessor *
  createAccessor(string filename, string accessMode, int size, int bands,
      int width, string interleved, string caster); //used for caster
  DataAccessor *
  createAccessor(string filename, string accessMode, int size, int bands,
      int width, string interleaved, string caster, float xmi, float xmq,
      int iqflip);
  DataAccessor *
  createAccessor(void * poly, string interleaved, int width, int length,
      int dataSize);
  void
  finalize(DataAccessor * accessor);
private:
};

#endif //AccessorFactory_h
