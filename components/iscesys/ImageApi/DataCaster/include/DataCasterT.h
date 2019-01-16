#ifndef DataCasterT_h
#define DataCasterT_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <stdint.h>

using namespace std;
template<typename F, typename T>
  class DataCasterT
  {
  public:

  public:
    DataCasterT()
    {
    }
    virtual
    ~DataCasterT()
    {
    }
    virtual void
    convert(char * in, char * out, int numEl) = 0;
    int
    getDataSizeIn()
    {
      return DataSizeIn;
    }
    int
    getDataSizeOut()
    {
      return DataSizeOut;
    }
  protected:
    int DataSizeIn;
    int DataSizeOut;
    void * TCaster;

  };
#endif //DataCasterT_h
