#include <fstream>

/**!\class Burst
 *  \brief A class to hold data for a COSAR burst
 *  \author Walter Szeliga
 *  \date 23 Sep. 2010
 */
class Burst
{
 private:
   bool isBigEndian;
   int rangeSamples;
   int azimuthSamples;
   int *asri;
   int *asfv;
   int *aslv;
 public:
  Burst(int rangeSamples,int azimuthSamples,bool isBigEndian);
  ~Burst();
  void parse(std::istream &fin,std::ostream &fout); 
  void parseAzimuthHeader(std::istream &fin);
  void parseRangeLine(std::istream &fin,std::ostream &fout,int lineNumber);
};
