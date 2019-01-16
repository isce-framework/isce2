#include <string>
#include <fstream>

/**!\class Header
 *  \brief A class to hold COSAR header data
 *  \author Walter Szeliga
 *  \date 23 Sep. 2010
 */
class Header
{
 private:
   bool isBigEndian;
   char format[5];
   int bytesInBurst;
   int rangeSampleRelativeIndex;
   int rangeSamples;
   int azimuthSamples;
   int burstIndex;
   int rangelineTotalNumberOfBytes;
   int totalNumberOfLines;
   int version;
   int oversamplingFactor;
   double inverseSPECANScalingRate;
 public:
  Header(bool isBigEndian);
  ~Header();
  void parse(std::istream &fin);
  void print();
  int getRangeSamples();
  int getAzimuthSamples();
  int getRangelineTotalNumberOfBytes();
  int getTotalNumberOfLines();
  int getBytesInBurst();
};
