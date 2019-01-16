#include <string>
#include <fstream>
#include "Header.hh"
#include "Burst.hh"

/**!\class Cosar
 *  \brief A class to parse COSAR files.
 *  \author Walter Szeliga
 *  \date 23 Sep. 2010
 */
class Cosar
{
 private:
   bool isBigEndian;
   int numberOfBursts;
   std::ifstream fin;
   std::ofstream fout;
   Header *header;
   Burst **bursts;

 public:
   Cosar(std::string input,std::string output);
   ~Cosar();
   void parse();
};
