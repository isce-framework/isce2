#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

int roi_exit(int exit_flag, char* file, long linenum)
{
  cout << "Exit function "<< file << " at line number " << linenum << endl;
  cout << "Status flag = " << exit_flag;
  return exit_flag;
}
