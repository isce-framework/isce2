#include <fftw3.h>
int sfftw_import_wisdom_from_filename(const char* filename)
{
  FILE * fp = 0;
  int ret = 0;
  fp = fopen(filename,"r");
  ret = fftwf_import_wisdom_from_file(fp);
  fclose(fp);
  return ret;
}
