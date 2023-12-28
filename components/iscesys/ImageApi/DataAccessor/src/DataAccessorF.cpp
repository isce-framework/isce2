#include "DataAccessorF.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

extern "C" {

void rewindAccessor_f(uint64_t * ptDataAccessor)
{
	((DataAccessor * ) (* ptDataAccessor))->rewindAccessor();
}


int getWidth_f(uint64_t * ptDataAccessor)
{
   return ((DataAccessor * ) (* ptDataAccessor))->getWidth();
}
int getNumberOfLines_f(uint64_t * ptDataAccessor)
{
    return ((DataAccessor * ) (* ptDataAccessor))->getNumberOfLines();
}
void setLineSequential_f(uint64_t * ptDataAccessor,  char * dataLine) 
{
    ((DataAccessor * ) (* ptDataAccessor))->setLineSequential(dataLine);
}
void setlinesequential_c_(uint64_t* acc, char* data)
{
    setLineSequential_f(acc, data);
}
void setlinesequential_r4_(uint64_t* acc, char* data)
{
    setLineSequential_f(acc, data);
}
void setlinesequential_r8_(uint64_t* acc, char* data)
{
    setLineSequential_f(acc, data);
}
void setLineSequentialBand_f(uint64_t * ptDataAccessor, char * dataLine, int * band)
{
    (*band) -=1;
    ((DataAccessor * ) (* ptDataAccessor))->setLineSequentialBand(dataLine, (*band));
    (*band) +=1;
}
void setlinesequentialband_c_(uint64_t* acc, char* data, int* band)
{
    setLineSequentialBand_f(acc, data, band);
}
void setlinesequentialband_r4_(uint64_t* acc, char* data, int* band)
{
    setLineSequentialBand_f(acc, data, band);
}
void getLineSequential_f(uint64_t * ptDataAccessor,  char * dataLine, int * ptFlag) 
{
    (*ptFlag) =  ((DataAccessor * ) (* ptDataAccessor))->getLineSequential(dataLine);
}
void getlinesequential_c_(uint64_t* acc,  char* data, int* flag)
{
    getLineSequential_f(acc, data, flag);
}
void getlinesequential_r4_(uint64_t* acc,  char* data, int* flag)
{
    getLineSequential_f(acc, data, flag);
}
void getlinesequential_r8_(uint64_t* acc,  char* data, int* flag)
{
    getLineSequential_f(acc, data, flag);
}
void getLineSequentialBand_f(uint64_t * ptDataAccessor, char * dataLine, int *band, int *ptFlag)
{
    (*band) -=1;
    int flag = ((DataAccessor * ) (* ptDataAccessor))->getLineSequentialBand(dataLine, (*band));
    (*band) +=1;
    (*ptFlag) = flag;
}
void getlinesequentialband_c_(uint64_t* acc, char* data, int *band, int* flag)
{
    getLineSequentialBand_f(acc, data, band, flag);
}
void getlinesequentialband_r4_(uint64_t* acc, char* data, int *band, int* flag)
{
    getLineSequentialBand_f(acc, data, band, flag);
}
void setLine_f(uint64_t * ptDataAccessor,  char * dataLine, int * ptLine) 
{
    // fortran is one based
    (*ptLine) -= 1;
    ((DataAccessor * ) (* ptDataAccessor))->setLine(dataLine, (*ptLine));
    (*ptLine) += 1;
}
void setline_c_(uint64_t* acc,  char* data, int* line)
{
    setLine_f(acc, data, line);
}
void setline_r4_(uint64_t* acc,  char* data, int* line)
{
    setLine_f(acc, data, line);
}
void setLineBand_f(uint64_t * ptDataAccessor, char * dataLine, int * ptLine, int * band)
{
    (*ptLine) -= 1;
    (*band) -= 1;
    ((DataAccessor * ) (* ptDataAccessor))->setLineBand(dataLine, (*ptLine), (*band));
    (*ptLine) += 1;
    (*band) +=1;
}
void setlineband_r4_(uint64_t* acc, char* data, int* line, int* band)
{
    setLineBand_f(acc, data, line, band);
}
void getLine_f(uint64_t * ptDataAccessor,  char * dataLine, int * ptLine) 
{
    // fortran is one based
    (*ptLine) -= 1;
    int flag = ((DataAccessor * ) (* ptDataAccessor))->getLine(dataLine, (*ptLine));
    if(flag < 0)
    {
        (*ptLine) = flag;
    }
    else
    {
        (*ptLine) += 1;
    }
}
void getline_r4_(uint64_t* acc,  char* data, int* line)
{
    getLine_f(acc, data, line);
}
void getline_r8_(uint64_t* acc,  char* data, int* line)
{
    getLine_f(acc, data, line);
}
void getLineBand_f(uint64_t * ptDataAccessor, char * dataLine, int *band, int *ptLine)
{
    int ptLine1, band1;
    ptLine1 = (*ptLine) - 1;
    band1 = (*band) - 1;
    int flag = ((DataAccessor * ) (* ptDataAccessor))->getLineBand(dataLine, ptLine1, band1);
    if (flag<0)
    {
        (*ptLine) = flag;
    }
//    else
//    {
//        (*ptLine) +=1;
//    }
}
void getlineband_c_(uint64_t* acc, char* data, int* band, int* line)
{
    getLineBand_f(acc, data, band, line);
}
void getlineband_r4_(uint64_t* acc, char* data, int* band, int* line)
{
    getLineBand_f(acc, data, band, line);
}

void setSequentialElements_f(uint64_t * ptDataAccessor,  char * dataLine, int * ptRow, int * ptCol, int * ptNumEl) 
{
    // fortran is one based
    (*ptRow) -= 1;
    (*ptCol) -= 1;
    ((DataAccessor * ) (* ptDataAccessor))->setSequentialElements(dataLine, (*ptRow),(*ptCol),(*ptNumEl));
    (*ptRow) += 1;
    (*ptCol) += 1;
}
void getSequentialElements_f(uint64_t * ptDataAccessor,  char * dataLine, int * ptRow, int * ptCol, int * ptNumEl) 
{
    // fortran is one based
    (*ptRow) -= 1;
    (*ptCol) -= 1;
    ((DataAccessor * ) (* ptDataAccessor))->getSequentialElements(dataLine, (*ptRow),(*ptCol),(*ptNumEl));
    (*ptRow) += 1;
    (*ptCol) += 1;
}
void setStream_f(uint64_t * ptDataAccessor,  char * dataLine, int * numEl)
{
	((DataAccessor * ) (* ptDataAccessor))->setStream(dataLine, (*numEl));
}
void getStream_f(uint64_t * ptDataAccessor,  char * dataLine, int * numEl)
{
	((DataAccessor * ) (* ptDataAccessor))->getStream(dataLine, (*numEl));
}
void setStreamAtPos_f(uint64_t * ptDataAccessor,  char * dataLine, int * pos, int * numEl)
{
    // fortran is one based
    (*pos) -= 1;
	((DataAccessor * ) (* ptDataAccessor))->setStreamAtPos(dataLine, (*pos), (*numEl));
    (*pos) += 1;
}
void getStreamAtPos_f(uint64_t * ptDataAccessor,  char * dataLine, int * pos, int * numEl)
{
    // fortran is one based
    (*pos) -= 1;
	((DataAccessor * ) (* ptDataAccessor))->getStreamAtPos(dataLine, (*pos), (*numEl));
    (*pos) += 1;
}
void initSequentialAccessor_f(uint64_t * ptDataAccessor, int * begLine)
{
    // fortran is one based
    (*begLine) -= 1;
    ((DataAccessor * ) (* ptDataAccessor))->initSequentialAccessor((*begLine));
    (*begLine) += 1;
}
double getPx1d_f(uint64_t * ptDataAccessor,int * ptPos)
{
  (*ptPos) -= 1;
  double ret = ((DataAccessor * ) (* ptDataAccessor))->getPx1d((*ptPos));
  (*ptPos) += 1;

  return ret;

}
double getPx2d_f(uint64_t * ptDataAccessor,int * ptRow, int * ptCol)
{
  (*ptRow) -= 1;
  (*ptCol) -= 1;
  double ret = ((DataAccessor * ) (* ptDataAccessor))->getPx2d((*ptRow),(*ptCol));
  (*ptRow) += 1;
  (*ptCol) += 1;
  return ret;

}
void setLineOffset_f(uint64_t * ptDataAccessor,int * lineoff)
{
  ((DataAccessor * ) (* ptDataAccessor))->setLineOffset((*lineoff));
}
int getLineOffset_f(uint64_t * ptDataAccessor)
{
  return ((DataAccessor * ) (* ptDataAccessor))->getLineOffset();
}

}
