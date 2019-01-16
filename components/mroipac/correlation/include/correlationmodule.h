#if !defined(__MROIPAC_CORRELATIONMODULE_H__)
#define __MROIPAC_CORRELATIONMODULE_H__

#include <Python.h>
#include "DataAccessor.h"

extern "C"
{
  //the fortran engine

  PyObject* correlation_C(PyObject*, PyObject*);
  
}

  int cchz_wave(int, DataAccessor*, DataAccessor*, DataAccessor*, int);

//Method Table
static PyMethodDef correlation_methods[] =
{
 {"correlation_Py", correlation_C, METH_VARARGS, " "},
 {NULL, NULL, 0 , NULL}
};

#endif

//end of file
