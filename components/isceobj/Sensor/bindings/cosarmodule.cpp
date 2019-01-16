#include <Python.h>
#include "cosarmodule.h"

using namespace std;

static const char* const __doc__ = "Python extension for cosar";

PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cosar",
    __doc__,
    -1,
    cosar_methods};

PyMODINIT_FUNC
PyInit_cosar()
{
       // create the module using moduledef struct defined above
      PyObject * module = PyModule_Create(&moduledef);
      // check whether module creation succeeded and raise an exception if not
      if (!module) {
        return module;
      }
      // otherwise, we have an initialized module
      // and return the newly created module
      return module;
}

PyObject *cosar_C(PyObject *self,PyObject *args)
{
        char *input,*output;
        Cosar *cosar;
        if(!PyArg_ParseTuple(args,"ss",&input,&output))
        {
                return NULL;
        }
        cosar = new Cosar(input,output);
        cosar->parse();

        return Py_BuildValue("i",0);
}
