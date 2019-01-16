#include <Python.h>
#include "correlationmodule.h"
#include "DataAccessor.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <fstream>
using namespace std;

static char * const __doc__ = "Python extension for correlation";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "correlationlib",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    correlation_methods,
};

// initialization function for the module
PyMODINIT_FUNC
PyInit_correlationlib()
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



PyObject* correlation_C(PyObject* self, PyObject* args)
{
    DataAccessor *intAcc, *ampAcc, *cohAcc;
    int flag, bx;
    uint64_t ptr1, ptr2, ptr3;

    if ( !PyArg_ParseTuple(args,"iKKKi",&flag,&ptr1,&ptr2,&ptr3,&bx))
    {
        return NULL;
    }
    intAcc = (DataAccessor*) ptr1;
    ampAcc = (DataAccessor*) ptr2;
    cohAcc = (DataAccessor*) ptr3;

    cchz_wave(flag, intAcc, ampAcc, cohAcc, bx);

    return Py_BuildValue("i",0); 
}

