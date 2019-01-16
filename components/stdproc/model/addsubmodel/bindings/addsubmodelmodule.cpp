//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//# Author: Piyush Agram
//# Copyright 2013, by the California Institute of Technology. ALL RIGHTS RESERVED.
//# United States Government Sponsorship acknowledged.
//# Any commercial use must be negotiated with the Office of Technology Transfer at
//# the California Institute of Technology.
//# This software may be subject to U.S. export control laws.
//# By accepting this software, the user agrees to comply with all applicable U.S.
//# export laws and regulations. User has the responsibility to obtain export licenses,
//# or other export authority as may be required before exporting such information to
//# foreign countries or providing access to foreign persons.
//#
//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <Python.h>
#include "DataAccessor.h"
#include "addsubmodelmodule.h"
#include "addsubmodel.h"
#include <iostream>
#include <string>
#include <stdint.h>
using namespace std;


static const char * const __doc__ = "Python extension for addsubmodel";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "addsubmodel",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    addsubmodel_methods,
};

// initialization function for the module
PyMODINIT_FUNC
PyInit_addsubmodel()
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

PyObject * createaddsubmodel_C(PyObject* self, PyObject *args)
{
    addsubmodel* ptr = new addsubmodel;
    return Py_BuildValue("K", (uint64_t) ptr);
}

PyObject * destroyaddsubmodel_C(PyObject* self, PyObject *args)
{
    uint64_t ptr;
    if(!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }

    if(((addsubmodel*)(ptr)) != NULL)
    {
        delete ((addsubmodel*)(ptr));
    }
    return Py_BuildValue("i", 0);
}


PyObject * setDims_C(PyObject * self, PyObject *args)
{
    uint64_t ptr = 0;
    int wid, len;
    if(!PyArg_ParseTuple(args, "Kii", &ptr, &wid, &len))
    {
        return NULL;
    }

    ((addsubmodel*)(ptr))->setDims(wid,len);
    return Py_BuildValue("i", 0);
}

PyObject * setScaleFactor_C(PyObject *self, PyObject* args)
{
      uint64_t ptr=0;
      float scl=0.0;
      if(!PyArg_ParseTuple(args,"Kf", &ptr, &scl))
      {
          return NULL;
      }

      ((addsubmodel*)(ptr))->setScaleFactor(scl);
      return Py_BuildValue("i",0);
}

PyObject * setFlip_C(PyObject *self, PyObject *args)
{
    uint64_t ptr = 0;
    int flag = 0;
    if(!PyArg_ParseTuple(args,"Kf", &ptr, &flag))
    {
        return NULL;
    }

    ((addsubmodel*)(ptr))->setFlip(flag);
    return Py_BuildValue("i",0);
}

PyObject* cpxCpxProcess_C(PyObject* self, PyObject* args)
{
    uint64_t ptr=0;
    uint64_t in= 0;
    uint64_t model=0;
    uint64_t out=0;

    if(!PyArg_ParseTuple(args,"KKKK", &ptr, &in, &model, &out))
    {
        return NULL;
    }

    ((addsubmodel*)(ptr))->cpxCpxprocess(in,model,out);
    return Py_BuildValue("i",0);
}

PyObject* cpxUnwProcess_C(PyObject* self, PyObject* args)
{
    uint64_t ptr=0;
    uint64_t in= 0;
    uint64_t model=0;
    uint64_t out=0;

    if(!PyArg_ParseTuple(args,"KKKK", &ptr, &in, &model, &out))
    {
        return NULL;
    }

    ((addsubmodel*)(ptr))->cpxUnwprocess(in,model,out);
    return Py_BuildValue("i",0);
}

PyObject* unwUnwProcess_C(PyObject* self, PyObject* args)
{
    uint64_t ptr=0;
    uint64_t in= 0;
    uint64_t model=0;
    uint64_t out=0;

    if(!PyArg_ParseTuple(args,"KKKK", &ptr, &in, &model, &out))
    {
        return NULL;
    }

    ((addsubmodel*)(ptr))->unwUnwprocess(in,model,out);
    return Py_BuildValue("i",0);
}
