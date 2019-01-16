//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// United States Government Sponsorship acknowledged. This software is subject to
// U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
// (No [Export] License Required except when exporting to an embargoed country,
// end user, or in support of a prohibited end use). By downloading this software,
// the user agrees to comply with all applicable U.S. export laws and regulations.
// The user has the responsibility to obtain export licenses, or other export
// authority as may be required before exporting this software to any 'EAR99'
// embargoed foreign country or citizen of those countries.
//
// Author: Piyush Agram
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#include <Python.h>
#include "aikimamodule.h"
#include <sstream>
#include <iostream>
#include <fstream>
using namespace std;

static const char * const __doc__ = "Python extension for aikima";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "aikima",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    aikima_methods,
};

// initialization function for the module
PyMODINIT_FUNC
PyInit_aikima()
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


//Create the python wrapper function that interfaces to the c/fortran function
PyObject * aikima_C(PyObject *self, PyObject *args)
{
    //temporary variables to handle the arguments passed from python
    uint64_t imgin;
    uint64_t imgout;
    int inband, outband;
    if(!PyArg_ParseTuple(args, "KKii", &imgin, &imgout, &inband, &outband))
    {
        return NULL;
    }

    //make the actual call
    aikima_f(&imgin, &imgout, &inband, &outband);

   //return success
   return Py_BuildValue("i", 0);
}


PyObject* setWidth_C(PyObject* self, PyObject* args)
{
    int width;
    if(!PyArg_ParseTuple(args,"i", &width))
    {
        return NULL;
    }

    setWidth_f(&width);
    return Py_BuildValue("i",0);
}

PyObject* setLength_C(PyObject* self, PyObject* args)
{
    int width;
    if(!PyArg_ParseTuple(args,"i",&width))
    {
        return NULL;
    }

    setLength_f(&width);
    return Py_BuildValue("i",0);
}

PyObject* setBlockSize_C(PyObject* self, PyObject* args)
{
    int width;
    if(!PyArg_ParseTuple(args,"i",&width))
    {
        return NULL;
    }

    setBlockSize_f(&width);
    return Py_BuildValue("i",0);
}

PyObject* setPadSize_C(PyObject* self, PyObject* args)
{
    int width;
    if(!PyArg_ParseTuple(args,"i",&width))
    {
        return NULL;
    }

    setPadSize_f(&width);
    return Py_BuildValue("i",0);
}

PyObject* setFirstPixelAcross_C(PyObject* self, PyObject* args)
{
    int width;
    if(!PyArg_ParseTuple(args,"i",&width))
    {
        return NULL;
    }

    width = width+1;
    setFirstPixelAcross_f(&width);
    return Py_BuildValue("i",0);
}

PyObject* setLastPixelAcross_C(PyObject* self, PyObject* args)
{
    int width;
    if(!PyArg_ParseTuple(args,"i",&width))
    {
        return NULL;
    }

    setLastPixelAcross_f(&width);
    return Py_BuildValue("i",0);
}


PyObject* setFirstLineDown_C(PyObject* self, PyObject* args)
{
    int width;
    if(!PyArg_ParseTuple(args,"i",&width))
    {
        return NULL;
    }
    width = width+1;
    setFirstLineDown_f(&width);
    return Py_BuildValue("i",0);
}

PyObject* setLastLineDown_C(PyObject* self, PyObject* args)
{
    int width;
    if(!PyArg_ParseTuple(args,"i",&width))
    {
        return NULL;
    }

    setLastLineDown_f(&width);
    return Py_BuildValue("i",0);
}

PyObject* setNumberPtsPartial_C(PyObject* self, PyObject* args)
{
    int width;
    if(!PyArg_ParseTuple(args,"i",&width))
    {
        return NULL;
    }

    setNumberPtsPartial_f(&width);
    return Py_BuildValue("i",0);
}

PyObject* setPrintFlag_C(PyObject* self, PyObject* args)
{
    int width;
    if(!PyArg_ParseTuple(args,"i",&width))
    {
        return NULL;
    }

    setPrintFlag_f(&width);
    return Py_BuildValue("i",0);
}

PyObject* setThreshold_C(PyObject* self, PyObject* args)
{
    float width;
    if(!PyArg_ParseTuple(args,"f",&width))
    {
        return NULL;
    }

    setThreshold_f(&width);
    return Py_BuildValue("i",0);
}
