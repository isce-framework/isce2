//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
// Author: Giangi Sacco
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "correct_geoid_i2_srtmmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "module for correct_geoid_i2_srtm.f";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "correct_geoid_i2_srtm",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    correct_geoid_i2_srtm_methods,
};

// initialization function for the module
// *must* be called PyInit_correct_geoid_i2_srtm
PyMODINIT_FUNC
PyInit_correct_geoid_i2_srtm()
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

PyObject * correct_geoid_i2_srtm_C(PyObject* self, PyObject* args)
{
    uint64_t var0;
    uint64_t var1;
    if(!PyArg_ParseTuple(args, "KK",&var0,&var1))
    {
        return NULL;
    }
    correct_geoid_i2_srtm_f(&var0,&var1);
    return Py_BuildValue("i", 0);
}

PyObject * setStdWriter_C(PyObject* self, PyObject* args)
{
    uint64_t var0;
    if(!PyArg_ParseTuple(args, "K",&var0))
    {
        return NULL;
    }
    setStdWriter_f(&var0);
    return Py_BuildValue("i", 0);
}

PyObject * setWidth_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setWidth_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject * setStartLatitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setStartLatitude_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject * setStartLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setStartLongitude_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject * setDeltaLatitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setDeltaLatitude_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject * setDeltaLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setDeltaLongitude_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject * setNumberLines_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNumberLines_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject * setConversionType_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setConversionType_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject * setGeoidFilename_C(PyObject* self, PyObject* args)
{
    char * varChar;
    Py_ssize_t var;
    if(!PyArg_ParseTuple(args, "s#", &varChar ,&var))
    {
        return NULL;
    }
    int ivar = Py_SAFE_DOWNCAST(var, Py_ssize_t, int);
    setGeoidFilename_f(varChar, &ivar);
    return Py_BuildValue("i", 0);
}

PyObject * setNullIsWater_C(PyObject* self, PyObject* args)
{
    int var;
    if (!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNullIsWater_f(&var);
    return Py_BuildValue("i", 0);
}
