//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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




#include <Python.h>
#include "geozeromodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for geozero";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "geozero",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    geozero_methods,
};

// initialization function for the module
// *must* be called PyInit_geozero
PyMODINIT_FUNC
PyInit_geozero()
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

PyObject * geozero_C(PyObject* self, PyObject* args)
{
    uint64_t var0;
    uint64_t var1;
    uint64_t var2;
    uint64_t var3;
    int b1, b2, b3, b4, b5;
    if(!PyArg_ParseTuple(args, "KKKKiiiii", &var0, &var1, &var2, &var3,
        &b1,&b2,&b3,&b4,&b5))
    {
        return NULL;
    }
    b1++;    //Python bandnumber to Fortran bandnumber
    b2++;    //Python bandnumber to Fortran bandnumber
    geozero_f(&var0,&var1,&var2,&var3,&b1,&b2,&b3,&b4,&b5);
    return Py_BuildValue("i", 0);
}
PyObject * setEllipsoidMajorSemiAxis_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setEllipsoidMajorSemiAxis_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setEllipsoidEccentricitySquared_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setEllipsoidEccentricitySquared_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setMinimumLatitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setMinimumLatitude_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setMinimumLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setMinimumLongitude_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setMaximumLatitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setMaximumLatitude_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setLookSide_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
    return NULL;
    }
    setLookSide_f(&var);
    return Py_BuildValue("i",0);
}
PyObject * setMaximumLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setMaximumLongitude_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRangePixelSpacing_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setRangePixelSpacing_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRangeFirstSample_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setRangeFirstSample_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setDopplerAccessor_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    cPoly1d* varptr;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    varptr = (cPoly1d*) var;
    setDopplerAccessor_f(varptr);
    return Py_BuildValue("i", 0);
}
PyObject * setPRF_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setPRF_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRadarWavelength_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setRadarWavelength_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setSensingStart_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setSensingStart_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setFirstLatitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setFirstLatitude_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setFirstLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setFirstLongitude_f(&var);
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
PyObject * setLength_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setLength_f(&var);
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
PyObject * setNumberRangeLooks_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNumberRangeLooks_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setNumberAzimuthLooks_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNumberAzimuthLooks_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setDemWidth_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setDemWidth_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setDemLength_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setDemLength_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setOrbit_C(PyObject* self, PyObject* args)
{
    uint64_t orbPtr;
    cOrbit* ptr;

    if(!PyArg_ParseTuple(args, "K", &orbPtr))
    {
        return NULL;
    }

    ptr = (cOrbit*) orbPtr;
    setOrbit_f(ptr);

    return Py_BuildValue("i", 0);
}

PyObject * getGeoWidth_C(PyObject* self, PyObject* args)
{
    int var;
    getGeoWidth_f(&var);
    return Py_BuildValue("i",var);
}
PyObject * getGeoLength_C(PyObject* self, PyObject* args)
{
    int var;
    getGeoLength_f(&var);
    return Py_BuildValue("i",var);
}
PyObject * getMinimumGeoLatitude_C(PyObject* self, PyObject* args)
{
    double var;
    getMinimumGeoLatitude_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getMinimumGeoLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    getMinimumGeoLongitude_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getMaximumGeoLatitude_C(PyObject* self, PyObject* args)
{
    double var;
    getMaximumGeoLatitude_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getMaximumGeoLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    getMaximumGeoLongitude_f(&var);
    return Py_BuildValue("d",var);
}

// end of file
