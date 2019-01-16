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
#include "bistaticgeo2rdrmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for bistaticgeo2rdr";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "bistaticgeo2rdr",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    bistaticgeo2rdr_methods,
};

// initialization function for the module
// *must* be called PyInit_geo2rdr
PyMODINIT_FUNC
PyInit_bistaticgeo2rdr()
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

PyObject * bistaticgeo2rdr_C(PyObject* self, PyObject* args)
{
    uint64_t var0;
    uint64_t var1;
    uint64_t var2;
    uint64_t var3;
    uint64_t var4;
    uint64_t var5;
    uint64_t var6;
    if(!PyArg_ParseTuple(args, "KKKKKKK", &var0, &var1, &var2, &var3,
        &var4,&var5,&var6))
    {
        return NULL;
    }
    bistaticgeo2rdr_f(&var0,&var1,&var2,&var3,&var4,&var5,&var6);
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
PyObject * setActiveRangeFirstSample_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setActiveRangeFirstSample_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setPassiveRangeFirstSample_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setPassiveRangeFirstSample_f(&var);
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
PyObject * setActiveOrbit_C(PyObject* self, PyObject* args)
{
    uint64_t orbPtr;
    cOrbit* ptr;

    if(!PyArg_ParseTuple(args, "K", &orbPtr))
    {
        return NULL;
    }

    ptr = (cOrbit*) orbPtr;
    setActiveOrbit_f(ptr);

    return Py_BuildValue("i", 0);
}
PyObject * setPassiveOrbit_C(PyObject* self, PyObject* args)
{
    uint64_t orbPtr;
    cOrbit* ptr;

    if(!PyArg_ParseTuple(args, "K", &orbPtr))
    {
        return NULL;
    }

    ptr = (cOrbit*) orbPtr;
    setPassiveOrbit_f(ptr);

    return Py_BuildValue("i", 0);
}
PyObject * setBistaticFlag_C(PyObject *self, PyObject* args)
{
    int flag;

    if (!PyArg_ParseTuple(args,"i", &flag))
    {
        return NULL;
    }

    setBistaticFlag_f(&flag);
    return Py_BuildValue("i",0);
}
PyObject * setOrbitMethod_C(PyObject *self, PyObject* args)
{
    int flag;

    if (!PyArg_ParseTuple(args,"i", &flag))
    {
        return NULL;
    }

    setOrbitMethod_f(&flag);
    return Py_BuildValue("i",0);
}
// end of file
