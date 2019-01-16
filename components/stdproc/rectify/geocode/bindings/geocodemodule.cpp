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
#include "geocodemodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for geocode";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "geocode",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    geocode_methods,
};

// initialization function for the module
// *must* be called PyInit_geocode
PyMODINIT_FUNC
PyInit_geocode()
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

PyObject * setStdWriter_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setStdWriter_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * allocate_s_mocomp_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    if(!PyArg_ParseTuple(args, "i", &dim1))
    {
        return NULL;
    }
    allocate_s_mocomp_f(&dim1);
    return Py_BuildValue("i", 0);
}

PyObject * deallocate_s_mocomp_C(PyObject* self, PyObject* args)
{
    deallocate_s_mocomp_f();
    return Py_BuildValue("i", 0);
}

PyObject * geocode_C(PyObject* self, PyObject* args)
{
    uint64_t var0;
    uint64_t var1;
    uint64_t var2;
    uint64_t var3;
    uint64_t var4;
    int b1, b2, b3,b4;
    if(!PyArg_ParseTuple(args, "KKKKKiiii", &var0, &var1, &var2, &var3, &var4,
        &b1,&b2,&b3,&b4))
    {
        return NULL;
    }
    b1++;    //Python bandnumber to Fortran bandnumber
    b2++;    //Python bandnumber to Fortrab bandnumber
    geocode_f(&var0,&var1,&var2,&var3,&var4,&b1,&b2,&b3,&b4);
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
PyObject * setPegLatitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setPegLatitude_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setPegLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setPegLongitude_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setPegHeading_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setPegHeading_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRangePixelSpacing_C(PyObject* self, PyObject* args)
{
    float var;
    if(!PyArg_ParseTuple(args, "f", &var))
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
PyObject * setHeight_C(PyObject* self, PyObject* args)
{
    float var;
    if(!PyArg_ParseTuple(args, "f", &var))
    {
        return NULL;
    }
    setHeight_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setPlanetLocalRadius_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setPlanetLocalRadius_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setVelocity_C(PyObject* self, PyObject* args)
{
    float var;
    if(!PyArg_ParseTuple(args, "f", &var))
    {
        return NULL;
    }
    setVelocity_f(&var);
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
    float var;
    if(!PyArg_ParseTuple(args, "f", &var))
    {
        return NULL;
    }
    setPRF_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRadarWavelength_C(PyObject* self, PyObject* args)
{
    float var;
    if(!PyArg_ParseTuple(args, "f", &var))
    {
        return NULL;
    }
    setRadarWavelength_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setSCoordinateFirstLine_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setSCoordinateFirstLine_f(&var);
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
PyObject * setNumberPointsPerDemPost_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNumberPointsPerDemPost_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setISMocomp_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setISMocomp_f(&var);
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
PyObject * setReferenceOrbit_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    PyObject * list;
    if(!PyArg_ParseTuple(args, "Oi", &list,&dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
        exit(1);
    }
    double *  vectorV = new double[dim1];
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * listEl = PyList_GetItem(list,i);
        if(listEl == NULL)
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
            exit(1);
        }
        vectorV[i] = (double) PyFloat_AsDouble(listEl);
        if(PyErr_Occurred() != NULL)
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setReferenceOrbit_f(vectorV, &dim1);
    delete [] vectorV;
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
PyObject * getLatitudeSpacing_C(PyObject* self, PyObject* args)
{
    double var;
    getLatitudeSpacing_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getLongitudeSpacing_C(PyObject* self, PyObject* args)
{
    double var;
    getLongitudeSpacing_f(&var);
    return Py_BuildValue("d",var);
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
PyObject * getMaxmumGeoLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    getMaxmumGeoLongitude_f(&var);
    return Py_BuildValue("d",var);
}

// end of file
