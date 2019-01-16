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




#include <Python.h>
#include "topomodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for topo.F";

PyModuleDef moduledef = {
    //header
    PyModuleDef_HEAD_INIT,
    //name of the module
    "topo",
    //module documentation string
    __doc__,
    //size of the per-interpreter state of the module
    //-1 if this state is global
    -1,
    topo_methods,
};

//initialization function for the module
//// *must* be called PyInit_topo
PyMODINIT_FUNC
PyInit_topo()
{
    //create the module using moduledef struct defined above
    PyObject * module = PyModule_Create(&moduledef);
    //check whether module create succeeded and raise exception if not
    if(!module)
    {
        return module;
    }
    //otherwise we have an initialized module
    //and return the newly created module
    return module;
}

PyObject * allocate_s_mocompArray_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    if(!PyArg_ParseTuple(args, "i", &dim1))
    {
        return NULL;
    }
    allocate_s_mocompArray_f(&dim1);
    return Py_BuildValue("i", 0);
}

PyObject * deallocate_s_mocompArray_C(PyObject* self, PyObject* args)
{
    deallocate_s_mocompArray_f();
    return Py_BuildValue("i", 0);
}

PyObject * allocate_squintshift_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    if(!PyArg_ParseTuple(args, "i", &dim1))
    {
        return NULL;
    }
    allocate_squintshift_f(&dim1);
    return Py_BuildValue("i", 0);
}

PyObject * deallocate_squintshift_C(PyObject* self, PyObject* args)
{
    deallocate_squintshift_f();
    return Py_BuildValue("i", 0);
}

PyObject * topo_C(PyObject* self, PyObject* args)
{
    uint64_t var0, var1;
    if(!PyArg_ParseTuple(args, "KK",&var0,&var1))
    {
        return NULL;
    }
    topo_f(&var0, &var1);
    return Py_BuildValue("i", 0);
}
PyObject * setNumberIterations_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNumberIterations_f(&var);
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
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                ". Expecting a list type object" << endl;
        exit(1);
    }
    double *  vectorV = new double[dim1];
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * listEl = PyList_GetItem(list,i);
        if(listEl == NULL)
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Cannot retrieve list element" << endl;
            exit(1);
        }
        vectorV[i] = (double) PyFloat_AsDouble(listEl);
        if(PyErr_Occurred() != NULL)
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setReferenceOrbit_f(vectorV, &dim1);
    delete [] vectorV;
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
PyObject * setLookSide_C(PyObject* self, PyObject *args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
    return NULL;
    }
    setLookSide_f(&var);
    return Py_BuildValue("i",0);
}
PyObject * setMethod_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args,"i",&var))
    {
        return NULL;
    }
    setMethod_f(&var);
    return Py_BuildValue("i",0);
}
PyObject * setSpacecraftHeight_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setSpacecraftHeight_f(&var);
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
PyObject * setBodyFixedVelocity_C(PyObject* self, PyObject* args)
{
    float var;
    if(!PyArg_ParseTuple(args, "f", &var))
    {
        return NULL;
    }
    setBodyFixedVelocity_f(&var);
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
PyObject * setLatitudePointer_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setLatitudePointer_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setLongitudePointer_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setLongitudePointer_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject * setHeightRPointer_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setHeightRPointer_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setHeightSchPointer_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setHeightSchPointer_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setLosPointer_C(PyObject* self, PyObject *args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args,"K", &var))
    {
    return NULL;
    }
    setLosPointer_f(&var);
    return Py_BuildValue("i",0);
}
PyObject * setIncPointer_C(PyObject* self, PyObject *args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args,"K", &var))
    {
    return NULL;
    }
    setIncPointer_f(&var);
    return Py_BuildValue("i",0);
}
PyObject * getAzimuthSpacing_C(PyObject* self, PyObject* args)
{
    double var;
    getAzimuthSpacing_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getPlanetLocalRadius_C(PyObject* self, PyObject* args)
{
    double var;
    getPlanetLocalRadius_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getSCoordinateFirstLine_C(PyObject* self, PyObject* args)
{
    double var;
    getSCoordinateFirstLine_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getSCoordinateLastLine_C(PyObject* self, PyObject* args)
{
    double var;
    getSCoordinateLastLine_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getMinimumLatitude_C(PyObject* self, PyObject* args)
{
    double var;
    getMinimumLatitude_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getMinimumLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    getMinimumLongitude_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getMaximumLatitude_C(PyObject* self, PyObject* args)
{
    double var;
    getMaximumLatitude_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getMaximumLongitude_C(PyObject* self, PyObject* args)
{
    double var;
    getMaximumLongitude_f(&var);
    return Py_BuildValue("d",var);
}
PyObject * getLength_C(PyObject* self, PyObject *args)
{
    int var;
    getLength_f(&var);
    return Py_BuildValue("i",var);
}
PyObject * getSquintShift_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    if(!PyArg_ParseTuple(args, "i", &dim1))
    {
        return NULL;
    }
    PyObject * list = PyList_New(dim1);
    double *  vectorV = new double[dim1];
    getSquintShift_f(vectorV, &dim1);
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
        if(listEl == NULL)
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Cannot set list element" << endl;
            exit(1);
        }
        PyList_SetItem(list,i, listEl);
    }
    delete [] vectorV;
    return Py_BuildValue("N",list);
}

PyObject* setSensingStart_C(PyObject *self, PyObject *args)
{
    double tstart;
    if(!PyArg_ParseTuple(args,"d", &tstart))
    {
        return NULL;
    }

    setSensingStart_f(&tstart);
    return Py_BuildValue("i", 0);
}

PyObject *setOrbit_C(PyObject *self, PyObject *args)
{
    uint64_t cptr;
    cOrbit* corb;

    if(!PyArg_ParseTuple(args,"K", &cptr))
    {
        return NULL;
    }
    corb = (cOrbit*) cptr;

    setOrbit_f(corb);
    return Py_BuildValue("i", 0);
}


// end of file
