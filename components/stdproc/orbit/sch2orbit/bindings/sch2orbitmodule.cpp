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
#include "sch2orbitmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for sch2orbit";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "sch2orbit",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    sch2orbit_methods,
};

// initialization function for the module
// *must* be called PyInit_sch2orbit
PyMODINIT_FUNC
PyInit_sch2orbit()
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
PyObject * allocateArrays_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    if(!PyArg_ParseTuple(args, "i", &dim1))
    {
        return NULL;
    }
    allocateArrays_f(&dim1);
    return Py_BuildValue("i", 0);
}

PyObject * deallocateArrays_C(PyObject* self, PyObject* args)
{
    deallocateArrays_f();
    return Py_BuildValue("i", 0);
}

PyObject * sch2orbit_C(PyObject* self, PyObject* args)
{
    sch2orbit_f();
    return Py_BuildValue("i", 0);
}
PyObject * setOrbitPosition_C(PyObject* self, PyObject* args)
{
    PyObject * list;
    int dim1 = 0;
    if(!PyArg_ParseTuple(args, "Oi", &list, &dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                ". Expecting a list type object" << endl;
        exit(1);
    }
    double *  vectorV = new double[dim1*3];
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * listEl = PyList_GetItem(list,i);
        if(!PyList_Check(listEl))
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Expecting a list type object" << endl;
            exit(1);
        }
        for(int j = 0; j < 3; ++j)
        {
            PyObject * listElEl = PyList_GetItem(listEl,j);
            if(listElEl == NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot retrieve list element" << endl;
                exit(1);
            }
            vectorV[3*i + j] = (double) PyFloat_AsDouble(listElEl);
            if(PyErr_Occurred() != NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot convert Py Object to C " << endl;
                exit(1);
            }
        }
    }
    setOrbitPosition_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}

PyObject * setOrbitVelocity_C(PyObject* self, PyObject* args)
{
    PyObject * list;
    int dim1 = 0;
    int dim2 = 3;
    if(!PyArg_ParseTuple(args, "Oi", &list, &dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                ". Expecting a list type object" << endl;
        exit(1);
    }
    double *  vectorV = new double[dim1*dim2];
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * listEl = PyList_GetItem(list,i);
        if(!PyList_Check(listEl))
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Expecting a list type object" << endl;
            exit(1);
        }
        for(int j = 0; j < dim2; ++j)
        {
            PyObject * listElEl = PyList_GetItem(listEl,j);
            if(listElEl == NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot retrieve list element" << endl;
                exit(1);
            }
            vectorV[dim2*i + j] = (double) PyFloat_AsDouble(listElEl);
            if(PyErr_Occurred() != NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot convert Py Object to C " << endl;
                exit(1);
            }
        }
    }
    setOrbitVelocity_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}

PyObject * setPlanetGM_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setPlanetGM_f(&var);
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
PyObject * setRadiusOfCurvature_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setRadiusOfCurvature_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * getXYZPosition_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    int dim2 = 3;
    if(!PyArg_ParseTuple(args, "i", &dim1))
    {
        return NULL;
    }
    PyObject * list1 = PyList_New(dim1);
    double *  vectorV = new double[dim1*dim2];
    getXYZPosition_f(vectorV, &dim1);
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * list2 = PyList_New(dim2);
        for(int j = 0; j  < dim2; ++j)
        {
            PyObject * listEl =  PyFloat_FromDouble(
                (double) vectorV[i*dim2 + j]);
            if(listEl == NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot set list element" << endl;
                exit(1);
            }
            PyList_SetItem(list2,j,listEl);
        }
        PyList_SetItem(list1,i,list2);
    }
    delete [] vectorV;
    return Py_BuildValue("N",list1);
}

PyObject * getXYZVelocity_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    int dim2 = 3;
    if(!PyArg_ParseTuple(args, "i", &dim1))
    {
        return NULL;
    }
    PyObject * list1 = PyList_New(dim1);
    double *  vectorV = new double[dim1*dim2];
    getXYZVelocity_f(vectorV, &dim1);
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * list2 = PyList_New(dim2);
        for(int j = 0; j  < dim2; ++j)
        {
            PyObject * listEl =  PyFloat_FromDouble(
                (double) vectorV[i*dim2 + j]);
            if(listEl == NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot set list element" << endl;
                exit(1);
            }
            PyList_SetItem(list2,j,listEl);
        }
        PyList_SetItem(list1,i,list2);
    }
    delete [] vectorV;
    return Py_BuildValue("N",list1);
}

PyObject * getXYZGravitationalAcceleration_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    int dim2 = 3;
    if(!PyArg_ParseTuple(args, "i", &dim1))
    {
        return NULL;
    }
    PyObject * list1 = PyList_New(dim1);
    double *  vectorV = new double[dim1*dim2];
    getXYZGravitationalAcceleration_f(vectorV, &dim1);
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * list2 = PyList_New(dim2);
        for(int j = 0; j  < dim2; ++j)
        {
            PyObject * listEl =  PyFloat_FromDouble(
                (double) vectorV[i*dim2 + j]);
            if(listEl == NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot set list element" << endl;
                exit(1);
            }
            PyList_SetItem(list2,j,listEl);
        }
        PyList_SetItem(list1,i,list2);
    }
    delete [] vectorV;
    return Py_BuildValue("N",list1);
}
