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
#include "offoutliersmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for offoutliers.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "offoutliers",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    offoutliers_methods,
};

// initialization function for the module
PyMODINIT_FUNC
PyInit_offoutliers()
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
PyObject * allocate_indexArray_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_indexArray_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_indexArray_C(PyObject* self, PyObject* args)
{
        deallocate_indexArray_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_xd_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_xd_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_xd_C(PyObject* self, PyObject* args)
{
        deallocate_xd_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_acshift_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_acshift_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_acshift_C(PyObject* self, PyObject* args)
{
        deallocate_acshift_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_yd_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_yd_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_yd_C(PyObject* self, PyObject* args)
{
        deallocate_yd_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_dnshift_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_dnshift_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_dnshift_C(PyObject* self, PyObject* args)
{
        deallocate_dnshift_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_sig_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_sig_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_sig_C(PyObject* self, PyObject* args)
{
        deallocate_sig_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_s_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_s_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_s_C(PyObject* self, PyObject* args)
{
        deallocate_s_f();
        return Py_BuildValue("i", 0);
}

PyObject * offoutliers_C(PyObject* self, PyObject* args)
{
        offoutliers_f();
        return Py_BuildValue("i", 0);
}
PyObject * getIndexArray_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        int *  vectorV = new int[dim1];
        getIndexArray_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyLong_FromLong((long int) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("N",list);
}

PyObject * getIndexArraySize_C(PyObject* self, PyObject* args)
{
        int var;
        getIndexArraySize_f(&var);
        return Py_BuildValue("i",var);
}
PyObject * getAverageOffsetDown_C(PyObject* self, PyObject* args)
{
        float var;
        getAverageOffsetDown_f(&var);
        return Py_BuildValue("f",var);
}
PyObject * getAverageOffsetAcross_C(PyObject* self, PyObject* args)
{
        float var;
        getAverageOffsetAcross_f(&var);
        return Py_BuildValue("f",var);
}
PyObject * setNumberOfPoints_C(PyObject* self, PyObject* args)
{
        int var;
        if(!PyArg_ParseTuple(args, "i", &var))
        {
                return NULL;
        }
        setNumberOfPoints_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * setLocationAcross_C(PyObject* self, PyObject* args)
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
        setLocationAcross_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setLocationAcrossOffset_C(PyObject* self, PyObject* args)
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
        setLocationAcrossOffset_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setLocationDown_C(PyObject* self, PyObject* args)
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
        setLocationDown_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setLocationDownOffset_C(PyObject* self, PyObject* args)
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
        setLocationDownOffset_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setDistance_C(PyObject* self, PyObject* args)
{
        float var;
        if(!PyArg_ParseTuple(args, "f", &var))
        {
                return NULL;
        }
        setDistance_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * setSign_C(PyObject* self, PyObject* args)
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
        setSign_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setSNR_C(PyObject* self, PyObject* args)
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
        setSNR_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}
