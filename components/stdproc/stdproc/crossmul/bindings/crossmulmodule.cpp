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
#include "crossmul.h"
#include "crossmulmodule.h"
#include <stdint.h>
using namespace std;

static const char * const __doc__ = "Python extension for crossmul.F";

PyModuleDef moduledef = {
    //header
    PyModuleDef_HEAD_INIT,
    //name of the module
    "crossmul",
    //module documentation string
    __doc__,
    //size of the per-interpreter state of the module
    //-1 if this state is global
    -1,
    crossmul_methods,
};

//initialization function for the module
//// *must* be called PyInit_crossmul
PyMODINIT_FUNC
PyInit_crossmul()
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

PyObject * createCrossMul_C(PyObject* self, PyObject* args)
{
    crossmulState* newObj = new crossmulState;
    return Py_BuildValue("K", (uint64_t) newObj);
}

PyObject * destroyCrossMul_C(PyObject* self, PyObject* args)
{
    uint64_t ptr;
    if(!PyArg_ParseTuple(args,"K",&ptr))
    {
        return NULL;
    }
    if ((crossmulState*)(ptr) != NULL)
    {
        delete ((crossmulState*)(ptr));
    }
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * setWidth_C(PyObject* self, PyObject* args)
{
    int var;
    uint64_t ptr;
    if(!PyArg_ParseTuple(args, "Ki", &ptr, &var))
    {
        return NULL;
    }
    ((crossmulState*)(ptr))->na = var;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * setLength_C(PyObject* self, PyObject* args)
{
    int var;
    uint64_t ptr;
    if(!PyArg_ParseTuple(args, "Ki", &ptr, &var))
    {
        return NULL;
    }
    ((crossmulState*)(ptr))->nd = var;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * setLooksAcross_C(PyObject* self, PyObject* args)
{
    int var;
    uint64_t ptr;
    if(!PyArg_ParseTuple(args, "Ki", &ptr, &var))
    {
        return NULL;
    }
    ((crossmulState*)(ptr))->looksac = var;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * setLooksDown_C(PyObject* self, PyObject* args)
{
    int var;
    uint64_t ptr;
    if(!PyArg_ParseTuple(args, "Ki", &ptr, &var))
    {
        return NULL;
    }
    ((crossmulState*)(ptr))->looksdn = var;
    Py_INCREF(Py_None);
    return Py_None;
}


PyObject * setScale_C(PyObject* self, PyObject* args)
{
    double var;
    uint64_t ptr;
    if(!PyArg_ParseTuple(args, "Kd", &ptr, &var))
    {
        return NULL;
    }
    ((crossmulState*)(ptr))->scale = var;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * setBlocksize_C(PyObject* self, PyObject* args)
{
    int var;
    uint64_t ptr;
    if(!PyArg_ParseTuple(args, "Ki", &ptr, &var))
    {
        return NULL;
    }
    ((crossmulState*)(ptr))->blocksize =  var;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * setWavelengths_C(PyObject* self, PyObject* args)
{
    double v1, v2;
    uint64_t ptr;
    if (!PyArg_ParseTuple(args,"Kdd", &ptr, &v1, &v2))
    {
        return NULL;
    }
    ((crossmulState*)(ptr))->wvl1 = v1;
    ((crossmulState*)(ptr))->wvl2 = v2;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * setSpacings_C(PyObject* self, PyObject* args)
{
    double v1, v2;
    uint64_t ptr;
    if (!PyArg_ParseTuple(args,"Kdd", &ptr, &v1, &v2))
    {
        return NULL;
    }
    ((crossmulState*)(ptr))->drg1 = v1;
    ((crossmulState*)(ptr))->drg2 = v2;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * setFlattenFlag_C(PyObject* self, PyObject* args)
{
    int var;
    uint64_t ptr;
    if(!PyArg_ParseTuple(args, "Ki", &ptr, &var))
    {
        return NULL;
    }
    ((crossmulState*)(ptr))->flatten =  var;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * setFilterWeight_C(PyObject* self, PyObject* args)
{
    double var;
    uint64_t ptr;
    if(!PyArg_ParseTuple(args, "Kd", &ptr, &var))
    {
        return NULL;
    }
    ((crossmulState*)(ptr))->wgt =  var;
    Py_INCREF(Py_None);
    return Py_None;
}


PyObject * crossmul_C(PyObject *self, PyObject *args)
{
    uint64_t state;
    uint64_t slc1, slc2, ifg, amp;
    if (!PyArg_ParseTuple(args,"KKKKK", &state, &slc1, &slc2, &ifg, &amp))
    {
        return NULL;
    }
    crossmul_f((crossmulState*)(state), &slc1, &slc2, &ifg, &amp);
    Py_INCREF(Py_None);
    return Py_None;
}
