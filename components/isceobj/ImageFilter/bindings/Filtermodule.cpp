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
#include "FilterFactory.h"
#include "Filtermodule.h"
#include "DataAccessor.h"
#include <iostream>
#include <fstream>
#include <string>
#include <complex>
#include <stdint.h>
#include <cstdio>
using namespace std;

static const char * const __doc__ = "module for Filter.cpp";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "Filter",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    Filter_methods,
};

// initialization function for the module
// *must* be called PyInit_Filter
PyMODINIT_FUNC
PyInit_Filter()
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

PyObject * createFilter_C(PyObject* self, PyObject* args)
{
    string filter;
    char * filterCh;
    int selector;
    if(!PyArg_ParseTuple(args, "si",&filterCh,&selector))
    {
        return NULL;
    }
    filter = filterCh;
    FilterFactory * FF = new FilterFactory();
    uint64_t ptFilter = 0;

    ptFilter = (uint64_t ) FF->createFilter(filter,selector);
    delete FF;
    return Py_BuildValue("K",ptFilter);

}
PyObject * extract_C(PyObject* self, PyObject* args)
{
    uint64_t ptFilter = 0;
    if(!PyArg_ParseTuple(args, "K", &ptFilter))
    {
        return NULL;
    }
    ((Filter *) ptFilter)->extract();
    return Py_BuildValue("i", 0);
}
PyObject * finalize_C(PyObject* self, PyObject* args)
{
    uint64_t ptFilter = 0;
    if(!PyArg_ParseTuple(args, "K", &ptFilter))
    {
        return NULL;
    }
    ((Filter *) ptFilter)->finalize();
    delete (Filter *)ptFilter;
    return Py_BuildValue("i", 0);
}
PyObject * init_C(PyObject* self, PyObject* args)
{
    uint64_t ptFilter = 0;
    uint64_t ptAccessorIn = 0;
    uint64_t ptAccessorOut = 0;
    if(!PyArg_ParseTuple(args, "KKK", &ptFilter,&ptAccessorIn,&ptAccessorOut))
    {
        return NULL;
    }
    ((Filter *) ptFilter)->init((DataAccessor *)ptAccessorIn,
        (DataAccessor *) ptAccessorOut);
    return Py_BuildValue("i", 0);
}
PyObject * selectBand_C(PyObject* self, PyObject* args)
{
    uint64_t ptFilter = 0;
    int band = 0;
    if(!PyArg_ParseTuple(args, "Ki", &ptFilter,&band))
    {
        return NULL;
    }
    ((Filter *) ptFilter)->selectBand(band);
    return Py_BuildValue("i", 0);
}
PyObject * setStartLine_C(PyObject* self, PyObject* args)
{
    uint64_t ptFilter = 0;
    int line = 0;
    if(!PyArg_ParseTuple(args, "Ki", &ptFilter,&line))
    {
        return NULL;
    }
    ((Filter *) ptFilter)->setStartLine(line);
    return Py_BuildValue("i", 0);
}
PyObject * setEndLine_C(PyObject* self, PyObject* args)
{
    uint64_t ptFilter = 0;
    int line = 0;
    if(!PyArg_ParseTuple(args, "Ki", &ptFilter,&line))
    {
        return NULL;
    }
    ((Filter *) ptFilter)->setEndLine(line);
    return Py_BuildValue("i", 0);
}
