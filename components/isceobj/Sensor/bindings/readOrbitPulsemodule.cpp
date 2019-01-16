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
#include "readOrbitPulsemodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "module for readOrbitPulse";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "readOrbitPulse",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    readOrbitPulse_methods,
};


// initialization function for the module
// *must* be called PyInit_alos
PyMODINIT_FUNC
PyInit_readOrbitPulse()
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


PyObject * readOrbitPulse_C(PyObject* self, PyObject* args)
{
        uint64_t var0;
        uint64_t var1;
        uint64_t var2;
        if(!PyArg_ParseTuple(args, "KKK",&var0,&var1,&var2))
        {
                return NULL;
        }
        readOrbitPulse_f(&var0,&var1,&var2);
        return Py_BuildValue("i", 0);
}
PyObject * setNumberBitesPerLine_C(PyObject* self, PyObject* args)
{
        int var;
        if(!PyArg_ParseTuple(args, "i", &var))
        {
                return NULL;
        }
        setNumberBitesPerLine_f(&var);
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
