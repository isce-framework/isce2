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




#ifndef crossmulmodule_h
#define crossmulmodule_h

#include <Python.h>
#include <stdint.h>
#include "crossmul.h"

extern "C"
{
    PyObject * createCrossMul_C(PyObject *, PyObject *);
    PyObject * setWidth_C(PyObject *, PyObject *);
    PyObject * setLength_C(PyObject *, PyObject *);
    PyObject * setLooksAcross_C(PyObject *, PyObject *);
    PyObject * setLooksDown_C(PyObject *, PyObject *);
    PyObject * setScale_C(PyObject *, PyObject *);
    PyObject * setBlocksize_C(PyObject *, PyObject *);
    PyObject * setWavelengths_C(PyObject *, PyObject *);
    PyObject * setSpacings_C(PyObject*, PyObject*);
    PyObject * setFlattenFlag_C(PyObject*, PyObject*);
    PyObject * setFilterWeight_C(PyObject*, PyObject*);
    void crossmul_f(crossmulState*, uint64_t*, uint64_t*, uint64_t*,
        uint64_t*);
    PyObject * crossmul_C(PyObject*, PyObject*);
    PyObject * destroyCrossMul_C(PyObject *, PyObject *);

}

static PyMethodDef crossmul_methods[] =
{
     {"createCrossMul_Py", createCrossMul_C, METH_VARARGS, " "},
     {"destroyCrossMul_Py", destroyCrossMul_C, METH_VARARGS, " "},
     {"setWidth_Py", setWidth_C, METH_VARARGS, " "},
     {"setLength_Py", setLength_C, METH_VARARGS, " "},
     {"setLooksAcross_Py", setLooksAcross_C, METH_VARARGS, " "},
     {"setLooksDown_Py", setLooksDown_C, METH_VARARGS, " "},
     {"setScale_Py", setScale_C, METH_VARARGS, " "},
     {"setBlocksize_Py", setBlocksize_C, METH_VARARGS, " "},
     {"setWavelengths_Py", setWavelengths_C, METH_VARARGS, " "},
     {"setSpacings_Py", setSpacings_C, METH_VARARGS, " "},
     {"setFlattenFlag_Py", setFlattenFlag_C, METH_VARARGS, " "},
     {"setFilterWeight_Py", setFilterWeight_C, METH_VARARGS, " "},
     {"crossmul_Py", crossmul_C, METH_VARARGS, " "},
     {NULL, NULL, 0, NULL}
};
#endif

// end of file
