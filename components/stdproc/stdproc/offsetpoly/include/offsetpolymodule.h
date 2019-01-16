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




#ifndef offsetpolymodule_h
#define offsetpolymodule_h

#include <Python.h>
#include <stdint.h>
#include "offsetpolymoduleFortTrans.h"

extern "C"
{
    void offsetpoly_f();
    PyObject * offsetpoly_C(PyObject *, PyObject *);

    void allocateFieldArrays_f(int *);
    PyObject *allocateFieldArrays_C(PyObject *, PyObject *);

    void deallocateFieldArrays_f();
    PyObject *deallocateFieldArrays_C(PyObject *, PyObject *);

    void allocatePolyArray_f(int *);
    PyObject *allocatePolyArray_C(PyObject *, PyObject *);

    void deallocatePolyArray_f();
    PyObject *deallocatePolyArray_C(PyObject *, PyObject *);

    PyObject * setLocationAcross_C(PyObject *, PyObject *);
    void setLocationAcross_f(double *, int *);

    void setOffset_f(double *, int *);
    PyObject * setOffset_C(PyObject *, PyObject*);


    void setLocationDown_f(double *, int *);
    PyObject * setLocationDown_C(PyObject *, PyObject *);


    void setSNR_f(double *, int *);
    PyObject * setSNR_C(PyObject *, PyObject *);

    PyObject* getOffsetPoly_C(PyObject*, PyObject *);
    void getOffsetPoly_f(double *, int *);
}

static PyMethodDef offsetpoly_methods[] =
{
    {"offsetpoly_Py", offsetpoly_C, METH_VARARGS, " "},
    {"setLocationAcross_Py", setLocationAcross_C, METH_VARARGS, " "},
    {"setOffset_Py", setOffset_C, METH_VARARGS, " "},
    {"setLocationDown_Py", setLocationDown_C, METH_VARARGS, " "},
    {"setSNR_Py", setSNR_C, METH_VARARGS, " "},
    {"allocateFieldArrays_Py", allocateFieldArrays_C, METH_VARARGS, " "},
    {"deallocateFieldArrays_Py", deallocateFieldArrays_C, METH_VARARGS, " "},
    {"allocatePolyArray_Py", allocatePolyArray_C, METH_VARARGS, " "},
    {"deallocatePolyArray_Py", deallocatePolyArray_C, METH_VARARGS, " "},
    {"getOffsetPoly_Py", getOffsetPoly_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif
// end of file
