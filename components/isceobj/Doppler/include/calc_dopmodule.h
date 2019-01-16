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





#ifndef calc_dopmodule_h
#define calc_dopmodule_h

#include <Python.h>
#include <stdint.h>
#include "calc_dopmoduleFortTrans.h"

extern "C"
{
    void calc_dop_f(uint64_t *);
    PyObject * calc_dop_C(PyObject *, PyObject *);
    void setHeader_f(int *);
    PyObject * setHeader_C(PyObject *, PyObject *);
    void setWidth_f(int *);
    PyObject * setWidth_C(PyObject *, PyObject *);
    void setLastLine_f(int *);
    PyObject * setLastLine_C(PyObject *, PyObject *);
    void setFirstLine_f(int *);
    PyObject * setFirstLine_C(PyObject *, PyObject *);
    void setIoffset_f(double *);
    PyObject * setIoffset_C(PyObject *, PyObject *);
    void setQoffset_f(double *);
    PyObject * setQoffset_C(PyObject *, PyObject *);
    void getRngDoppler_f(double *, int *);
    void allocate_rngDoppler_f(int *);
    void deallocate_rngDoppler_f();
    PyObject * allocate_rngDoppler_C(PyObject *, PyObject *);
    PyObject * deallocate_rngDoppler_C(PyObject *, PyObject *);
    PyObject * getRngDoppler_C(PyObject *, PyObject *);
    void getDoppler_f(double *);
    PyObject * getDoppler_C(PyObject *, PyObject *);
}

static PyMethodDef calc_dop_methods[] =
{
    {"calc_dop_Py", calc_dop_C, METH_VARARGS, " "},
    {"setHeader_Py", setHeader_C, METH_VARARGS, " "},
    {"setWidth_Py", setWidth_C, METH_VARARGS, " "},
    {"setLastLine_Py", setLastLine_C, METH_VARARGS, " "},
    {"setFirstLine_Py", setFirstLine_C, METH_VARARGS, " "},
    {"setIoffset_Py", setIoffset_C, METH_VARARGS, " "},
    {"setQoffset_Py", setQoffset_C, METH_VARARGS, " "},
    {"allocate_rngDoppler_Py", allocate_rngDoppler_C, METH_VARARGS, " "},
    {"deallocate_rngDoppler_Py", deallocate_rngDoppler_C, METH_VARARGS, " "},
    {"getRngDoppler_Py", getRngDoppler_C, METH_VARARGS, " "},
    {"getDoppler_Py", getDoppler_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif //calc_dopmodule_h
