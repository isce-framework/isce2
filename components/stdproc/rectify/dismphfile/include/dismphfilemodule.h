//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2011 California Institute of Technology. ALL RIGHTS RESERVED.
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




#ifndef dismphfilemodule_h
#define dismphfilemodule_h

#include <Python.h>
#include <stdint.h>
#include "dismphfilemoduleFortTrans.h"

extern "C"
{
    void dismphfile_f(uint64_t *, uint64_t *);
    PyObject * dismphfile_C(PyObject *, PyObject *);
    void setLength_f(int *);
    PyObject * setLength_C(PyObject *, PyObject *);
    void setFirstLine_f(int *);
    PyObject * setFirstLine_C(PyObject *, PyObject *);
    void setNumberLines_f(int *);
    PyObject * setNumberLines_C(PyObject *, PyObject *);
    void setFlipFlag_f(int *);
    PyObject * setFlipFlag_C(PyObject *, PyObject *);
    void setScale_f(float *);
    PyObject * setScale_C(PyObject *, PyObject *);
    void setExponent_f(float *);
    PyObject * setExponent_C(PyObject *, PyObject *);

}

static PyMethodDef dismphfile_methods[] =
{
    {"dismphfile_Py", dismphfile_C, METH_VARARGS, " "},
    {"setLength_Py", setLength_C, METH_VARARGS, " "},
    {"setFirstLine_Py", setFirstLine_C, METH_VARARGS, " "},
    {"setNumberLines_Py", setNumberLines_C, METH_VARARGS, " "},
    {"setFlipFlag_Py", setFlipFlag_C, METH_VARARGS, " "},
    {"setScale_Py", setScale_C, METH_VARARGS, " "},
    {"setExponent_Py", setExponent_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif //dismphfilemodule_h
