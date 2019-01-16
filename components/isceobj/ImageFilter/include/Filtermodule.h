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




#ifndef Filtermodule_h
#define Filtermodule_h

#include <Python.h>

extern "C"
{
    PyObject * createFilter_C(PyObject *, PyObject *);
    PyObject * selectBand_C(PyObject *, PyObject *);
    PyObject * setStartLine_C(PyObject *, PyObject *);
    PyObject * setEndLine_C(PyObject *, PyObject *);
    PyObject * finalize_C(PyObject *, PyObject *);
    PyObject * init_C(PyObject *, PyObject *);
    PyObject * extract_C(PyObject *, PyObject *);
}


static PyMethodDef Filter_methods[] =
{
    {"createFilter", createFilter_C, METH_VARARGS, " "},
    {"selectBand", selectBand_C, METH_VARARGS, " "},
    {"setStartLine", setStartLine_C, METH_VARARGS, " "},
    {"setEndLine", setEndLine_C, METH_VARARGS, " "},
    {"extract", extract_C, METH_VARARGS, " "},
    {"finalize", finalize_C, METH_VARARGS, " "},
    {"init", init_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif
// end of file

