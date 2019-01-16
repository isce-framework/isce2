//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2009 California Institute of Technology. ALL RIGHTS RESERVED.
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




#ifndef LineAccessormodule_h
#define LineAccessormodule_h

#include <Python.h>

extern "C"
{
    PyObject * getLineAccessorObject_C(PyObject *, PyObject *);
    PyObject * getMachineEndianness_C(PyObject *, PyObject *);
    PyObject * finalizeLineAccessor_C(PyObject *, PyObject *);
    PyObject * initLineAccessor_C(PyObject *, PyObject *);
    PyObject * changeBandScheme_C(PyObject *, PyObject *);
    PyObject * convertFileEndianness_C(PyObject *, PyObject *);
    PyObject * getFileLength_C(PyObject *, PyObject *);
    PyObject * getFileWidth_C(PyObject *, PyObject *);
    PyObject * createFile_C(PyObject *, PyObject *);
    PyObject * rewindImage_C(PyObject *, PyObject *);
    PyObject * getTypeSize_C(PyObject *, PyObject *);
    PyObject * printObjectInfo_C(PyObject *, PyObject *);
    PyObject * printAvailableDataTypesAndSizes_C(PyObject *, PyObject *);
}

BandSchemeType convertIntToBandSchemeType(int band);

static PyMethodDef LineAccessor_methods[] =
{
    {"getLineAccessorObject", getLineAccessorObject_C, METH_VARARGS, " "},
    {"getMachineEndianness", getMachineEndianness_C, METH_VARARGS, " "},
    {"finalizeLineAccessor", finalizeLineAccessor_C, METH_VARARGS, " "},
    {"initLineAccessor", initLineAccessor_C, METH_VARARGS, " "},
    {"changeBandScheme", changeBandScheme_C, METH_VARARGS, " "},
    {"convertFileEndianness", convertFileEndianness_C, METH_VARARGS, " "},
    {"getFileLength", getFileLength_C, METH_VARARGS, " "},
    {"getFileWidth", getFileWidth_C, METH_VARARGS, " "},
    {"createFile", createFile_C, METH_VARARGS, " "},
    {"rewindImage", rewindImage_C, METH_VARARGS, " "},
    {"getTypeSize", getTypeSize_C, METH_VARARGS, " "},
    {"printObjectInfo", printObjectInfo_C, METH_VARARGS, " "},
    {"printAvailableDataTypesAndSizes", printAvailableDataTypesAndSizes_C,
        METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif
// end of file

