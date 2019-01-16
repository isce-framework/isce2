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




#ifndef correct_geoid_i2_srtmmodule_h
#define correct_geoid_i2_srtmmodule_h

#include <Python.h>
#include <stdint.h>
#include "correct_geoid_i2_srtmmoduleFortTrans.h"

extern "C"
{
    void correct_geoid_i2_srtm_f(uint64_t *,uint64_t *);
    PyObject * correct_geoid_i2_srtm_C(PyObject *, PyObject *);
    void setWidth_f(int *);
    PyObject * setWidth_C(PyObject *, PyObject *);
    void setStartLatitude_f(double *);
    PyObject * setStartLatitude_C(PyObject *, PyObject *);
    void setStartLongitude_f(double *);
    PyObject * setStartLongitude_C(PyObject *, PyObject *);
    void setDeltaLatitude_f(double *);
    PyObject * setDeltaLatitude_C(PyObject *, PyObject *);
    void setDeltaLongitude_f(double *);
    PyObject * setDeltaLongitude_C(PyObject *, PyObject *);
    void setNumberLines_f(int *);
    PyObject * setNumberLines_C(PyObject *, PyObject *);
    void setConversionType_f(int *);
    PyObject * setConversionType_C(PyObject *, PyObject *);
    void setGeoidFilename_f(char *, int*);
    PyObject * setGeoidFilename_C(PyObject *, PyObject *);
    void setNullIsWater_f(int*);
    PyObject * setNullIsWater_C(PyObject*, PyObject*);
    void setStdWriter_f(uint64_t *);
    PyObject * setStdWriter_C(PyObject *, PyObject *);
}

static PyMethodDef correct_geoid_i2_srtm_methods[] =
{
    {"correct_geoid_i2_srtm_Py", correct_geoid_i2_srtm_C, METH_VARARGS, " "},
    {"setWidth_Py", setWidth_C, METH_VARARGS, " "},
    {"setStartLatitude_Py", setStartLatitude_C, METH_VARARGS, " "},
    {"setStartLongitude_Py", setStartLongitude_C, METH_VARARGS, " "},
    {"setDeltaLatitude_Py", setDeltaLatitude_C, METH_VARARGS, " "},
    {"setDeltaLongitude_Py", setDeltaLongitude_C, METH_VARARGS, " "},
    {"setNumberLines_Py", setNumberLines_C, METH_VARARGS, " "},
    {"setConversionType_Py", setConversionType_C, METH_VARARGS, " "},
    {"setGeoidFilename_Py", setGeoidFilename_C, METH_VARARGS, " "},
    {"setNullIsWater_Py", setNullIsWater_C, METH_VARARGS, " "},
    {"setStdWriter_Py", setStdWriter_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif
// end of file
