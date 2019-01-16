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




#ifndef resamp_slcmodule_h
#define resamp_slcmodule_h

#include <Python.h>
#include <stdint.h>
#include "resamp_slcmoduleFortTrans.h"
#include "poly2d.h"

extern "C"
{
    void resamp_slc_f(uint64_t *,uint64_t *, uint64_t*, uint64_t*);
    PyObject * resamp_slc_C(PyObject *, PyObject *);
    void setInputWidth_f(int *);
    PyObject * setInputWidth_C(PyObject *, PyObject *);
    void setOutputWidth_f(int *);
    PyObject * setOutputWidth_C(PyObject *, PyObject *);
    void setInputLines_f(int *);
    PyObject *setInputLines_C(PyObject *, PyObject *);
    void setOutputLines_f(int *);
    PyObject * setOutputLines_C(PyObject *, PyObject *);
    void setRadarWavelength_f(double*);
    PyObject * setRadarWavelength_C(PyObject *, PyObject *);
    void setReferenceWavelength_f(double*);
    PyObject * setReferenceWavelength_C(PyObject*, PyObject*);
    void setSlantRangePixelSpacing_f(double *);
    PyObject * setSlantRangePixelSpacing_C(PyObject *, PyObject *);
    void setReferenceSlantRangePixelSpacing_f(double*);
    PyObject * setReferenceSlantRangePixelSpacing_C(PyObject*, PyObject*);
    void setStartingRange_f(double*);
    PyObject * setStartingRange_C(PyObject*, PyObject*);
    void setReferenceStartingRange_f(double*);
    PyObject * setReferenceStartingRange_C(PyObject*, PyObject*);
    void setAzimuthCarrier_f(cPoly2d *);
    PyObject *setAzimuthCarrier_C(PyObject*, PyObject*);
    void setRangeCarrier_f(cPoly2d *);
    PyObject *setRangeCarrier_C(PyObject*, PyObject*);
    void setAzimuthOffsetsPoly_f(cPoly2d*);
    PyObject *setAzimuthOffsetsPoly_C(PyObject*, PyObject*);
    void setRangeOffsetsPoly_f(cPoly2d*);
    PyObject *setRangeOffsetsPoly_C(PyObject*, PyObject*);
    void setIsComplex_f(int*);
    PyObject *setIsComplex_C(PyObject*, PyObject*);
    void setMethod_f(int*);
    PyObject *setMethod_C(PyObject*, PyObject*);
    void setFlatten_f(int*);
    PyObject *setFlatten_C(PyObject*, PyObject*);
    void setDopplerPoly_f(cPoly2d*);
    PyObject *setDopplerPoly_C(PyObject*, PyObject*);
}

static PyMethodDef resamp_slc_methods[] =
{
    {"resamp_slc_Py", resamp_slc_C, METH_VARARGS, " "},
    {"setInputWidth_Py", setInputWidth_C, METH_VARARGS, " "},
    {"setOutputWidth_Py", setOutputWidth_C, METH_VARARGS, " "},
    {"setInputLines_Py", setInputLines_C, METH_VARARGS, " "},
    {"setOutputLines_Py", setOutputLines_C, METH_VARARGS, " "},
    {"setRadarWavelength_Py", setRadarWavelength_C, METH_VARARGS, " "},
    {"setReferenceWavelength_Py", setReferenceWavelength_C, METH_VARARGS, " "},
    {"setSlantRangePixelSpacing_Py", setSlantRangePixelSpacing_C, METH_VARARGS," "},
    {"setReferenceSlantRangePixelSpacing_Py", setReferenceSlantRangePixelSpacing_C, METH_VARARGS, " "},
    {"setAzimuthCarrier_Py", setAzimuthCarrier_C, METH_VARARGS, " "},
    {"setRangeCarrier_Py", setRangeCarrier_C, METH_VARARGS, " "},
    {"setAzimuthOffsetsPoly_Py", setAzimuthOffsetsPoly_C, METH_VARARGS, " "},
    {"setRangeOffsetsPoly_Py", setRangeOffsetsPoly_C, METH_VARARGS, " "},
    {"setDopplerPoly_Py", setDopplerPoly_C, METH_VARARGS, " "},
    {"setIsComplex_Py", setIsComplex_C, METH_VARARGS, " "},
    {"setMethod_Py", setMethod_C, METH_VARARGS, " "},
    {"setFlatten_Py", setFlatten_C, METH_VARARGS, " "},
    {"setStartingRange_Py", setStartingRange_C, METH_VARARGS, " "},
    {"setReferenceStartingRange_Py", setReferenceStartingRange_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif

// end of file
