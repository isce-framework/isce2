//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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





#ifndef fdmocompmodule_h
#define fdmocompmodule_h

#include <Python.h>
#include <stdint.h>
#include "fdmocompmoduleFortTrans.h"

extern "C"
{
    void fdmocomp_f();
    PyObject * fdmocomp_C(PyObject *, PyObject *);
    void setStartingRange_f(double *);
    PyObject * setStartingRange_C(PyObject *, PyObject *);
    void setPRF_f(double *);
    PyObject * setPRF_C(PyObject *, PyObject *);
    void setRadarWavelength_f(double *);
    PyObject * setRadarWavelength_C(PyObject *, PyObject *);
    void setWidth_f(int *);
    PyObject * setWidth_C(PyObject *, PyObject *);
    void setHeigth_f(int *);
    PyObject * setHeigth_C(PyObject *, PyObject *);
    void setPlatformHeigth_f(int *);
    PyObject * setPlatformHeigth_C(PyObject *, PyObject *);
    void setRangeSamplingRate_f(double *);
    PyObject * setRangeSamplingRate_C(PyObject *, PyObject *);
    void setRadiusOfCurvature_f(double *);
    PyObject * setRadiusOfCurvature_C(PyObject *, PyObject *);
    void setDopplerCoefficients_f(double *, int *);
    void allocate_fdArray_f(int *);
    void deallocate_fdArray_f();
    PyObject * allocate_fdArray_C(PyObject *, PyObject *);
    PyObject * deallocate_fdArray_C(PyObject *, PyObject *);
    PyObject * setDopplerCoefficients_C(PyObject *, PyObject *);
    void setSchVelocity_f(double *, int *, int *);
    void allocate_vsch_f(int *,int *);
    void deallocate_vsch_f();
    PyObject * allocate_vsch_C(PyObject *, PyObject *);
    PyObject * deallocate_vsch_C(PyObject *, PyObject *);
    PyObject * setSchVelocity_C(PyObject *, PyObject *);
    void getCorrectedDoppler_f(double *);
    PyObject * getCorrectedDoppler_C(PyObject *, PyObject *);
    void setLookSide_f(int *);
    PyObject * setLookSide_C(PyObject *, PyObject *);
}

static PyMethodDef fdmocomp_methods[] =
{
    {"fdmocomp_Py", fdmocomp_C, METH_VARARGS, " "},
    {"setStartingRange_Py", setStartingRange_C, METH_VARARGS, " "},
    {"setPRF_Py", setPRF_C, METH_VARARGS, " "},
    {"setRadarWavelength_Py", setRadarWavelength_C, METH_VARARGS, " "},
    {"setWidth_Py", setWidth_C, METH_VARARGS, " "},
    {"setHeigth_Py", setHeigth_C, METH_VARARGS, " "},
    {"setPlatformHeigth_Py", setPlatformHeigth_C, METH_VARARGS, " "},
    {"setRangeSamplingRate_Py", setRangeSamplingRate_C, METH_VARARGS, " "},
    {"setRadiusOfCurvature_Py", setRadiusOfCurvature_C, METH_VARARGS, " "},
    {"allocate_fdArray_Py", allocate_fdArray_C, METH_VARARGS, " "},
    {"deallocate_fdArray_Py", deallocate_fdArray_C, METH_VARARGS, " "},
    {"setDopplerCoefficients_Py", setDopplerCoefficients_C, METH_VARARGS, " "},
    {"allocate_vsch_Py", allocate_vsch_C, METH_VARARGS, " "},
    {"deallocate_vsch_Py", deallocate_vsch_C, METH_VARARGS, " "},
    {"setSchVelocity_Py", setSchVelocity_C, METH_VARARGS, " "},
    {"getCorrectedDoppler_Py", getCorrectedDoppler_C, METH_VARARGS, " "},
    {"setLookSide_Py", setLookSide_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif
// end of file

