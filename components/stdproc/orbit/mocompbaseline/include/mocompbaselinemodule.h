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




#ifndef mocompbaselinemodule_h
#define mocompbaselinemodule_h

#include <Python.h>
#include <stdint.h>
#include "mocompbaselinemoduleFortTrans.h"

extern "C"
{
    void setStdWriter_f(uint64_t *);
    PyObject * setStdWriter_C(PyObject *, PyObject *);
    void mocompbaseline_f();
    PyObject * mocompbaseline_C(PyObject *, PyObject *);
    void setSchPosition1_f(double *, int *, int *);
    void allocate_sch1_f(int *,int *);
    void deallocate_sch1_f();
    PyObject * allocate_sch1_C(PyObject *, PyObject *);
    PyObject * deallocate_sch1_C(PyObject *, PyObject *);
    PyObject * setSchPosition1_C(PyObject *, PyObject *);
    void setSchPosition2_f(double *, int *, int *);
    void allocate_sch2_f(int *,int *);
    void deallocate_sch2_f();
    PyObject * allocate_sch2_C(PyObject *, PyObject *);
    PyObject * deallocate_sch2_C(PyObject *, PyObject *);
    PyObject * setSchPosition2_C(PyObject *, PyObject *);
    void setMocompPosition1_f(double *, int *);
    void allocate_s1_f(int *);
    void deallocate_s1_f();
    PyObject * allocate_s1_C(PyObject *, PyObject *);
    PyObject * deallocate_s1_C(PyObject *, PyObject *);
    PyObject * setMocompPosition1_C(PyObject *, PyObject *);
    void setMocompPositionIndex1_f(int *, int *);
    void allocate_is1_f(int *);
    void deallocate_is1_f();
    PyObject * allocate_is1_C(PyObject *, PyObject *);
    PyObject * deallocate_is1_C(PyObject *, PyObject *);
    PyObject * setMocompPositionIndex1_C(PyObject *, PyObject *);
    void setMocompPosition2_f(double *, int *);
    void allocate_s2_f(int *);
    void deallocate_s2_f();
    PyObject * allocate_s2_C(PyObject *, PyObject *);
    PyObject * deallocate_s2_C(PyObject *, PyObject *);
    PyObject * setMocompPosition2_C(PyObject *, PyObject *);
    void setMocompPositionIndex2_f(int *, int *);
    void allocate_is2_f(int *);
    void deallocate_is2_f();
    PyObject * allocate_is2_C(PyObject *, PyObject *);
    PyObject * deallocate_is2_C(PyObject *, PyObject *);
    PyObject * setMocompPositionIndex2_C(PyObject *, PyObject *);
    void setEllipsoidMajorSemiAxis_f(double *);
    PyObject * setEllipsoidMajorSemiAxis_C(PyObject *, PyObject *);
    void setEllipsoidEccentricitySquared_f(double *);
    PyObject * setEllipsoidEccentricitySquared_C(PyObject *, PyObject *);
    void setPlanetLocalRadius_f(double *);
    PyObject * setPlanetLocalRadius_C(PyObject *, PyObject *);
    void setPegLatitude_f(double *);
    PyObject * setPegLatitude_C(PyObject *, PyObject *);
    void setPegLongitude_f(double *);
    PyObject * setPegLongitude_C(PyObject *, PyObject *);
    void setPegHeading_f(double *);
    PyObject * setPegHeading_C(PyObject *, PyObject *);
    void setHeight_f(double *);
    PyObject * setHeight_C(PyObject *, PyObject *);
    void getBaseline_f(double *, int *, int *);
    void allocate_baselineArray_f(int *,int *);
    void deallocate_baselineArray_f();
    PyObject * allocate_baselineArray_C(PyObject *, PyObject *);
    PyObject * deallocate_baselineArray_C(PyObject *, PyObject *);
    PyObject * getBaseline_C(PyObject *, PyObject *);
    void getMidpoint_f(double *, int *, int *);
    void allocate_midPointArray_f(int *,int *);
    void deallocate_midPointArray_f();
    PyObject * allocate_midPointArray_C(PyObject *, PyObject *);
    PyObject * deallocate_midPointArray_C(PyObject *, PyObject *);
    PyObject * getMidpoint_C(PyObject *, PyObject *);
    void getMidpoint1_f(double *, int *, int *);
    void allocate_midPointArray1_f(int *,int *);
    void deallocate_midPointArray1_f();
    PyObject * allocate_midPointArray1_C(PyObject *, PyObject *);
    PyObject * deallocate_midPointArray1_C(PyObject *, PyObject *);
    PyObject * getMidpoint1_C(PyObject *, PyObject *);
    void getMidpoint2_f(double *, int *, int *);
    void allocate_midPointArray2_f(int *,int *);
    void deallocate_midPointArray2_f();
    PyObject * allocate_midPointArray2_C(PyObject *, PyObject *);
    PyObject * deallocate_midPointArray2_C(PyObject *, PyObject *);
    PyObject * getMidpoint2_C(PyObject *, PyObject *);
    void getBaseline1_f(double *, int *, int *);
    void allocate_baselineArray1_f(int *,int *);
    void deallocate_baselineArray1_f();
    PyObject * allocate_baselineArray1_C(PyObject *, PyObject *);
    PyObject * deallocate_baselineArray1_C(PyObject *, PyObject *);
    PyObject * getBaseline1_C(PyObject *, PyObject *);
    void getBaseline2_f(double *, int *, int *);
    void allocate_baselineArray2_f(int *,int *);
    void deallocate_baselineArray2_f();
    PyObject * allocate_baselineArray2_C(PyObject *, PyObject *);
    PyObject * deallocate_baselineArray2_C(PyObject *, PyObject *);
    PyObject * getBaseline2_C(PyObject *, PyObject *);
    void getSch_f(double *, int *, int *);
    void allocate_schArray_f(int *,int *);
    void deallocate_schArray_f();
    PyObject * allocate_schArray_C(PyObject *, PyObject *);
    PyObject * deallocate_schArray_C(PyObject *, PyObject *);
    PyObject * getSch_C(PyObject *, PyObject *);
    void getSc_f(double *, int *, int *);
    void get_dim1_s1_f(int*);
    void allocate_scArray_f(int *,int *);
    void deallocate_scArray_f();
    PyObject * allocate_scArray_C(PyObject *, PyObject *);
    PyObject * deallocate_scArray_C(PyObject *, PyObject *);
    PyObject * getSc_C(PyObject *, PyObject *);
    PyObject * get_dim1_s1_C(PyObject *, PyObject *);

}

static PyMethodDef mocompbaseline_methods[] =
{
    {"setStdWriter_Py", setStdWriter_C, METH_VARARGS, " "},
    {"mocompbaseline_Py", mocompbaseline_C, METH_VARARGS, " "},
    {"allocate_sch1_Py", allocate_sch1_C, METH_VARARGS, " "},
    {"deallocate_sch1_Py", deallocate_sch1_C, METH_VARARGS, " "},
    {"setSchPosition1_Py", setSchPosition1_C, METH_VARARGS, " "},
    {"allocate_sch2_Py", allocate_sch2_C, METH_VARARGS, " "},
    {"deallocate_sch2_Py", deallocate_sch2_C, METH_VARARGS, " "},
    {"setSchPosition2_Py", setSchPosition2_C, METH_VARARGS, " "},
    {"allocate_s1_Py", allocate_s1_C, METH_VARARGS, " "},
    {"deallocate_s1_Py", deallocate_s1_C, METH_VARARGS, " "},
    {"setMocompPosition1_Py", setMocompPosition1_C, METH_VARARGS, " "},
    {"allocate_is1_Py", allocate_is1_C, METH_VARARGS, " "},
    {"deallocate_is1_Py", deallocate_is1_C, METH_VARARGS, " "},
    {"setMocompPositionIndex1_Py", setMocompPositionIndex1_C, METH_VARARGS,
        " "},
    {"allocate_s2_Py", allocate_s2_C, METH_VARARGS, " "},
    {"deallocate_s2_Py", deallocate_s2_C, METH_VARARGS, " "},
    {"setMocompPosition2_Py", setMocompPosition2_C, METH_VARARGS, " "},
    {"allocate_is2_Py", allocate_is2_C, METH_VARARGS, " "},
    {"deallocate_is2_Py", deallocate_is2_C, METH_VARARGS, " "},
    {"setMocompPositionIndex2_Py", setMocompPositionIndex2_C, METH_VARARGS,
        " "},
    {"setEllipsoidMajorSemiAxis_Py", setEllipsoidMajorSemiAxis_C, METH_VARARGS,
        " "},
    {"setEllipsoidEccentricitySquared_Py", setEllipsoidEccentricitySquared_C,
        METH_VARARGS, " "},
    {"setPlanetLocalRadius_Py", setPlanetLocalRadius_C, METH_VARARGS, " "},
    {"setPegLatitude_Py", setPegLatitude_C, METH_VARARGS, " "},
    {"setPegLongitude_Py", setPegLongitude_C, METH_VARARGS, " "},
    {"setPegHeading_Py", setPegHeading_C, METH_VARARGS, " "},
    {"setHeight_Py", setHeight_C, METH_VARARGS, " "},
    {"allocate_baselineArray_Py", allocate_baselineArray_C, METH_VARARGS, " "},
    {"deallocate_baselineArray_Py", deallocate_baselineArray_C, METH_VARARGS,
        " "},
    {"getBaseline_Py", getBaseline_C, METH_VARARGS, " "},
    {"allocate_midPointArray_Py", allocate_midPointArray_C, METH_VARARGS, " "},
    {"deallocate_midPointArray_Py", deallocate_midPointArray_C, METH_VARARGS,
        " "},
    {"getMidpoint_Py", getMidpoint_C, METH_VARARGS, " "},
    {"allocate_midPointArray1_Py", allocate_midPointArray1_C, METH_VARARGS,
        " "},
    {"deallocate_midPointArray1_Py", deallocate_midPointArray1_C, METH_VARARGS,
        " "},
    {"getMidpoint1_Py", getMidpoint1_C, METH_VARARGS, " "},
    {"allocate_midPointArray2_Py", allocate_midPointArray2_C, METH_VARARGS,
        " "},
    {"deallocate_midPointArray2_Py", deallocate_midPointArray2_C, METH_VARARGS,
        " "},
    {"getMidpoint2_Py", getMidpoint2_C, METH_VARARGS, " "},
    {"allocate_baselineArray1_Py", allocate_baselineArray1_C, METH_VARARGS,
        " "},
    {"deallocate_baselineArray1_Py", deallocate_baselineArray1_C, METH_VARARGS,
        " "},
    {"getBaseline1_Py", getBaseline1_C, METH_VARARGS, " "},
    {"allocate_baselineArray2_Py", allocate_baselineArray2_C, METH_VARARGS,
        " "},
    {"deallocate_baselineArray2_Py", deallocate_baselineArray2_C, METH_VARARGS,
        " "},
    {"getBaseline2_Py", getBaseline2_C, METH_VARARGS, " "},
    {"allocate_schArray_Py", allocate_schArray_C, METH_VARARGS, " "},
    {"deallocate_schArray_Py", deallocate_schArray_C, METH_VARARGS, " "},
    {"getSch_Py", getSch_C, METH_VARARGS, " "},
    {"allocate_scArray_Py", allocate_scArray_C, METH_VARARGS, " "},
    {"deallocate_scArray_Py", deallocate_scArray_C, METH_VARARGS, " "},
    {"getSc_Py", getSc_C, METH_VARARGS, " "},
    {"get_dim1_s1_Py", get_dim1_s1_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif

// end of file
