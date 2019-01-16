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




#ifndef setmocomppathmodule_h
#define setmocomppathmodule_h

#include <Python.h>
#include <stdint.h>
#include "setmocomppathmoduleFortTrans.h"

extern "C"
{
    void setmocomppath_f();
    PyObject * setmocomppath_C(PyObject *, PyObject *);
    void setFirstPosition_f(double *, int *, int *);
    void allocate_xyz1_f(int *,int *);
    void deallocate_xyz1_f();
    PyObject * allocate_xyz1_C(PyObject *, PyObject *);
    PyObject * deallocate_xyz1_C(PyObject *, PyObject *);
    PyObject * setFirstPosition_C(PyObject *, PyObject *);
    void setFirstVelocity_f(double *, int *, int *);
    void allocate_vxyz1_f(int *,int *);
    void deallocate_vxyz1_f();
    PyObject * allocate_vxyz1_C(PyObject *, PyObject *);
    PyObject * deallocate_vxyz1_C(PyObject *, PyObject *);
    PyObject * setFirstVelocity_C(PyObject *, PyObject *);
    void setSecondPosition_f(double *, int *, int *);
    void allocate_xyz2_f(int *,int *);
    void deallocate_xyz2_f();
    PyObject * allocate_xyz2_C(PyObject *, PyObject *);
    PyObject * deallocate_xyz2_C(PyObject *, PyObject *);
    PyObject * setSecondPosition_C(PyObject *, PyObject *);
    void setSecondVelocity_f(double *, int *, int *);
    void allocate_vxyz2_f(int *,int *);
    void deallocate_vxyz2_f();
    PyObject * allocate_vxyz2_C(PyObject *, PyObject *);
    PyObject * deallocate_vxyz2_C(PyObject *, PyObject *);
    PyObject * setSecondVelocity_C(PyObject *, PyObject *);
    void setStdWriter_f(uint64_t *);
    PyObject * setStdWriter_C(PyObject *, PyObject *);
    void setPlanetGM_f(double *);
    PyObject * setPlanetGM_C(PyObject *, PyObject *);
    void setEllipsoidMajorSemiAxis_f(double *);
    PyObject * setEllipsoidMajorSemiAxis_C(PyObject *, PyObject *);
    void setEllipsoidEccentricitySquared_f(double *);
    PyObject * setEllipsoidEccentricitySquared_C(PyObject *, PyObject *);
    void getPegLatitude_f(double *);
    PyObject * getPegLatitude_C(PyObject *, PyObject *);
    void getPegLongitude_f(double *);
    PyObject * getPegLongitude_C(PyObject *, PyObject *);
    void getPegHeading_f(double *);
    PyObject * getPegHeading_C(PyObject *, PyObject *);
    void getPegRadiusOfCurvature_f(double *);
    PyObject * getPegRadiusOfCurvature_C(PyObject *, PyObject *);
    void getFirstAverageHeight_f(double *);
    PyObject * getFirstAverageHeight_C(PyObject *, PyObject *);
    void getSecondAverageHeight_f(double *);
    PyObject * getSecondAverageHeight_C(PyObject *, PyObject *);
    void getFirstProcVelocity_f(double *);
    PyObject * getFirstProcVelocity_C(PyObject *, PyObject *);
    void getSecondProcVelocity_f(double *);
    PyObject * getSecondProcVelocity_C(PyObject *, PyObject *);

}

static PyMethodDef setmocomppath_methods[] =
{
    {"setmocomppath_Py", setmocomppath_C, METH_VARARGS, " "},
    {"allocate_xyz1_Py", allocate_xyz1_C, METH_VARARGS, " "},
    {"deallocate_xyz1_Py", deallocate_xyz1_C, METH_VARARGS, " "},
    {"setFirstPosition_Py", setFirstPosition_C, METH_VARARGS, " "},
    {"allocate_vxyz1_Py", allocate_vxyz1_C, METH_VARARGS, " "},
    {"deallocate_vxyz1_Py", deallocate_vxyz1_C, METH_VARARGS, " "},
    {"setFirstVelocity_Py", setFirstVelocity_C, METH_VARARGS, " "},
    {"allocate_xyz2_Py", allocate_xyz2_C, METH_VARARGS, " "},
    {"deallocate_xyz2_Py", deallocate_xyz2_C, METH_VARARGS, " "},
    {"setSecondPosition_Py", setSecondPosition_C, METH_VARARGS, " "},
    {"allocate_vxyz2_Py", allocate_vxyz2_C, METH_VARARGS, " "},
    {"deallocate_vxyz2_Py", deallocate_vxyz2_C, METH_VARARGS, " "},
    {"setSecondVelocity_Py", setSecondVelocity_C, METH_VARARGS, " "},
    {"setStdWriter_Py", setStdWriter_C, METH_VARARGS, " "},
    {"setPlanetGM_Py", setPlanetGM_C, METH_VARARGS, " "},
    {"setEllipsoidMajorSemiAxis_Py", setEllipsoidMajorSemiAxis_C, METH_VARARGS,
        " "},
    {"setEllipsoidEccentricitySquared_Py", setEllipsoidEccentricitySquared_C,
        METH_VARARGS, " "},
    {"getPegLatitude_Py", getPegLatitude_C, METH_VARARGS, " "},
    {"getPegLongitude_Py", getPegLongitude_C, METH_VARARGS, " "},
    {"getPegHeading_Py", getPegHeading_C, METH_VARARGS, " "},
    {"getPegRadiusOfCurvature_Py", getPegRadiusOfCurvature_C, METH_VARARGS,
        " "},
    {"getFirstAverageHeight_Py", getFirstAverageHeight_C, METH_VARARGS, " "},
    {"getSecondAverageHeight_Py", getSecondAverageHeight_C, METH_VARARGS, " "},
    {"getFirstProcVelocity_Py", getFirstProcVelocity_C, METH_VARARGS, " "},
    {"getSecondProcVelocity_Py", getSecondProcVelocity_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif

// end of file
