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
// Authors: Piyush Agram, Giangi Sacco
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





#ifndef getpegmodule_h
#define getpegmodule_h

#include <Python.h>
#include <stdint.h>
#include "getpegmoduleFortTrans.h"

extern "C"
{
    void getpeg_f();
    PyObject * getpeg_C(PyObject *, PyObject *);
    void setPosition_f(double *, int *, int *);
    void allocate_xyz_f(int *,int *);
    void deallocate_xyz_f();
    PyObject * allocate_xyz_C(PyObject *, PyObject *);
    PyObject * deallocate_xyz_C(PyObject *, PyObject *);
    PyObject * setPosition_C(PyObject *, PyObject *);
    void setVelocity_f(double *, int *, int *);
    void allocate_vxyz_f(int *,int *);
    void deallocate_vxyz_f();
    PyObject * allocate_vxyz_C(PyObject *, PyObject *);
    PyObject * deallocate_vxyz_C(PyObject *, PyObject *);
    PyObject * setVelocity_C(PyObject *, PyObject *);
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
    void getAverageHeight_f(double *);
    PyObject * getAverageHeight_C(PyObject *, PyObject *);
    void getProcVelocity_f(double *);
    PyObject * getProcVelocity_C(PyObject *, PyObject *);

}

static PyMethodDef getpeg_methods[] =
{
    {"getpeg_Py", getpeg_C, METH_VARARGS, " "},
    {"allocate_xyz_Py", allocate_xyz_C, METH_VARARGS, " "},
    {"deallocate_xyz_Py", deallocate_xyz_C, METH_VARARGS, " "},
    {"setPosition_Py", setPosition_C, METH_VARARGS, " "},
    {"allocate_vxyz_Py", allocate_vxyz_C, METH_VARARGS, " "},
    {"deallocate_vxyz_Py", deallocate_vxyz_C, METH_VARARGS, " "},
    {"setVelocity_Py", setVelocity_C, METH_VARARGS, " "},
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
    {"getAverageHeight_Py", getAverageHeight_C, METH_VARARGS, " "},
    {"getProcVelocity_Py", getProcVelocity_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif

// end of file
