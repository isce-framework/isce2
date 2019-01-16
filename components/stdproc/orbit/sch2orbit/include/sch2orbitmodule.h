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




#ifndef sch2orbitmodule_h
#define sch2orbitmodule_h

#include <Python.h>
#include <stdint.h>
#include "sch2orbitmoduleFortTrans.h"

extern "C"
{
    void setStdWriter_f(uint64_t *);
    PyObject * setStdWriter_C(PyObject *, PyObject *);
    void sch2orbit_f();
    PyObject * sch2orbit_C(PyObject *, PyObject *);
    void setOrbitPosition_f(double *, int *);
    void allocateArrays_f(int *);
    PyObject * allocateArrays_C(PyObject *, PyObject *);
    void deallocateArrays_f();
    PyObject * deallocateArrays_C(PyObject*, PyObject*);
    PyObject * setOrbitPosition_C(PyObject *, PyObject *);
    void setOrbitVelocity_f(double *, int *);
    PyObject * setOrbitVelocity_C(PyObject *, PyObject *);
    void setPlanetGM_f(double *);
    PyObject * setPlanetGM_C(PyObject *, PyObject *);
    void setEllipsoidMajorSemiAxis_f(double *);
    PyObject * setEllipsoidMajorSemiAxis_C(PyObject *, PyObject *);
    void setEllipsoidEccentricitySquared_f(double *);
    PyObject * setEllipsoidEccentricitySquared_C(PyObject *, PyObject *);
    void setPegLatitude_f(double *);
    PyObject * setPegLatitude_C(PyObject *, PyObject *);
    void setPegLongitude_f(double *);
    PyObject * setPegLongitude_C(PyObject *, PyObject *);
    void setPegHeading_f(double *);
    PyObject * setPegHeading_C(PyObject *, PyObject *);
    void setRadiusOfCurvature_f(double *);
    PyObject * setRadiusOfCurvature_C(PyObject *, PyObject *);
    void getXYZPosition_f(double *, int *);
    PyObject * getXYZPosition_C(PyObject *, PyObject *);
    void getXYZVelocity_f(double *, int *);
    PyObject * getXYZVelocity_C(PyObject *, PyObject *);
    void getXYZGravitationalAcceleration_f(double *, int *);
    PyObject * getXYZGravitationalAcceleration_C(PyObject *, PyObject *);

}

static PyMethodDef sch2orbit_methods[] =
{
    {"setStdWriter_Py", setStdWriter_C, METH_VARARGS, " "},
    {"sch2orbit_Py", sch2orbit_C, METH_VARARGS, " "},
    {"allocateArrays_Py", allocateArrays_C, METH_VARARGS, " "},
    {"deallocateArrays_Py", deallocateArrays_C, METH_VARARGS, " "},
    {"setOrbitPosition_Py", setOrbitPosition_C, METH_VARARGS, " "},
    {"setOrbitVelocity_Py", setOrbitVelocity_C, METH_VARARGS, " "},
    {"setPlanetGM_Py", setPlanetGM_C, METH_VARARGS, " "},
    {"setEllipsoidMajorSemiAxis_Py", setEllipsoidMajorSemiAxis_C, METH_VARARGS,
        " "},
    {"setEllipsoidEccentricitySquared_Py", setEllipsoidEccentricitySquared_C,
        METH_VARARGS, " "},
    {"setPegLatitude_Py", setPegLatitude_C, METH_VARARGS, " "},
    {"setPegLongitude_Py", setPegLongitude_C, METH_VARARGS, " "},
    {"setPegHeading_Py", setPegHeading_C, METH_VARARGS, " "},
    {"setRadiusOfCurvature_Py", setRadiusOfCurvature_C, METH_VARARGS, " "},
    {"getXYZPosition_Py", getXYZPosition_C, METH_VARARGS, " "},
    {"getXYZVelocity_Py", getXYZVelocity_C, METH_VARARGS, " "},
    {"getXYZGravitationalAcceleration_Py", getXYZGravitationalAcceleration_C,
        METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif //sch2orbitmodule_h
