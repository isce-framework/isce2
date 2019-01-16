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




#ifndef orbit2schmodule_h
#define orbit2schmodule_h

#include <Python.h>
#include <stdint.h>
#include "orbit2schmoduleFortTrans.h"

extern "C"
{
    void setStdWriter_f(uint64_t *);
    PyObject * setStdWriter_C(PyObject *, PyObject *);
    void orbit2sch_f();
    PyObject * orbit2sch_C(PyObject *, PyObject *);
    void setOrbitPosition_f(double *, int *, int *);
    void allocate_xyz_f(int *,int *);
    void deallocate_xyz_f();
    PyObject * allocate_xyz_C(PyObject *, PyObject *);
    PyObject * deallocate_xyz_C(PyObject *, PyObject *);
    PyObject * setOrbitPosition_C(PyObject *, PyObject *);
    void setOrbitVelocity_f(double *, int *, int *);
    void allocate_vxyz_f(int *,int *);
    void deallocate_vxyz_f();
    PyObject * allocate_vxyz_C(PyObject *, PyObject *);
    PyObject * deallocate_vxyz_C(PyObject *, PyObject *);
    PyObject * setOrbitVelocity_C(PyObject *, PyObject *);
    void setPlanetGM_f(double *);
    PyObject * setPlanetGM_C(PyObject *, PyObject *);
    void setEllipsoidMajorSemiAxis_f(double *);
    PyObject * setEllipsoidMajorSemiAxis_C(PyObject *, PyObject *);
    void setEllipsoidEccentricitySquared_f(double *);
    PyObject * setEllipsoidEccentricitySquared_C(PyObject *, PyObject *);
    void setComputePegInfoFlag_f(int *);
    PyObject * setComputePegInfoFlag_C(PyObject *, PyObject *);
    void setPegLatitude_f(double *);
    PyObject * setPegLatitude_C(PyObject *, PyObject *);
    void setPegLongitude_f(double *);
    PyObject * setPegLongitude_C(PyObject *, PyObject *);
    void setPegHeading_f(double *);
    PyObject * setPegHeading_C(PyObject *, PyObject *);
    void setAverageHeight_f(double *);
    PyObject * setAverageHeight_C(PyObject *, PyObject *);
    void getSchPosition_f(double *, int *, int *);
    void allocate_sch_f(int *,int *);
    void deallocate_sch_f();
    PyObject * allocate_sch_C(PyObject *, PyObject *);
    PyObject * deallocate_sch_C(PyObject *, PyObject *);
    PyObject * getSchPosition_C(PyObject *, PyObject *);
    void getSchVelocity_f(double *, int *, int *);
    void allocate_vsch_f(int *,int *);
    void deallocate_vsch_f();
    PyObject * allocate_vsch_C(PyObject *, PyObject *);
    PyObject * deallocate_vsch_C(PyObject *, PyObject *);
    PyObject * getSchVelocity_C(PyObject *, PyObject *);
    void getSchGravitationalAcceleration_f(double *, int *, int *);
    void allocate_asch_f(int *,int *);
    void deallocate_asch_f();
    PyObject * allocate_asch_C(PyObject *, PyObject *);
    PyObject * deallocate_asch_C(PyObject *, PyObject *);
    PyObject * getSchGravitationalAcceleration_C(PyObject *, PyObject *);

}

static PyMethodDef orbit2sch_methods[] =
{
    {"setStdWriter_Py", setStdWriter_C, METH_VARARGS, " "},
    {"orbit2sch_Py", orbit2sch_C, METH_VARARGS, " "},
    {"allocate_xyz_Py", allocate_xyz_C, METH_VARARGS, " "},
    {"deallocate_xyz_Py", deallocate_xyz_C, METH_VARARGS, " "},
    {"setOrbitPosition_Py", setOrbitPosition_C, METH_VARARGS, " "},
    {"allocate_vxyz_Py", allocate_vxyz_C, METH_VARARGS, " "},
    {"deallocate_vxyz_Py", deallocate_vxyz_C, METH_VARARGS, " "},
    {"setOrbitVelocity_Py", setOrbitVelocity_C, METH_VARARGS, " "},
    {"setPlanetGM_Py", setPlanetGM_C, METH_VARARGS, " "},
    {"setEllipsoidMajorSemiAxis_Py", setEllipsoidMajorSemiAxis_C, METH_VARARGS,
        " "},
    {"setEllipsoidEccentricitySquared_Py", setEllipsoidEccentricitySquared_C,
        METH_VARARGS, " "},
    {"setComputePegInfoFlag_Py", setComputePegInfoFlag_C, METH_VARARGS, " "},
    {"setPegLatitude_Py", setPegLatitude_C, METH_VARARGS, " "},
    {"setPegLongitude_Py", setPegLongitude_C, METH_VARARGS, " "},
    {"setPegHeading_Py", setPegHeading_C, METH_VARARGS, " "},
    {"setAverageHeight_Py", setAverageHeight_C, METH_VARARGS, " "},
    {"allocate_sch_Py", allocate_sch_C, METH_VARARGS, " "},
    {"deallocate_sch_Py", deallocate_sch_C, METH_VARARGS, " "},
    {"getSchPosition_Py", getSchPosition_C, METH_VARARGS, " "},
    {"allocate_vsch_Py", allocate_vsch_C, METH_VARARGS, " "},
    {"deallocate_vsch_Py", deallocate_vsch_C, METH_VARARGS, " "},
    {"getSchVelocity_Py", getSchVelocity_C, METH_VARARGS, " "},
    {"allocate_asch_Py", allocate_asch_C, METH_VARARGS, " "},
    {"deallocate_asch_Py", deallocate_asch_C, METH_VARARGS, " "},
    {"getSchGravitationalAcceleration_Py", getSchGravitationalAcceleration_C,
        METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif //orbit2schmodule_h
