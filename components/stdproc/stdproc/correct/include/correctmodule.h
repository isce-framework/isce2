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




#ifndef correctmodule_h
#define correctmodule_h

#include <Python.h>
#include <stdint.h>
#include "correctmoduleFortTrans.h"

extern "C"
{
    void correct_f(uint64_t *,uint64_t *,uint64_t *,uint64_t *,uint64_t *,uint64_t *);
    PyObject * correct_C(PyObject *, PyObject *);
    void setReferenceOrbit_f(double *, int *);
    void allocate_s_mocompArray_f(int *);
    void deallocate_s_mocompArray_f();
    PyObject * allocate_s_mocompArray_C(PyObject *, PyObject *);
    PyObject * deallocate_s_mocompArray_C(PyObject *, PyObject *);
    PyObject * setReferenceOrbit_C(PyObject *, PyObject *);
    void setMocompBaseline_f(double *, int *, int *);
    void allocate_mocbaseArray_f(int *,int *);
    void deallocate_mocbaseArray_f();
    PyObject * allocate_mocbaseArray_C(PyObject *, PyObject *);
    PyObject * deallocate_mocbaseArray_C(PyObject *, PyObject *);
    PyObject * setMocompBaseline_C(PyObject *, PyObject *);
    void setISMocomp_f(int *);
    PyObject * setISMocomp_C(PyObject *, PyObject *);
    void setEllipsoidMajorSemiAxis_f(double *);
    PyObject * setEllipsoidMajorSemiAxis_C(PyObject *, PyObject *);
    void setEllipsoidEccentricitySquared_f(double *);
    PyObject * setEllipsoidEccentricitySquared_C(PyObject *, PyObject *);
    void setLength_f(int *);
    PyObject * setLength_C(PyObject *, PyObject *);
    void setWidth_f(int *);
    PyObject * setWidth_C(PyObject *, PyObject *);
    void setRangePixelSpacing_f(double *);
    PyObject * setRangePixelSpacing_C(PyObject *, PyObject *);
    void setRangeFirstSample_f(double *);
    PyObject * setRangeFirstSample_C(PyObject *, PyObject *);
    void setSpacecraftHeight_f(double *);
    PyObject * setSpacecraftHeight_C(PyObject *, PyObject *);
    void setPlanetLocalRadius_f(double *);
    PyObject * setPlanetLocalRadius_C(PyObject *, PyObject *);
    void setBodyFixedVelocity_f(float *);
    PyObject * setBodyFixedVelocity_C(PyObject *, PyObject *);
    void setNumberRangeLooks_f(int *);
    PyObject * setNumberRangeLooks_C(PyObject *, PyObject *);
    void setNumberAzimuthLooks_f(int *);
    PyObject * setNumberAzimuthLooks_C(PyObject *, PyObject *);
    void setPegLatitude_f(double *);
    PyObject * setPegLatitude_C(PyObject *, PyObject *);
    void setPegLongitude_f(double *);
    PyObject * setPegLongitude_C(PyObject *, PyObject *);
    void setPegHeading_f(double *);
    PyObject * setPegHeading_C(PyObject *, PyObject *);
    void setPRF_f(double *);
    PyObject * setPRF_C(PyObject *, PyObject *);
    void setRadarWavelength_f(double *);
    PyObject * setRadarWavelength_C(PyObject *, PyObject *);
    void setMidpoint_f(double *, int *, int *);
    void allocate_midpoint_f(int *,int *);
    void deallocate_midpoint_f();
    PyObject * allocate_midpoint_C(PyObject *, PyObject *);
    PyObject * deallocate_midpoint_C(PyObject *, PyObject *);
    PyObject * setMidpoint_C(PyObject *, PyObject *);
    void setSch1_f(double *, int *, int *);
    void allocate_s1sch_f(int *,int *);
    void deallocate_s1sch_f();
    PyObject * allocate_s1sch_C(PyObject *, PyObject *);
    PyObject * deallocate_s1sch_C(PyObject *, PyObject *);
    PyObject * setSch1_C(PyObject *, PyObject *);
    void setSch2_f(double *, int *, int *);
    void allocate_s2sch_f(int *,int *);
    void deallocate_s2sch_f();
    PyObject * allocate_s2sch_C(PyObject *, PyObject *);
    PyObject * deallocate_s2sch_C(PyObject *, PyObject *);
    PyObject * setSch2_C(PyObject *, PyObject *);
    void setSc_f(double *, int *, int *);
    void allocate_smsch_f(int *,int *);
    void deallocate_smsch_f();
    PyObject * allocate_smsch_C(PyObject *, PyObject *);
    PyObject * deallocate_smsch_C(PyObject *, PyObject *);
    PyObject * setDopCoeff_C(PyObject * ,PyObject *);
    void setDopCoeff_f(uint64_t *);
    PyObject * setSc_C(PyObject *, PyObject *);
    void setLookSide_f(int *);
    PyObject * setLookSide_C(PyObject*, PyObject*);
}


static PyMethodDef correct_methods[] =
{
    {"correct_Py", correct_C, METH_VARARGS, " "},
    {"allocate_s_mocompArray_Py", allocate_s_mocompArray_C, METH_VARARGS, " "},
    {"deallocate_s_mocompArray_Py", deallocate_s_mocompArray_C, METH_VARARGS,
        " "},
    {"setReferenceOrbit_Py", setReferenceOrbit_C, METH_VARARGS, " "},
    {"allocate_mocbaseArray_Py", allocate_mocbaseArray_C, METH_VARARGS, " "},
    {"deallocate_mocbaseArray_Py", deallocate_mocbaseArray_C, METH_VARARGS,
        " "},
    {"setMocompBaseline_Py", setMocompBaseline_C, METH_VARARGS, " "},
    {"setISMocomp_Py", setISMocomp_C, METH_VARARGS, " "},
    {"setEllipsoidMajorSemiAxis_Py", setEllipsoidMajorSemiAxis_C, METH_VARARGS,
        " "},
    {"setEllipsoidEccentricitySquared_Py", setEllipsoidEccentricitySquared_C,
        METH_VARARGS, " "},
    {"setLength_Py", setLength_C, METH_VARARGS, " "},
    {"setWidth_Py", setWidth_C, METH_VARARGS, " "},
    {"setRangePixelSpacing_Py", setRangePixelSpacing_C, METH_VARARGS, " "},
    {"setRangeFirstSample_Py", setRangeFirstSample_C, METH_VARARGS, " "},
    {"setSpacecraftHeight_Py", setSpacecraftHeight_C, METH_VARARGS, " "},
    {"setPlanetLocalRadius_Py", setPlanetLocalRadius_C, METH_VARARGS, " "},
    {"setBodyFixedVelocity_Py", setBodyFixedVelocity_C, METH_VARARGS, " "},
    {"setNumberRangeLooks_Py", setNumberRangeLooks_C, METH_VARARGS, " "},
    {"setNumberAzimuthLooks_Py", setNumberAzimuthLooks_C, METH_VARARGS, " "},
    {"setPegLatitude_Py", setPegLatitude_C, METH_VARARGS, " "},
    {"setPegLongitude_Py", setPegLongitude_C, METH_VARARGS, " "},
    {"setPegHeading_Py", setPegHeading_C, METH_VARARGS, " "},
    {"setPRF_Py", setPRF_C, METH_VARARGS, " "},
    {"setRadarWavelength_Py", setRadarWavelength_C, METH_VARARGS, " "},
    {"allocate_midpoint_Py", allocate_midpoint_C, METH_VARARGS, " "},
    {"deallocate_midpoint_Py", deallocate_midpoint_C, METH_VARARGS, " "},
    {"setMidpoint_Py", setMidpoint_C, METH_VARARGS, " "},
    {"allocate_s1sch_Py", allocate_s1sch_C, METH_VARARGS, " "},
    {"deallocate_s1sch_Py", deallocate_s1sch_C, METH_VARARGS, " "},
    {"setSch1_Py", setSch1_C, METH_VARARGS, " "},
    {"allocate_s2sch_Py", allocate_s2sch_C, METH_VARARGS, " "},
    {"deallocate_s2sch_Py", deallocate_s2sch_C, METH_VARARGS, " "},
    {"setSch2_Py", setSch2_C, METH_VARARGS, " "},
    {"allocate_smsch_Py", allocate_smsch_C, METH_VARARGS, " "},
    {"deallocate_smsch_Py", deallocate_smsch_C, METH_VARARGS, " "},
    {"setSc_Py", setSc_C, METH_VARARGS, " "},
    {"setDopCoeff_Py", setDopCoeff_C, METH_VARARGS, " "},
    {"setLookSide_Py", setLookSide_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif

// end of file
