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




#ifndef mocompTSXmodule_h
#define mocompTSXmodule_h

#include <Python.h>
#include <stdint.h>
#include "mocompTSXmoduleFortTrans.h"

extern "C"
{
    #include "orbit.h"
    void mocompTSX_f(uint64_t *,uint64_t *);
    PyObject * mocompTSX_C(PyObject *, PyObject *);
    void setStdWriter_f(uint64_t *);
    PyObject * setStdWriter_C(PyObject *, PyObject *);
    void setNumberRangeBins_f(int *);
    PyObject * setNumberRangeBins_C(PyObject *, PyObject *);
    void setNumberAzLines_f(int *);
    PyObject * setNumberAzLines_C(PyObject *, PyObject *);
    void setDopplerCentroidCoefficients_f(double *, int *);
    void allocate_dopplerCentroidCoefficients_f(int *);
    void deallocate_dopplerCentroidCoefficients_f();
    PyObject * allocate_dopplerCentroidCoefficients_C(PyObject *, PyObject *);
    PyObject * deallocate_dopplerCentroidCoefficients_C(PyObject *, PyObject *);
    PyObject * setDopplerCentroidCoefficients_C(PyObject *, PyObject *);
    void setTime_f(double *, int *);
    void allocate_time_f(int *);
    void deallocate_time_f();
    PyObject * allocate_time_C(PyObject *, PyObject *);
    PyObject * deallocate_time_C(PyObject *, PyObject *);
    PyObject * setTime_C(PyObject *, PyObject *);
    void setPosition_f(double *, int *, int *);
    void allocate_sch_f(int *,int *);
    void deallocate_sch_f();
    PyObject * allocate_sch_C(PyObject *, PyObject *);
    PyObject * deallocate_sch_C(PyObject *, PyObject *);
    PyObject * setPosition_C(PyObject *, PyObject *);
    void setPlanetLocalRadius_f(double *);
    PyObject * setPlanetLocalRadius_C(PyObject *, PyObject *);
    void setBodyFixedVelocity_f(double *);
    PyObject * setBodyFixedVelocity_C(PyObject *, PyObject *);
    void setSpacecraftHeight_f(double *);
    PyObject * setSpacecraftHeight_C(PyObject *, PyObject *);
    void setPRF_f(double *);
    PyObject * setPRF_C(PyObject *, PyObject *);
    void setRangeSamplingRate_f(double *);
    PyObject * setRangeSamplingRate_C(PyObject *, PyObject *);
    void setRadarWavelength_f(double *);
    PyObject * setRadarWavelength_C(PyObject *, PyObject *);
    void setRangeFisrtSample_f(double *);
    PyObject * setRangeFisrtSample_C(PyObject *, PyObject *);
    void getMocompIndex_f(double *, int *);
    PyObject * getMocompIndex_C(PyObject *, PyObject *);
    void getMocompPosition_f(double *, int *, int *);
    PyObject * getMocompPosition_C(PyObject *, PyObject *);
    void getMocompPositionSize_f(int *);
    PyObject * getMocompPositionSize_C(PyObject *, PyObject *);
    void setLookSide_f(int *);
    PyObject * setLookSide_C(PyObject *, PyObject *);
    void getStartingRange_f(double *);
    PyObject* getStartingRange_C(PyObject *, PyObject *);
    void setOrbit_f(cOrbit*);
    PyObject *setOrbit_C(PyObject *, PyObject*);
    void setMocompOrbit_f(cOrbit*);
    PyObject *setMocompOrbit_C(PyObject*, PyObject*);
    PyObject *setEllipsoid_C(PyObject *self, PyObject *args);
    void setEllipsoid_f(double *a, double *e2);
    PyObject *setPlanet_C(PyObject *self, PyObject *args);
    void setPlanet_f(double *spin, double *gm);
    PyObject *setPegPoint_C(PyObject *self, PyObject *args);
    void setPegPoint_f(double *lat, double *lon, double *hdg);
    void getSlcSensingStart_f(double*);
    PyObject *getSlcSensingStart_C(PyObject*, PyObject*);
    void setSensingStart_f(double*);
    PyObject *setSensingStart_C(PyObject*, PyObject*);
    void getMocompRange_f(double*);
    PyObject *getMocompRange_C(PyObject*, PyObject*);
}

static PyMethodDef mocompTSX_methods[] =
{
    {"mocompTSX_Py", mocompTSX_C, METH_VARARGS, " "},
    {"setStdWriter_Py", setStdWriter_C, METH_VARARGS, " "},
    {"setNumberRangeBins_Py", setNumberRangeBins_C, METH_VARARGS, " "},
    {"setNumberAzLines_Py", setNumberAzLines_C, METH_VARARGS, " "},
    {"allocate_dopplerCentroidCoefficients_Py",
        allocate_dopplerCentroidCoefficients_C, METH_VARARGS, " "},
    {"deallocate_dopplerCentroidCoefficients_Py",
        deallocate_dopplerCentroidCoefficients_C, METH_VARARGS, " "},
    {"setDopplerCentroidCoefficients_Py", setDopplerCentroidCoefficients_C,
        METH_VARARGS, " "},
    {"allocate_time_Py", allocate_time_C, METH_VARARGS, " "},
    {"deallocate_time_Py", deallocate_time_C, METH_VARARGS, " "},
    {"setTime_Py", setTime_C, METH_VARARGS, " "},
    {"allocate_sch_Py", allocate_sch_C, METH_VARARGS, " "},
    {"deallocate_sch_Py", deallocate_sch_C, METH_VARARGS, " "},
    {"setPosition_Py", setPosition_C, METH_VARARGS, " "},
    {"setPlanetLocalRadius_Py", setPlanetLocalRadius_C, METH_VARARGS, " "},
    {"setBodyFixedVelocity_Py", setBodyFixedVelocity_C, METH_VARARGS, " "},
    {"setSpacecraftHeight_Py", setSpacecraftHeight_C, METH_VARARGS, " "},
    {"setPRF_Py", setPRF_C, METH_VARARGS, " "},
    {"setRangeSamplingRate_Py", setRangeSamplingRate_C, METH_VARARGS, " "},
    {"setRadarWavelength_Py", setRadarWavelength_C, METH_VARARGS, " "},
    {"setRangeFisrtSample_Py", setRangeFisrtSample_C, METH_VARARGS, " "},
    {"getMocompIndex_Py", getMocompIndex_C, METH_VARARGS, " "},
    {"getMocompPosition_Py", getMocompPosition_C, METH_VARARGS, " "},
    {"getMocompPositionSize_Py", getMocompPositionSize_C, METH_VARARGS, " "},
    {"setLookSide_Py", setLookSide_C, METH_VARARGS, " "},
    {"getStartingRange_Py", getStartingRange_C, METH_VARARGS, " "},
    {"setOrbit_Py", setOrbit_C, METH_VARARGS, " "},
    {"setMocompOrbit_Py", setMocompOrbit_C, METH_VARARGS, " "},
    {"getMocompRange_Py", getMocompRange_C, METH_VARARGS, " "},
    {"setPegPoint_Py", setPegPoint_C, METH_VARARGS, " "},
    {"setEllipsoid_Py", setEllipsoid_C, METH_VARARGS, " "},
    {"setPlanet_Py", setPlanet_C, METH_VARARGS, " "},
    {"setSensingStart_Py", setSensingStart_C, METH_VARARGS, " "},
    {"getSlcSensingStart_Py", getSlcSensingStart_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif
// end of file
