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




#ifndef geocodemodule_h
#define geocodemodule_h

#include <Python.h>
#include <stdint.h>
#include "geocodemoduleFortTrans.h"
#include "poly1d.h"
extern "C"
{
    void setStdWriter_f(uint64_t *);
    PyObject * setStdWriter_C(PyObject *, PyObject *);
    void geocode_f(uint64_t *, uint64_t *, uint64_t *, uint64_t *, uint64_t *,
        int*, int*, int*, int*);
    PyObject * geocode_C(PyObject *, PyObject *);
    void setEllipsoidMajorSemiAxis_f(double *);
    PyObject * setEllipsoidMajorSemiAxis_C(PyObject *, PyObject *);
    void setEllipsoidEccentricitySquared_f(double *);
    PyObject * setEllipsoidEccentricitySquared_C(PyObject *, PyObject *);
    void setMinimumLatitude_f(double *);
    PyObject * setMinimumLatitude_C(PyObject *, PyObject *);
    void setMinimumLongitude_f(double *);
    PyObject * setMinimumLongitude_C(PyObject *, PyObject *);
    void setMaximumLatitude_f(double *);
    PyObject * setMaximumLatitude_C(PyObject *, PyObject *);
    void setMaximumLongitude_f(double *);
    PyObject * setMaximumLongitude_C(PyObject *, PyObject *);
    void setPegLatitude_f(double *);
    PyObject * setPegLatitude_C(PyObject *, PyObject *);
    void setPegLongitude_f(double *);
    PyObject * setPegLongitude_C(PyObject *, PyObject *);
    void setPegHeading_f(double *);
    PyObject * setPegHeading_C(PyObject *, PyObject *);
    void setRangePixelSpacing_f(float *);
    PyObject * setRangePixelSpacing_C(PyObject *, PyObject *);
    void setRangeFirstSample_f(double *);
    PyObject * setRangeFirstSample_C(PyObject *, PyObject *);
    void setHeight_f(float *);
    PyObject * setHeight_C(PyObject *, PyObject *);
    void setPlanetLocalRadius_f(double *);
    PyObject * setPlanetLocalRadius_C(PyObject *, PyObject *);
    void setVelocity_f(float *);
    PyObject * setVelocity_C(PyObject *, PyObject *);
    void setDopplerAccessor_f(cPoly1d *);
    PyObject * setDopplerAccessor_C(PyObject *, PyObject *);
    void setPRF_f(float *);
    PyObject * setPRF_C(PyObject *, PyObject *);
    void setRadarWavelength_f(float *);
    PyObject * setRadarWavelength_C(PyObject *, PyObject *);
    void setSCoordinateFirstLine_f(double *);
    PyObject * setSCoordinateFirstLine_C(PyObject *, PyObject *);
    void setFirstLatitude_f(double *);
    PyObject * setFirstLatitude_C(PyObject *, PyObject *);
    void setFirstLongitude_f(double *);
    PyObject * setFirstLongitude_C(PyObject *, PyObject *);
    void setDeltaLatitude_f(double *);
    PyObject * setDeltaLatitude_C(PyObject *, PyObject *);
    void setDeltaLongitude_f(double *);
    PyObject * setDeltaLongitude_C(PyObject *, PyObject *);
    void setLength_f(int *);
    PyObject * setLength_C(PyObject *, PyObject *);
    void setLookSide_f(int *);
    PyObject * setLookSide_C(PyObject *, PyObject *);
    void setWidth_f(int *);
    PyObject * setWidth_C(PyObject *, PyObject *);
    void setNumberRangeLooks_f(int *);
    PyObject * setNumberRangeLooks_C(PyObject *, PyObject *);
    void setNumberAzimuthLooks_f(int *);
    PyObject * setNumberAzimuthLooks_C(PyObject *, PyObject *);
    void setNumberPointsPerDemPost_f(int *);
    PyObject * setNumberPointsPerDemPost_C(PyObject *, PyObject *);
    void setISMocomp_f(int *);
    PyObject * setISMocomp_C(PyObject *, PyObject *);
    void setDemWidth_f(int *);
    PyObject * setDemWidth_C(PyObject *, PyObject *);
    void setDemLength_f(int *);
    PyObject * setDemLength_C(PyObject *, PyObject *);
    void setReferenceOrbit_f(double *, int *);
    void allocate_s_mocomp_f(int *);
    void deallocate_s_mocomp_f();
    PyObject * allocate_s_mocomp_C(PyObject *, PyObject *);
    PyObject * deallocate_s_mocomp_C(PyObject *, PyObject *);
    PyObject * setReferenceOrbit_C(PyObject *, PyObject *);
    void getGeoWidth_f(int *);
    PyObject * getGeoWidth_C(PyObject *, PyObject *);
    void getGeoLength_f(int *);
    PyObject * getGeoLength_C(PyObject *, PyObject *);
    void getLatitudeSpacing_f(double *);
    PyObject * getLatitudeSpacing_C(PyObject *, PyObject *);
    void getLongitudeSpacing_f(double *);
    PyObject * getLongitudeSpacing_C(PyObject *, PyObject *);
    void getMinimumGeoLatitude_f(double *);
    PyObject * getMinimumGeoLatitude_C(PyObject *, PyObject *);
    void getMinimumGeoLongitude_f(double *);
    PyObject * getMinimumGeoLongitude_C(PyObject *, PyObject *);
    void getMaximumGeoLatitude_f(double *);
    PyObject * getMaximumGeoLatitude_C(PyObject *, PyObject *);
    void getMaxmumGeoLongitude_f(double *);
    PyObject * getMaxmumGeoLongitude_C(PyObject *, PyObject *);

}

static PyMethodDef geocode_methods[] =
{
    {"setStdWriter_Py", setStdWriter_C, METH_VARARGS, " "},
    {"geocode_Py", geocode_C, METH_VARARGS, " "},
    {"setEllipsoidMajorSemiAxis_Py", setEllipsoidMajorSemiAxis_C, METH_VARARGS,
        " "},
    {"setEllipsoidEccentricitySquared_Py", setEllipsoidEccentricitySquared_C,
        METH_VARARGS, " "},
    {"setMinimumLatitude_Py", setMinimumLatitude_C, METH_VARARGS, " "},
    {"setMinimumLongitude_Py", setMinimumLongitude_C, METH_VARARGS, " "},
    {"setMaximumLatitude_Py", setMaximumLatitude_C, METH_VARARGS, " "},
    {"setMaximumLongitude_Py", setMaximumLongitude_C, METH_VARARGS, " "},
    {"setPegLatitude_Py", setPegLatitude_C, METH_VARARGS, " "},
    {"setPegLongitude_Py", setPegLongitude_C, METH_VARARGS, " "},
    {"setPegHeading_Py", setPegHeading_C, METH_VARARGS, " "},
    {"setRangePixelSpacing_Py", setRangePixelSpacing_C, METH_VARARGS, " "},
    {"setRangeFirstSample_Py", setRangeFirstSample_C, METH_VARARGS, " "},
    {"setHeight_Py", setHeight_C, METH_VARARGS, " "},
    {"setPlanetLocalRadius_Py", setPlanetLocalRadius_C, METH_VARARGS, " "},
    {"setVelocity_Py", setVelocity_C, METH_VARARGS, " "},
    {"setDopplerAccessor_Py", setDopplerAccessor_C,METH_VARARGS, " "},
    {"setPRF_Py", setPRF_C, METH_VARARGS, " "},
    {"setRadarWavelength_Py", setRadarWavelength_C, METH_VARARGS, " "},
    {"setSCoordinateFirstLine_Py", setSCoordinateFirstLine_C, METH_VARARGS,
        " "},
    {"setFirstLatitude_Py", setFirstLatitude_C, METH_VARARGS, " "},
    {"setFirstLongitude_Py", setFirstLongitude_C, METH_VARARGS, " "},
    {"setDeltaLatitude_Py", setDeltaLatitude_C, METH_VARARGS, " "},
    {"setDeltaLongitude_Py", setDeltaLongitude_C, METH_VARARGS, " "},
    {"setLength_Py", setLength_C, METH_VARARGS, " "},
    {"setWidth_Py", setWidth_C, METH_VARARGS, " "},
    {"setLookSide_Py", setLookSide_C, METH_VARARGS, " "},
    {"setNumberRangeLooks_Py", setNumberRangeLooks_C, METH_VARARGS, " "},
    {"setNumberAzimuthLooks_Py", setNumberAzimuthLooks_C, METH_VARARGS, " "},
    {"setNumberPointsPerDemPost_Py", setNumberPointsPerDemPost_C, METH_VARARGS,
        " "},
    {"setISMocomp_Py", setISMocomp_C, METH_VARARGS, " "},
    {"setDemWidth_Py", setDemWidth_C, METH_VARARGS, " "},
    {"setDemLength_Py", setDemLength_C, METH_VARARGS, " "},
    {"allocate_s_mocomp_Py", allocate_s_mocomp_C, METH_VARARGS, " "},
    {"deallocate_s_mocomp_Py", deallocate_s_mocomp_C, METH_VARARGS, " "},
    {"setReferenceOrbit_Py", setReferenceOrbit_C, METH_VARARGS, " "},
    {"getGeoWidth_Py", getGeoWidth_C, METH_VARARGS, " "},
    {"getGeoLength_Py", getGeoLength_C, METH_VARARGS, " "},
    {"getLatitudeSpacing_Py", getLatitudeSpacing_C, METH_VARARGS, " "},
    {"getLongitudeSpacing_Py", getLongitudeSpacing_C, METH_VARARGS, " "},
    {"getMinimumGeoLatitude_Py", getMinimumGeoLatitude_C, METH_VARARGS, " "},
    {"getMinimumGeoLongitude_Py", getMinimumGeoLongitude_C, METH_VARARGS, " "},
    {"getMaximumGeoLatitude_Py", getMaximumGeoLatitude_C, METH_VARARGS, " "},
    {"getMaxmumGeoLongitude_Py", getMaxmumGeoLongitude_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif

// end of file
