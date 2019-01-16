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





#ifndef topomodule_h
#define topomodule_h

#include <Python.h>
#include <stdint.h>
#include "topomoduleFortTrans.h"

extern "C"
{
        #include "orbit.h"

        void topo_f(uint64_t *, uint64_t *);
        PyObject * topo_C(PyObject *, PyObject *);
        void setNumberIterations_f(int *);
        PyObject * setNumberIterations_C(PyObject *, PyObject *);
        void setDemWidth_f(int *);
        PyObject * setDemWidth_C(PyObject *, PyObject *);
        void setDemLength_f(int *);
        PyObject * setDemLength_C(PyObject *, PyObject *);
        void setReferenceOrbit_f(double *, int *);
        void allocate_s_mocompArray_f(int *);
        void deallocate_s_mocompArray_f();
        PyObject * allocate_s_mocompArray_C(PyObject *, PyObject *);
        PyObject * deallocate_s_mocompArray_C(PyObject *, PyObject *);
        PyObject * setReferenceOrbit_C(PyObject *, PyObject *);
        void setFirstLatitude_f(double *);
        PyObject * setFirstLatitude_C(PyObject *, PyObject *);
        void setFirstLongitude_f(double *);
        PyObject * setFirstLongitude_C(PyObject *, PyObject *);
        void setDeltaLatitude_f(double *);
        PyObject * setDeltaLatitude_C(PyObject *, PyObject *);
        void setDeltaLongitude_f(double *);
        PyObject * setDeltaLongitude_C(PyObject *, PyObject *);
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
        void setLookSide_f(int *);
        PyObject * setLookSide_C(PyObject *, PyObject *);
        void setPegLatitude_f(double *);
        PyObject * setPegLatitude_C(PyObject *, PyObject *);
        void setPegLongitude_f(double *);
        PyObject * setPegLongitude_C(PyObject *, PyObject *);
        void setPegHeading_f(double *);
        PyObject * setPegHeading_C(PyObject *, PyObject *);
        void setPRF_f(double *);
        PyObject * setPRF_C(PyObject *, PyObject *);
        void setMethod_f(int *);
        PyObject * setMethod_C(PyObject *, PyObject *);
        void setRadarWavelength_f(double *);
        PyObject * setRadarWavelength_C(PyObject *, PyObject *);
        void setLatitudePointer_f(uint64_t *);
        PyObject * setLatitudePointer_C(PyObject *, PyObject *);
        void setLongitudePointer_f(uint64_t *);
        PyObject * setLongitudePointer_C(PyObject *, PyObject *);
        void setHeightRPointer_f(uint64_t *);
        PyObject * setHeightRPointer_C(PyObject *, PyObject *);
        void setHeightSchPointer_f(uint64_t *);
        PyObject * setHeightSchPointer_C(PyObject *, PyObject *);
        void setLosPointer_f(uint64_t *);
        PyObject * setLosPointer_C(PyObject *, PyObject *);
        void setIncPointer_f(uint64_t *);
        PyObject * setIncPointer_C(PyObject *, PyObject *);
        void getAzimuthSpacing_f(double *);
        PyObject * getAzimuthSpacing_C(PyObject *, PyObject *);
        void getPlanetLocalRadius_f(double *);
        PyObject * getPlanetLocalRadius_C(PyObject *, PyObject *);
        void getSCoordinateFirstLine_f(double *);
        PyObject * getSCoordinateFirstLine_C(PyObject *, PyObject *);
        void getSCoordinateLastLine_f(double *);
        PyObject * getSCoordinateLastLine_C(PyObject *, PyObject *);
        void getMinimumLatitude_f(double *);
        PyObject * getMinimumLatitude_C(PyObject *, PyObject *);
        void getMinimumLongitude_f(double *);
        PyObject * getMinimumLongitude_C(PyObject *, PyObject *);
        void getMaximumLatitude_f(double *);
        PyObject * getMaximumLatitude_C(PyObject *, PyObject *);
        void getMaximumLongitude_f(double *);
        PyObject * getMaximumLongitude_C(PyObject *, PyObject *);
        void getLength_f(int *);
        PyObject * getLength_C(PyObject *, PyObject *);
        void getSquintShift_f(double *, int *);
        void allocate_squintshift_f(int *);
        void deallocate_squintshift_f();
        PyObject * allocate_squintshift_C(PyObject *, PyObject *);
        PyObject * deallocate_squintshift_C(PyObject *, PyObject *);
        PyObject * getSquintShift_C(PyObject *, PyObject *);

        void setSensingStart_f(double*);
        PyObject * setSensingStart_C(PyObject*, PyObject*);

        void setOrbit_f(cOrbit*);
        PyObject * setOrbit_C(PyObject*, PyObject*);
}

static PyMethodDef topo_methods[] =
{
        {"topo_Py", topo_C, METH_VARARGS, " "},
        {"setNumberIterations_Py", setNumberIterations_C, METH_VARARGS, " "},
        {"setDemWidth_Py", setDemWidth_C, METH_VARARGS, " "},
        {"setDemLength_Py", setDemLength_C, METH_VARARGS, " "},
        {"allocate_s_mocompArray_Py", allocate_s_mocompArray_C, METH_VARARGS, " "},
        {"deallocate_s_mocompArray_Py", deallocate_s_mocompArray_C, METH_VARARGS, " "},
        {"setReferenceOrbit_Py", setReferenceOrbit_C, METH_VARARGS, " "},
        {"setFirstLatitude_Py", setFirstLatitude_C, METH_VARARGS, " "},
        {"setFirstLongitude_Py", setFirstLongitude_C, METH_VARARGS, " "},
        {"setDeltaLatitude_Py", setDeltaLatitude_C, METH_VARARGS, " "},
        {"setDeltaLongitude_Py", setDeltaLongitude_C, METH_VARARGS, " "},
        {"setISMocomp_Py", setISMocomp_C, METH_VARARGS, " "},
        {"setEllipsoidMajorSemiAxis_Py", setEllipsoidMajorSemiAxis_C, METH_VARARGS, " "},
        {"setEllipsoidEccentricitySquared_Py", setEllipsoidEccentricitySquared_C, METH_VARARGS, " "},
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
        {"setLatitudePointer_Py", setLatitudePointer_C, METH_VARARGS, " "},
        {"setLongitudePointer_Py", setLongitudePointer_C, METH_VARARGS, " "},
        {"setHeightRPointer_Py", setHeightRPointer_C, METH_VARARGS, " "},
        {"setHeightSchPointer_Py", setHeightSchPointer_C, METH_VARARGS, " "},
        {"setLosPointer_Py", setLosPointer_C, METH_VARARGS, " "},
        {"setIncPointer_Py", setIncPointer_C, METH_VARARGS, " "},
        {"setLookSide_Py", setLookSide_C, METH_VARARGS, " "},
        {"setMethod_Py", setMethod_C, METH_VARARGS, " "},
        {"getAzimuthSpacing_Py", getAzimuthSpacing_C, METH_VARARGS, " "},
        {"getPlanetLocalRadius_Py", getPlanetLocalRadius_C, METH_VARARGS, " "},
        {"getSCoordinateFirstLine_Py", getSCoordinateFirstLine_C, METH_VARARGS, " "},
        {"getSCoordinateLastLine_Py", getSCoordinateLastLine_C, METH_VARARGS, " "},
        {"getMinimumLatitude_Py", getMinimumLatitude_C, METH_VARARGS, " "},
        {"getMinimumLongitude_Py", getMinimumLongitude_C, METH_VARARGS, " "},
        {"getMaximumLatitude_Py", getMaximumLatitude_C, METH_VARARGS, " "},
        {"getMaximumLongitude_Py", getMaximumLongitude_C, METH_VARARGS, " "},
        {"getLength_Py", getLength_C, METH_VARARGS, " "},
        {"allocate_squintshift_Py", allocate_squintshift_C, METH_VARARGS, " "},
        {"deallocate_squintshift_Py", deallocate_squintshift_C, METH_VARARGS, " "},
        {"getSquintShift_Py", getSquintShift_C, METH_VARARGS, " "},
        {"setSensingStart_Py", setSensingStart_C, METH_VARARGS, " "},
        {"setOrbit_Py", setOrbit_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //topomodule_h
