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




#ifndef formslcmodule_h
#define formslcmodule_h

#include <Python.h>
#include <stdint.h>
#include "formslcmoduleFortTrans.h"

extern "C"
{
    #include "orbit.h"

    void setStdWriter_f(uint64_t *);
    PyObject * setStdWriter_C(PyObject *, PyObject *);
    void formslc_f(uint64_t *,uint64_t *);
    PyObject * formslc_C(PyObject *, PyObject *);
    void setNumberGoodBytes_f(int *);
    PyObject * setNumberGoodBytes_C(PyObject *, PyObject *);
    void setNumberBytesPerLine_f(int *);
    PyObject * setNumberBytesPerLine_C(PyObject *, PyObject *);
    void setFirstLine_f(int *);
    PyObject * setFirstLine_C(PyObject *, PyObject *);
    void setNumberValidPulses_f(int *);
    PyObject * setNumberValidPulses_C(PyObject *, PyObject *);
    void setFirstSample_f(int *);
    PyObject * setFirstSample_C(PyObject *, PyObject *);
    void setNumberPatches_f(int *);
    PyObject * setNumberPatches_C(PyObject *, PyObject *);
    void setStartRangeBin_f(int *);
    PyObject * setStartRangeBin_C(PyObject *, PyObject *);
    void setNumberRangeBin_f(int *);
    PyObject * setNumberRangeBin_C(PyObject *, PyObject *);
    void setNumberAzimuthLooks_f(int *);
    PyObject * setNumberAzimuthLooks_C(PyObject *, PyObject *);
    void setRangeChirpExtensionPoints_f(int *);
    PyObject * setRangeChirpExtensionPoints_C(PyObject *, PyObject *);
    void setAzimuthPatchSize_f(int *);
    PyObject * setAzimuthPatchSize_C(PyObject *, PyObject *);
    void setOverlap_f(int *);
    PyObject * setOverlap_C(PyObject *, PyObject *);
    void setRanfftov_f(int *);
    PyObject * setRanfftov_C(PyObject *, PyObject *);
    void setRanfftiq_f(int *);
    PyObject * setRanfftiq_C(PyObject *, PyObject *);
    void setDebugFlag_f(int *);
    PyObject * setDebugFlag_C(PyObject *, PyObject *);
    void setCaltoneLocation_f(double *);
    PyObject * setCaltoneLocation_C(PyObject *, PyObject *);
    void setPlanetLocalRadius_f(double *);
    PyObject * setPlanetLocalRadius_C(PyObject *, PyObject *);
    void setBodyFixedVelocity_f(double *);
    PyObject * setBodyFixedVelocity_C(PyObject *, PyObject *);
    void setSpacecraftHeight_f(double *);
    PyObject * setSpacecraftHeight_C(PyObject *, PyObject *);
    void setPRF_f(double *);
    PyObject * setPRF_C(PyObject *, PyObject *);
    void setInPhaseValue_f(double *);
    PyObject * setInPhaseValue_C(PyObject *, PyObject *);
    void setQuadratureValue_f(double *);
    PyObject * setQuadratureValue_C(PyObject *, PyObject *);
    void setAzimuthResolution_f(double *);
    PyObject * setAzimuthResolution_C(PyObject *, PyObject *);
    void setRangeSamplingRate_f(double *);
    PyObject * setRangeSamplingRate_C(PyObject *, PyObject *);
    void setChirpSlope_f(double *);
    PyObject * setChirpSlope_C(PyObject *, PyObject *);
    void setRangePulseDuration_f(double *);
    PyObject * setRangePulseDuration_C(PyObject *, PyObject *);
    void setRadarWavelength_f(double *);
    PyObject * setRadarWavelength_C(PyObject *, PyObject *);
    void setRangeFirstSample_f(double *);
    PyObject * setRangeFirstSample_C(PyObject *, PyObject *);
    void setRangeSpectralWeighting_f(double *);
    PyObject * setRangeSpectralWeighting_C(PyObject *, PyObject *);
    void setSpectralShiftFraction_f(double *);
    PyObject * setSpectralShiftFraction_C(PyObject *, PyObject *);
    void setIMRC1_f(uint64_t *);
    PyObject * setIMRC1_C(PyObject *, PyObject *);
    void setIMMocomp_f(uint64_t *);
    PyObject * setIMMocomp_C(PyObject *, PyObject *);
    void setIMRCAS1_f(uint64_t *);
    PyObject * setIMRCAS1_C(PyObject *, PyObject *);
    void setIMRCRM1_f(uint64_t *);
    PyObject * setIMRCRM1_C(PyObject *, PyObject *);
    void setTransDat_f(uint64_t *);
    PyObject * setTransDat_C(PyObject *, PyObject *);
    void setIQFlip_f(char *, int *);
    PyObject * setIQFlip_C(PyObject *, PyObject *);
    void setDeskewFlag_f(char *, int *);
    PyObject * setDeskewFlag_C(PyObject *, PyObject *);
    void setSecondaryRangeMigrationFlag_f(char *, int *);
    PyObject * setSecondaryRangeMigrationFlag_C(PyObject *, PyObject *);
    void setPosition_f(double *, int *, int *);
    void allocate_sch_f(int *,int *);
    void deallocate_sch_f();
    PyObject * allocate_sch_C(PyObject *, PyObject *);
    PyObject * deallocate_sch_C(PyObject *, PyObject *);
    PyObject * setPosition_C(PyObject *, PyObject *);
    void setVelocity_f(double *, int *, int *);
    void allocate_vsch_f(int *,int *);
    void deallocate_vsch_f();
    PyObject * allocate_vsch_C(PyObject *, PyObject *);
    PyObject * deallocate_vsch_C(PyObject *, PyObject *);
    PyObject * setVelocity_C(PyObject *, PyObject *);
    void setTime_f(double *, int *);
    void allocate_time_f(int *);
    void deallocate_time_f();
    PyObject * allocate_time_C(PyObject *, PyObject *);
    PyObject * deallocate_time_C(PyObject *, PyObject *);
    PyObject * setTime_C(PyObject *, PyObject *);
    void setDopplerCentroidCoefficients_f(double *, int *);
    void allocate_dopplerCoefficients_f(int *);
    void deallocate_dopplerCoefficients_f();
    PyObject * allocate_dopplerCoefficients_C(PyObject *, PyObject *);
    PyObject * deallocate_dopplerCoefficients_C(PyObject *, PyObject *);
    PyObject * setDopplerCentroidCoefficients_C(PyObject *, PyObject *);
    PyObject *setPegPoint_C(PyObject *self, PyObject *args);
    void setPegPoint_f(double *lat, double *lon, double *hdg);
    void getMocompPosition_f(double *, int *, int *);
    PyObject * getMocompPosition_C(PyObject *, PyObject *);
    void getMocompIndex_f(int *, int *);
    PyObject * getMocompIndex_C(PyObject *, PyObject *);
    void getMocompPositionSize_f(int *);
    PyObject * getMocompPositionSize_C(PyObject *, PyObject *);
    PyObject *setEllipsoid_C(PyObject *self, PyObject *args);
    void setEllipsoid_f(double *a, double *e2);
    PyObject *setPlanet_C(PyObject *self, PyObject *args);
    void setPlanet_f(double *spin, double *gm);
    PyObject *setSlcWidth_C(PyObject *self, PyObject *args);
    void setSlcWidth_f(int *);
    PyObject *getStartingRange_C(PyObject *, PyObject *);
    void getStartingRange_f(double *);
    PyObject *setStartingRange_C(PyObject *, PyObject *);
    void setStartingRange_f(double *);
    PyObject *setLookSide_C(PyObject *, PyObject *);
    void setLookSide_f(int *);
    void setShift_f(double *); //ML
    PyObject * setShift_C(PyObject *, PyObject *); //ML

    void setOrbit_f(cOrbit*);
    PyObject *setOrbit_C(PyObject *, PyObject*);

    void setSensingStart_f(double*);
    PyObject *setSensingStart_C(PyObject*, PyObject*);

    void setMocompOrbit_f(cOrbit*);
    PyObject *setMocompOrbit_C(PyObject*, PyObject*);

    void getMocompRange_f(double*);
    PyObject *getMocompRange_C(PyObject*, PyObject*);

    void getSlcSensingStart_f(double*);
    PyObject *getSlcSensingStart_C(PyObject*, PyObject*);
}

static PyMethodDef formslc_methods[] =
{
    {"setStdWriter_Py", setStdWriter_C, METH_VARARGS, " "},
    {"formslc_Py", formslc_C, METH_VARARGS, " "},
    {"setNumberGoodBytes_Py", setNumberGoodBytes_C, METH_VARARGS, " "},
    {"setNumberBytesPerLine_Py", setNumberBytesPerLine_C, METH_VARARGS, " "},
    {"setFirstLine_Py", setFirstLine_C, METH_VARARGS, " "},
    {"setNumberValidPulses_Py", setNumberValidPulses_C, METH_VARARGS, " "},
    {"setFirstSample_Py", setFirstSample_C, METH_VARARGS, " "},
    {"setNumberPatches_Py", setNumberPatches_C, METH_VARARGS, " "},
    {"setStartRangeBin_Py", setStartRangeBin_C, METH_VARARGS, " "},
    {"setNumberRangeBin_Py", setNumberRangeBin_C, METH_VARARGS, " "},
    {"setNumberAzimuthLooks_Py", setNumberAzimuthLooks_C, METH_VARARGS, " "},
    {"setRangeChirpExtensionPoints_Py", setRangeChirpExtensionPoints_C,
        METH_VARARGS, " "},
    {"setAzimuthPatchSize_Py", setAzimuthPatchSize_C, METH_VARARGS, " "},
    {"setOverlap_Py", setOverlap_C, METH_VARARGS, " "},
    {"setRanfftov_Py", setRanfftov_C, METH_VARARGS, " "},
    {"setRanfftiq_Py", setRanfftiq_C, METH_VARARGS, " "},
    {"setDebugFlag_Py", setDebugFlag_C, METH_VARARGS, " "},
    {"setCaltoneLocation_Py", setCaltoneLocation_C, METH_VARARGS, " "},
    {"setPlanetLocalRadius_Py", setPlanetLocalRadius_C, METH_VARARGS, " "},
    {"setBodyFixedVelocity_Py", setBodyFixedVelocity_C, METH_VARARGS, " "},
    {"setSpacecraftHeight_Py", setSpacecraftHeight_C, METH_VARARGS, " "},
    {"setPRF_Py", setPRF_C, METH_VARARGS, " "},
    {"setInPhaseValue_Py", setInPhaseValue_C, METH_VARARGS, " "},
    {"setQuadratureValue_Py", setQuadratureValue_C, METH_VARARGS, " "},
    {"setAzimuthResolution_Py", setAzimuthResolution_C, METH_VARARGS, " "},
    {"setRangeSamplingRate_Py", setRangeSamplingRate_C, METH_VARARGS, " "},
    {"setChirpSlope_Py", setChirpSlope_C, METH_VARARGS, " "},
    {"setRangePulseDuration_Py", setRangePulseDuration_C, METH_VARARGS, " "},
    {"setRadarWavelength_Py", setRadarWavelength_C, METH_VARARGS, " "},
    {"setRangeFirstSample_Py", setRangeFirstSample_C, METH_VARARGS, " "},
    {"setRangeSpectralWeighting_Py", setRangeSpectralWeighting_C, METH_VARARGS,
        " "},
    {"setSpectralShiftFraction_Py", setSpectralShiftFraction_C, METH_VARARGS,
        " "},
    {"setIMRC1_Py", setIMRC1_C, METH_VARARGS, " "},
    {"setIMMocomp_Py", setIMMocomp_C, METH_VARARGS, " "},
    {"setIMRCAS1_Py", setIMRCAS1_C, METH_VARARGS, " "},
    {"setIMRCRM1_Py", setIMRCRM1_C, METH_VARARGS, " "},
    {"setTransDat_Py", setTransDat_C, METH_VARARGS, " "},
    {"setIQFlip_Py", setIQFlip_C, METH_VARARGS, " "},
    {"setDeskewFlag_Py", setDeskewFlag_C, METH_VARARGS, " "},
    {"setSecondaryRangeMigrationFlag_Py", setSecondaryRangeMigrationFlag_C,
        METH_VARARGS, " "},
    {"allocate_sch_Py", allocate_sch_C, METH_VARARGS, " "},
    {"deallocate_sch_Py", deallocate_sch_C, METH_VARARGS, " "},
    {"allocate_vsch_Py", allocate_vsch_C, METH_VARARGS, " "},
    {"deallocate_vsch_Py", deallocate_vsch_C, METH_VARARGS, " "},
    {"setPosition_Py", setPosition_C, METH_VARARGS, " "},
    {"setVelocity_Py", setVelocity_C, METH_VARARGS, " "},
    {"allocate_time_Py", allocate_time_C, METH_VARARGS, " "},
    {"deallocate_time_Py", deallocate_time_C, METH_VARARGS, " "},
    {"setTime_Py", setTime_C, METH_VARARGS, " "},
    {"allocate_dopplerCoefficients_Py", allocate_dopplerCoefficients_C,
        METH_VARARGS, " "},
    {"deallocate_dopplerCoefficients_Py", deallocate_dopplerCoefficients_C,
        METH_VARARGS, " "},
    {"setDopplerCentroidCoefficients_Py", setDopplerCentroidCoefficients_C,
        METH_VARARGS, " "},
    {"setPegPoint_Py", setPegPoint_C, METH_VARARGS, " "},
    {"getMocompPosition_Py", getMocompPosition_C, METH_VARARGS, " "},
    {"getMocompIndex_Py", getMocompIndex_C, METH_VARARGS, " "},
    {"getMocompPositionSize_Py", getMocompPositionSize_C, METH_VARARGS, " "},
    {"setEllipsoid_Py", setEllipsoid_C, METH_VARARGS, " "},
    {"setPlanet_Py", setPlanet_C, METH_VARARGS, " "},
    {"setSlcWidth_Py", setSlcWidth_C, METH_VARARGS, " "},
    {"getStartingRange_Py", getStartingRange_C, METH_VARARGS, " "},
    {"setStartingRange_Py", setStartingRange_C, METH_VARARGS, " "},
    {"setLookSide_Py", setLookSide_C, METH_VARARGS, " "},
    {"setShift_Py", setShift_C, METH_VARARGS, " "}, //ML
    {"setOrbit_Py", setOrbit_C, METH_VARARGS, " "},
    {"setSensingStart_Py", setSensingStart_C, METH_VARARGS, " "},
    {"setMocompOrbit_Py", setMocompOrbit_C, METH_VARARGS, " "},
    {"getMocompRange_Py", getMocompRange_C, METH_VARARGS, " "},
    {"getSlcSensingStart_Py", getSlcSensingStart_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};
#endif

// end of file

