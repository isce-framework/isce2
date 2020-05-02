//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef formslcmodule_h
#define formslcmodule_h

#include "formslcmoduleFortTrans.h"
#include <stdint.h>

extern "C"
{
        void formslc_f(uint64_t *, uint64_t *);
        PyObject * formslc_C(PyObject *, PyObject *);
        void setNumberGoodBytes_f(int *);
        PyObject * setNumberGoodBytes_C(PyObject *, PyObject *);
        void setNumberBytesPerLine_f(int *);
        PyObject * setNumberBytesPerLine_C(PyObject *, PyObject *);
        void setDebugFlag_f(int *);
        PyObject * setDebugFlag_C(PyObject *, PyObject *);
        void setDeskewFlag_f(int *);
        PyObject * setDeskewFlag_C(PyObject *, PyObject *);
        void setSecondaryRangeMigrationFlag_f(int *);
        PyObject * setSecondaryRangeMigrationFlag_C(PyObject *, PyObject *);
        void setFirstLine_f(int *);
        PyObject * setFirstLine_C(PyObject *, PyObject *);
        void setNumberPatches_f(int *);
        PyObject * setNumberPatches_C(PyObject *, PyObject *);
        void setFirstSample_f(int *);
        PyObject * setFirstSample_C(PyObject *, PyObject *);
        void setAzimuthPatchSize_f(int *);
        PyObject * setAzimuthPatchSize_C(PyObject *, PyObject *);
        void setNumberValidPulses_f(int *);
        PyObject * setNumberValidPulses_C(PyObject *, PyObject *);
        void setCaltoneLocation_f(float *);
        PyObject * setCaltoneLocation_C(PyObject *, PyObject *);
        void setStartRangeBin_f(int *);
        PyObject * setStartRangeBin_C(PyObject *, PyObject *);
        void setNumberRangeBin_f(int *);
        PyObject * setNumberRangeBin_C(PyObject *, PyObject *);
        void setDopplerCentroidCoefficients_f(double *, int *);
        PyObject * setDopplerCentroidCoefficients_C(PyObject *, PyObject *);
        void setPlanetRadiusOfCurvature_f(double *);
        PyObject * setPlanetRadiusOfCurvature_C(PyObject *, PyObject *);
        void setBodyFixedVelocity_f(float *);
        PyObject * setBodyFixedVelocity_C(PyObject *, PyObject *);
        void setSpacecraftHeight_f(double *);
        PyObject * setSpacecraftHeight_C(PyObject *, PyObject *);
        void setPlanetGravitationalConstant_f(double *);
        PyObject * setPlanetGravitationalConstant_C(PyObject *, PyObject *);
        void setPointingDirection_f(int *);
        PyObject * setPointingDirection_C(PyObject *, PyObject *);
        void setAntennaSCHVelocity_f(double *, int *);
        PyObject * setAntennaSCHVelocity_C(PyObject *, PyObject *);
        void setAntennaSCHAcceleration_f(double *, int *);
        PyObject * setAntennaSCHAcceleration_C(PyObject *, PyObject *);
        void setRangeFirstSample_f(double *);
        PyObject * setRangeFirstSample_C(PyObject *, PyObject *);
        void setPRF_f(float *);
        PyObject * setPRF_C(PyObject *, PyObject *);
        void setInPhaseValue_f(float *);
        PyObject * setInPhaseValue_C(PyObject *, PyObject *);
        void setQuadratureValue_f(float *);
        PyObject * setQuadratureValue_C(PyObject *, PyObject *);
        void setIQFlip_f(int *);
        PyObject * setIQFlip_C(PyObject *, PyObject *);
        void setAzimuthResolution_f(float *);
        PyObject * setAzimuthResolution_C(PyObject *, PyObject *);
        void setNumberAzimuthLooks_f(int *);
        PyObject * setNumberAzimuthLooks_C(PyObject *, PyObject *);
        void setRangeSamplingRate_f(float *);
        PyObject * setRangeSamplingRate_C(PyObject *, PyObject *);
        void setChirpSlope_f(float *);
        PyObject * setChirpSlope_C(PyObject *, PyObject *);
        void setRangePulseDuration_f(float *);
        PyObject * setRangePulseDuration_C(PyObject *, PyObject *);
        void setRangeChirpExtensionPoints_f(int *);
        PyObject * setRangeChirpExtensionPoints_C(PyObject *, PyObject *);
        void setRadarWavelength_f(double *);
        PyObject * setRadarWavelength_C(PyObject *, PyObject *);
        void setRangeSpectralWeighting_f(float *);
        PyObject * setRangeSpectralWeighting_C(PyObject *, PyObject *);
        void setSpectralShiftFractions_f(float *, int *);
        PyObject * setSpectralShiftFractions_C(PyObject *, PyObject *);
        void setLinearResamplingCoefficiets_f(double *, int *);
        PyObject * setLinearResamplingCoefficiets_C(PyObject *, PyObject *);
        void setLinearResamplingDeltas_f(double *, int *);
        PyObject * setLinearResamplingDeltas_C(PyObject *, PyObject *);
        void getSLCStartingRange_f(double*);
        PyObject * getSLCStartingRange_C(PyObject*, PyObject*);
        void getSLCStartingLine_f(int*);
        PyObject * getSLCStartingLine_C(PyObject*, PyObject*);

}

static PyMethodDef formslc_methods[] =
{
        {"formslc_Py", formslc_C, METH_VARARGS, " "},
        {"setNumberGoodBytes_Py", setNumberGoodBytes_C, METH_VARARGS, " "},
        {"setNumberBytesPerLine_Py", setNumberBytesPerLine_C, METH_VARARGS, " "},
        {"setDebugFlag_Py", setDebugFlag_C, METH_VARARGS, " "},
        {"setDeskewFlag_Py", setDeskewFlag_C, METH_VARARGS, " "},
        {"setSecondaryRangeMigrationFlag_Py", setSecondaryRangeMigrationFlag_C, METH_VARARGS, " "},
        {"setFirstLine_Py", setFirstLine_C, METH_VARARGS, " "},
        {"setNumberPatches_Py", setNumberPatches_C, METH_VARARGS, " "},
        {"setFirstSample_Py", setFirstSample_C, METH_VARARGS, " "},
        {"setAzimuthPatchSize_Py", setAzimuthPatchSize_C, METH_VARARGS, " "},
        {"setNumberValidPulses_Py", setNumberValidPulses_C, METH_VARARGS, " "},
        {"setCaltoneLocation_Py", setCaltoneLocation_C, METH_VARARGS, " "},
        {"setStartRangeBin_Py", setStartRangeBin_C, METH_VARARGS, " "},
        {"setNumberRangeBin_Py", setNumberRangeBin_C, METH_VARARGS, " "},
        {"setDopplerCentroidCoefficients_Py", setDopplerCentroidCoefficients_C, METH_VARARGS, " "},
        {"setPlanetRadiusOfCurvature_Py", setPlanetRadiusOfCurvature_C, METH_VARARGS, " "},
        {"setBodyFixedVelocity_Py", setBodyFixedVelocity_C, METH_VARARGS, " "},
        {"setSpacecraftHeight_Py", setSpacecraftHeight_C, METH_VARARGS, " "},
        {"setPlanetGravitationalConstant_Py", setPlanetGravitationalConstant_C, METH_VARARGS, " "},
        {"setPointingDirection_Py", setPointingDirection_C, METH_VARARGS, " "},
        {"setAntennaSCHVelocity_Py", setAntennaSCHVelocity_C, METH_VARARGS, " "},
        {"setAntennaSCHAcceleration_Py", setAntennaSCHAcceleration_C, METH_VARARGS, " "},
        {"setRangeFirstSample_Py", setRangeFirstSample_C, METH_VARARGS, " "},
        {"setPRF_Py", setPRF_C, METH_VARARGS, " "},
        {"setInPhaseValue_Py", setInPhaseValue_C, METH_VARARGS, " "},
        {"setQuadratureValue_Py", setQuadratureValue_C, METH_VARARGS, " "},
        {"setIQFlip_Py", setIQFlip_C, METH_VARARGS, " "},
        {"setAzimuthResolution_Py", setAzimuthResolution_C, METH_VARARGS, " "},
        {"setNumberAzimuthLooks_Py", setNumberAzimuthLooks_C, METH_VARARGS, " "},
        {"setRangeSamplingRate_Py", setRangeSamplingRate_C, METH_VARARGS, " "},
        {"setChirpSlope_Py", setChirpSlope_C, METH_VARARGS, " "},
        {"setRangePulseDuration_Py", setRangePulseDuration_C, METH_VARARGS, " "},
        {"setRangeChirpExtensionPoints_Py", setRangeChirpExtensionPoints_C, METH_VARARGS, " "},
        {"setRadarWavelength_Py", setRadarWavelength_C, METH_VARARGS, " "},
        {"setRangeSpectralWeighting_Py", setRangeSpectralWeighting_C, METH_VARARGS, " "},
        {"setSpectralShiftFractions_Py", setSpectralShiftFractions_C, METH_VARARGS, " "},
        {"setLinearResamplingCoefficiets_Py", setLinearResamplingCoefficiets_C, METH_VARARGS, " "},
        {"setLinearResamplingDeltas_Py", setLinearResamplingDeltas_C, METH_VARARGS, " "},
        {"getSLCStartingRange_Py", getSLCStartingRange_C, METH_VARARGS, " "},
        {"getSLCStartingLine_Py", getSLCStartingLine_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //formslcmodule_h
