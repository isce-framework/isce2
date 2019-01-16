#include <Python.h>
#include "icumodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <fstream>
using namespace std;

static const char * const __doc__ = "Python extension for icu.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "icu",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    icu_methods,
};

// initialization function for the module
// *must* be called PyInit_formslc
PyMODINIT_FUNC
PyInit_icu()
{
    // create the module using moduledef struct defined above
    PyObject * module = PyModule_Create(&moduledef);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return module;
    }
    // otherwise, we have an initialized module
    // and return the newly created module
    return module;
}


PyObject* icu_C(PyObject* self, PyObject *args)
{
    uint64_t intAcc, ampAcc, filtAcc, conncompAcc;
    uint64_t corrAcc, gccAcc, phsigcorrAcc, unwAcc;

    if (!PyArg_ParseTuple(args,"KKKKKKKK", &intAcc, &ampAcc, &filtAcc, &corrAcc, &gccAcc, &phsigcorrAcc, &unwAcc,&conncompAcc))
    {
        return NULL;
    }

    icu_f(&intAcc, &ampAcc, &filtAcc, &corrAcc, &gccAcc, &phsigcorrAcc, &unwAcc, &conncompAcc);
    return Py_BuildValue("i",0);
}

//set state variable methods

PyObject* setWidth_C(PyObject* self, PyObject *args)
{
    int len;
    if (!PyArg_ParseTuple(args,"i",&len))
    {
        return NULL;
    }

    setWidth_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setStartSample_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i",&len))
    {
        return NULL;
    }

    setStartSample_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setEndSample_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setEndSample_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setStartingLine_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setStartingLine_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setLength_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setLength_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setAzimuthBufferSize_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setAzimuthBufferSize_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setOverlap_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setOverlap_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setFilteringFlag_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setFilteringFlag_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setUnwrappingFlag_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setUnwrappingFlag_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setLPRangeWinSize_C(PyObject* self, PyObject* args)
{
    float len;
    if(!PyArg_ParseTuple(args,"f", &len))
    {
        return NULL;
    }

    setLPRangeWinSize_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setLPAzimuthWinSize_C(PyObject* self, PyObject* args)
{
    float len;
    if(!PyArg_ParseTuple(args,"f", &len))
    {
        return NULL;
    }

    setLPAzimuthWinSize_f(&len);
    return Py_BuildValue("i",0);
}


PyObject* setUseAmplitudeFlag_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setUseAmplitudeFlag_f(&len);
    return Py_BuildValue("i",0);
}


PyObject* setCorrelationType_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setCorrelationType_f(&len);
    return Py_BuildValue("i",0);
}


PyObject* setFilterType_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setFilterType_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setFilterExponent_C(PyObject* self, PyObject* args)
{
    float len;
    if(!PyArg_ParseTuple(args,"f", &len))
    {
        return NULL;
    }

    setFilterExponent_f(&len);
    return Py_BuildValue("i",0);
}


PyObject* setCorrelationBoxSize_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setCorrelationBoxSize_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setPhaseSigmaBoxSize_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setPhaseSigmaBoxSize_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setPhaseVarThreshold_C(PyObject* self, PyObject* args)
{
    float len;
    if(!PyArg_ParseTuple(args,"f", &len))
    {
        return NULL;
    }

    setPhaseVarThreshold_f(&len);
    return Py_BuildValue("i",0);
}


PyObject* setInitCorrThreshold_C(PyObject* self, PyObject* args)
{
    float len;
    if(!PyArg_ParseTuple(args,"f", &len))
    {
        return NULL;
    }

    setInitCorrThreshold_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setCorrThreshold_C(PyObject* self, PyObject* args)
{
    float len;
    if(!PyArg_ParseTuple(args,"f", &len))
    {
        return NULL;
    }

    setCorrThreshold_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setCorrThresholdInc_C(PyObject* self, PyObject* args)
{
    float len;
    if(!PyArg_ParseTuple(args,"f", &len))
    {
        return NULL;
    }

    setCorrThresholdInc_f(&len);
    return Py_BuildValue("i",0);
}


PyObject* setNeuTypes_C(PyObject* self, PyObject* args)
{
    int len,len1;
    if(!PyArg_ParseTuple(args,"ii", &len,&len1))
    {
        return NULL;
    }

    setNeuTypes_f(&len, &len1);
    return Py_BuildValue("i",0);
}


PyObject* setNeuThreshold_C(PyObject* self, PyObject* args)
{
    float len, len1, len2;
    if(!PyArg_ParseTuple(args,"fff", &len, &len1, &len2))
    {
        return NULL;
    }

    setNeuThreshold_f(&len, &len1, &len2);
    return Py_BuildValue("i",0);
}


PyObject* setBootstrapSize_C(PyObject* self, PyObject* args)
{
    int len, len1;
    if(!PyArg_ParseTuple(args,"ii", &len, &len1))
    {
        return NULL;
    }

    setBootstrapSize_f(&len, &len1);
    return Py_BuildValue("i",0);
}

PyObject* setNumTreeSets_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setNumTreeSets_f(&len);
    return Py_BuildValue("i",0);
}

PyObject* setTreeType_C(PyObject* self, PyObject* args)
{
    int len;
    if(!PyArg_ParseTuple(args,"i", &len))
    {
        return NULL;
    }

    setTreeType_f(&len);
    return Py_BuildValue("i",0);
}
