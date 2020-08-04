//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#include <Python.h>
#include "formslcmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for formslc.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "formslc",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    formslc_methods,
};

// initialization function for the module
PyMODINIT_FUNC
PyInit_formslc()
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


PyObject * formslc_C(PyObject * self, PyObject * args)
{
    uint64_t ptLAGet = 0;
    uint64_t ptLASet = 0;
    if(!PyArg_ParseTuple(args, "KK", &ptLAGet, &ptLASet))
    {
        return NULL;
    }




    formslc_f(&ptLAGet, &ptLASet);

    return Py_BuildValue("i", 0);

}

PyObject * setNumberGoodBytes_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setNumberGoodBytes_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setNumberBytesPerLine_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setNumberBytesPerLine_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setDebugFlag_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setDebugFlag_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setDeskewFlag_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i",&varInt))
        {
                return NULL;
        }
        setDeskewFlag_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setSecondaryRangeMigrationFlag_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setSecondaryRangeMigrationFlag_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setFirstLine_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setFirstLine_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setNumberPatches_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setNumberPatches_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setFirstSample_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setFirstSample_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setAzimuthPatchSize_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setAzimuthPatchSize_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setNumberValidPulses_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setNumberValidPulses_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setCaltoneLocation_C(PyObject* self, PyObject* args)
{
        float varFloat;
        if(!PyArg_ParseTuple(args, "f", &varFloat))
        {
                return NULL;
        }
        setCaltoneLocation_f(&varFloat);
        return Py_BuildValue("i", 0);
}
PyObject * setStartRangeBin_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setStartRangeBin_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setNumberRangeBin_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setNumberRangeBin_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setDopplerCentroidCoefficients_C(PyObject* self, PyObject* args)
{
        int dim1 = 4;
        PyObject * list;
        if(!PyArg_ParseTuple(args, "O", &list))
        {
                return NULL;
        }
        if(!PyList_Check(list))
        {
                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                exit(1);
        }
        double *  vectorV = new double[dim1];
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyList_GetItem(list,i);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
                        exit(1);
                }
                vectorV[i] = (double) PyFloat_AsDouble(listEl);
                if(PyErr_Occurred() != NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
                        exit(1);
                }
        }
        setDopplerCentroidCoefficients_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setPlanetRadiusOfCurvature_C(PyObject* self, PyObject* args)
{
        double varDouble;
        if(!PyArg_ParseTuple(args, "d", &varDouble))
        {
                return NULL;
        }
        setPlanetRadiusOfCurvature_f(&varDouble);
        return Py_BuildValue("i", 0);
}
PyObject * setBodyFixedVelocity_C(PyObject* self, PyObject* args)
{
        float varFloat;
        if(!PyArg_ParseTuple(args, "f", &varFloat))
        {
                return NULL;
        }
        setBodyFixedVelocity_f(&varFloat);
        return Py_BuildValue("i", 0);
}
PyObject * setSpacecraftHeight_C(PyObject* self, PyObject* args)
{
        double varDouble;
        if(!PyArg_ParseTuple(args, "d", &varDouble))
        {
                return NULL;
        }
        setSpacecraftHeight_f(&varDouble);
        return Py_BuildValue("i", 0);
}
PyObject * setPlanetGravitationalConstant_C(PyObject* self, PyObject* args)
{
        double varDouble;
        if(!PyArg_ParseTuple(args, "d", &varDouble))
        {
                return NULL;
        }
        setPlanetGravitationalConstant_f(&varDouble);
        return Py_BuildValue("i", 0);
}
PyObject * setPointingDirection_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setPointingDirection_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setAntennaSCHVelocity_C(PyObject* self, PyObject* args)
{
        int dim1 = 3;
        PyObject * list;
        if(!PyArg_ParseTuple(args, "O", &list))
        {
                return NULL;
        }
        if(!PyList_Check(list))
        {
                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                exit(1);
        }
        double *  vectorV = new double[dim1];
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyList_GetItem(list,i);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
                        exit(1);
                }
                vectorV[i] = (double) PyFloat_AsDouble(listEl);
                if(PyErr_Occurred() != NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
                        exit(1);
                }
        }
        setAntennaSCHVelocity_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setAntennaSCHAcceleration_C(PyObject* self, PyObject* args)
{
        int dim1 = 3;
        PyObject * list;
        if(!PyArg_ParseTuple(args, "O", &list))
        {
                return NULL;
        }
        if(!PyList_Check(list))
        {
                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                exit(1);
        }
        double *  vectorV = new double[dim1];
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyList_GetItem(list,i);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
                        exit(1);
                }
                vectorV[i] = (double) PyFloat_AsDouble(listEl);
                if(PyErr_Occurred() != NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
                        exit(1);
                }
        }
        setAntennaSCHAcceleration_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setRangeFirstSample_C(PyObject* self, PyObject* args)
{
        double varDouble;
        if(!PyArg_ParseTuple(args, "d", &varDouble))
        {
                return NULL;
        }
        setRangeFirstSample_f(&varDouble);
        return Py_BuildValue("i", 0);
}
PyObject * setPRF_C(PyObject* self, PyObject* args)
{
        float varFloat;
        if(!PyArg_ParseTuple(args, "f", &varFloat))
        {
                return NULL;
        }
        setPRF_f(&varFloat);
        return Py_BuildValue("i", 0);
}
PyObject * setInPhaseValue_C(PyObject* self, PyObject* args)
{
        float varFloat;
        if(!PyArg_ParseTuple(args, "f", &varFloat))
        {
                return NULL;
        }
        setInPhaseValue_f(&varFloat);
        return Py_BuildValue("i", 0);
}
PyObject * setQuadratureValue_C(PyObject* self, PyObject* args)
{
        float varFloat;
        if(!PyArg_ParseTuple(args, "f", &varFloat))
        {
                return NULL;
        }
        setQuadratureValue_f(&varFloat);
        return Py_BuildValue("i", 0);
}
PyObject * setIQFlip_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setIQFlip_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setAzimuthResolution_C(PyObject* self, PyObject* args)
{
        float varFloat;
        if(!PyArg_ParseTuple(args, "f", &varFloat))
        {
                return NULL;
        }
        setAzimuthResolution_f(&varFloat);
        return Py_BuildValue("i", 0);
}
PyObject * setNumberAzimuthLooks_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setNumberAzimuthLooks_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setRangeSamplingRate_C(PyObject* self, PyObject* args)
{
        float varFloat;
        if(!PyArg_ParseTuple(args, "f", &varFloat))
        {
                return NULL;
        }
        setRangeSamplingRate_f(&varFloat);
        return Py_BuildValue("i", 0);
}
PyObject * setChirpSlope_C(PyObject* self, PyObject* args)
{
        float varFloat;
        if(!PyArg_ParseTuple(args, "f", &varFloat))
        {
                return NULL;
        }
        setChirpSlope_f(&varFloat);
        return Py_BuildValue("i", 0);
}
PyObject * setRangePulseDuration_C(PyObject* self, PyObject* args)
{
        float varFloat;
        if(!PyArg_ParseTuple(args, "f", &varFloat))
        {
                return NULL;
        }
        setRangePulseDuration_f(&varFloat);
        return Py_BuildValue("i", 0);
}
PyObject * setRangeChirpExtensionPoints_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setRangeChirpExtensionPoints_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setRadarWavelength_C(PyObject* self, PyObject* args)
{
        double varDouble;
        if(!PyArg_ParseTuple(args, "d", &varDouble))
        {
                return NULL;
        }
        setRadarWavelength_f(&varDouble);
        return Py_BuildValue("i", 0);
}
PyObject * setRangeSpectralWeighting_C(PyObject* self, PyObject* args)
{
        float varFloat;
        if(!PyArg_ParseTuple(args, "f", &varFloat))
        {
                return NULL;
        }
        setRangeSpectralWeighting_f(&varFloat);
        return Py_BuildValue("i", 0);
}
PyObject* getSLCStartingRange_C(PyObject* self, PyObject* args)
{
    double varDbl;
    getSLCStartingRange_f(&varDbl);
    return Py_BuildValue("d", varDbl);
}
PyObject* getSLCStartingLine_C(PyObject *self, PyObject* args)
{
    int varInt;
    getSLCStartingLine_f(&varInt);
    return Py_BuildValue("i", varInt);
}
PyObject * setSpectralShiftFractions_C(PyObject* self, PyObject* args)
{
        int dim1 = 2;
        PyObject * list;
        if(!PyArg_ParseTuple(args, "O", &list))
        {
                return NULL;
        }
        if(!PyList_Check(list))
        {
                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                exit(1);
        }
        float *  vectorV = new float[dim1];
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyList_GetItem(list,i);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
                        exit(1);
                }
                vectorV[i] = (float) PyFloat_AsDouble(listEl);
                if(PyErr_Occurred() != NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
                        exit(1);
                }
        }
        setSpectralShiftFractions_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setLinearResamplingCoefficiets_C(PyObject* self, PyObject* args)
{
        int dim1 = 4;
        PyObject * list;
        if(!PyArg_ParseTuple(args, "O", &list))
        {
                return NULL;
        }
        if(!PyList_Check(list))
        {
                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                exit(1);
        }
        double *  vectorV = new double[dim1];
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyList_GetItem(list,i);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
                        exit(1);
                }
                vectorV[i] = (double) PyFloat_AsDouble(listEl);
                if(PyErr_Occurred() != NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
                        exit(1);
                }
        }
        setLinearResamplingCoefficiets_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setLinearResamplingDeltas_C(PyObject* self, PyObject* args)
{
        int dim1 = 4;
        PyObject * list;
        if(!PyArg_ParseTuple(args, "O", &list))
        {
                return NULL;
        }
        if(!PyList_Check(list))
        {
                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                exit(1);
        }
        double *  vectorV = new double[dim1];
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyList_GetItem(list,i);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
                        exit(1);
                }
                vectorV[i] = (double) PyFloat_AsDouble(listEl);
                if(PyErr_Occurred() != NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
                        exit(1);
                }
        }
        setLinearResamplingDeltas_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}
