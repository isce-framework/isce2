//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
// Author: Piyush Agram
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "estambmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for estamb.F";

PyModuleDef moduledef = {
    //header
    PyModuleDef_HEAD_INIT,
    //name of the module
    "estamb",
    //module documentation string
    __doc__,
    //size of the per-interpreter state of the module
    //-1 if this state is global
    -1,
    estamb_methods,
};

//initialization function for the module
//// *must* be called PyInit_estamb
PyMODINIT_FUNC
PyInit_estamb()
{
    //create the module using moduledef struct defined above
    PyObject * module = PyModule_Create(&moduledef);
    //check whether module create succeeded and raise exception if not
    if(!module)
    {
        return module;
    }
    //otherwise we have an initialized module
    //and return the newly created module
    return module;
}

PyObject * setStdWriter_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setStdWriter_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * allocate_sch_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    int dim2 = 0;
    if(!PyArg_ParseTuple(args, "ii", &dim1, &dim2))
    {
        return NULL;
    }
    allocate_sch_f(&dim1, &dim2);
    return Py_BuildValue("i", 0);
}

PyObject * deallocate_sch_C(PyObject* self, PyObject* args)
{
    deallocate_sch_f();
    return Py_BuildValue("i", 0);
}

PyObject * allocate_vsch_C(PyObject *self, PyObject *args)
{
    int dim1 = 0;
    int dim2 = 0;
    if(!PyArg_ParseTuple(args,"ii",&dim1,&dim2))
    {
        return NULL;
    }
    allocate_vsch_f(&dim1, &dim2);
    return Py_BuildValue("i",0);
}

PyObject * deallocate_vsch_C(PyObject *self, PyObject *args)
{
    deallocate_vsch_f();
    return Py_BuildValue("i",0);
}

PyObject * allocate_entropy_C(PyObject *self, PyObject *args)
{
    int dim1 = 0;
    if(!PyArg_ParseTuple(args,"i",&dim1))
    {
    return NULL;
    }
    allocate_entropy_f(&dim1);
    return Py_BuildValue("i", 0);
}

PyObject * deallocate_entropy_C(PyObject *self, PyObject *args)
{
    deallocate_entropy_f();
    return Py_BuildValue("i",0);
}

PyObject * allocate_time_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    if(!PyArg_ParseTuple(args, "i", &dim1))
    {
        return NULL;
    }
    allocate_time_f(&dim1);
    return Py_BuildValue("i", 0);
}

PyObject * deallocate_time_C(PyObject* self, PyObject* args)
{
    deallocate_time_f();
    return Py_BuildValue("i", 0);
}

PyObject * allocate_dopplerCoefficients_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    if(!PyArg_ParseTuple(args, "i", &dim1))
    {
        return NULL;
    }
    allocate_dopplerCoefficients_f(&dim1);
    return Py_BuildValue("i", 0);
}

PyObject * deallocate_dopplerCoefficients_C(PyObject* self, PyObject* args)
{
    deallocate_dopplerCoefficients_f();
    return Py_BuildValue("i", 0);
}


PyObject * estamb_C(PyObject* self, PyObject* args)
{
    uint64_t var0;
    if(!PyArg_ParseTuple(args, "K",&var0))
    {
        return NULL;
    }
    estamb_f(&var0);
    return Py_BuildValue("i", 0);
}
PyObject * setNumberGoodBytes_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNumberGoodBytes_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setNumberBytesPerLine_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNumberBytesPerLine_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setFirstLine_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setFirstLine_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setLookSide_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
    return NULL;
    }
    setLookSide_f(&var);
    return Py_BuildValue("i",0);
}
PyObject * setNumberValidPulses_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNumberValidPulses_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setFirstSample_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setFirstSample_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setNumberPatches_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNumberPatches_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setStartRangeBin_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setStartRangeBin_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setNumberRangeBin_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setNumberRangeBin_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRangeChirpExtensionPoints_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setRangeChirpExtensionPoints_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setAzimuthPatchSize_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setAzimuthPatchSize_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setOverlap_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setOverlap_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRanfftov_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setRanfftov_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRanfftiq_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setRanfftiq_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setDebugFlag_C(PyObject* self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setDebugFlag_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setCaltoneLocation_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setCaltoneLocation_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setPlanetLocalRadius_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setPlanetLocalRadius_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setBodyFixedVelocity_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setBodyFixedVelocity_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setSpacecraftHeight_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setSpacecraftHeight_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setPRF_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setPRF_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setInPhaseValue_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setInPhaseValue_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setQuadratureValue_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setQuadratureValue_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setAzimuthResolution_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setAzimuthResolution_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRangeSamplingRate_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setRangeSamplingRate_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setChirpSlope_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setChirpSlope_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRangePulseDuration_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setRangePulseDuration_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRadarWavelength_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setRadarWavelength_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRangeFirstSample_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setRangeFirstSample_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setRangeSpectralWeighting_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setRangeSpectralWeighting_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setSpectralShiftFraction_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setSpectralShiftFraction_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setIMRC1_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setIMRC1_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setIMMocomp_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setIMMocomp_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setIMRCAS1_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setIMRCAS1_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setIMRCRM1_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setIMRCRM1_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setTransDat_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setTransDat_f(&var);
    return Py_BuildValue("i", 0);
}
PyObject * setIQFlip_C(PyObject* self, PyObject* args)
{
    char * var;
    Py_ssize_t varSize;
    if(!PyArg_ParseTuple(args, "s#", &var, &varSize))
    {
        return NULL;
    }
    int varInt = Py_SAFE_DOWNCAST(varSize, Py_ssize_t, int);
    setIQFlip_f(var,&varInt);
    return Py_BuildValue("i", 0);
}
PyObject * setDeskewFlag_C(PyObject* self, PyObject* args)
{
    char * var;
    Py_ssize_t varSize;
    if(!PyArg_ParseTuple(args, "s#", &var, &varSize))
    {
        return NULL;
    }
    int varInt = Py_SAFE_DOWNCAST(varSize, Py_ssize_t, int);
    setDeskewFlag_f(var,&varInt);
    return Py_BuildValue("i", 0);
}
PyObject * setSecondaryRangeMigrationFlag_C(PyObject* self, PyObject* args)
{
    char * var;
    Py_ssize_t varSize;
    if(!PyArg_ParseTuple(args, "s#", &var, &varSize))
    {
        return NULL;
    }
    int varInt = Py_SAFE_DOWNCAST(varSize, Py_ssize_t, int);
    setSecondaryRangeMigrationFlag_f(var,&varInt);
    return Py_BuildValue("i", 0);
}
PyObject * setPosition_C(PyObject* self, PyObject* args)
{
    PyObject * list;
    int dim1 = 0;
    int dim2 = 0;
    if(!PyArg_ParseTuple(args, "Oii", &list, &dim1, &dim2))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                ". Expecting a list type object" << endl;
        exit(1);
    }
    double *  vectorV = new double[dim1*dim2];
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * listEl = PyList_GetItem(list,i);
        if(!PyList_Check(listEl))
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Expecting a list type object" << endl;
            exit(1);
        }
        for(int j = 0; j < dim2; ++j)
        {
            PyObject * listElEl = PyList_GetItem(listEl,j);
            if(listElEl == NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot retrieve list element" << endl;
                exit(1);
            }
            vectorV[dim2*i + j] = (double) PyFloat_AsDouble(listElEl);
            if(PyErr_Occurred() != NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot convert Py Object to C " << endl;
                exit(1);
            }
        }
    }
    setPosition_f(vectorV, &dim1, &dim2);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}

PyObject * setVelocity_C(PyObject* self, PyObject* args)
{
    PyObject * list;
    int dim1 = 0;
    int dim2 = 0;
    if(!PyArg_ParseTuple(args, "Oii", &list, &dim1, &dim2))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                ". Expecting a list type object" << endl;
        exit(1);
    }
    double *  vectorV = new double[dim1*dim2];
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * listEl = PyList_GetItem(list,i);
        if(!PyList_Check(listEl))
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Expecting a list type object" << endl;
            exit(1);
        }
        for(int j = 0; j < dim2; ++j)
        {
            PyObject * listElEl = PyList_GetItem(listEl,j);
            if(listElEl == NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot retrieve list element" << endl;
                exit(1);
            }
            vectorV[dim2*i + j] = (double) PyFloat_AsDouble(listElEl);
            if(PyErr_Occurred() != NULL)
            {
                cout << "Error in file " << __FILE__ << " at line " <<
                        __LINE__ << ". Cannot convert Py Object to C " << endl;
                exit(1);
            }
        }
    }
    setVelocity_f(vectorV, &dim1, &dim2);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}

PyObject * setTime_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    PyObject * list;
    if(!PyArg_ParseTuple(args, "Oi", &list,&dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " <<
                __LINE__ << ". Expecting a list type object" << endl;
        exit(1);
    }
    double *  vectorV = new double[dim1];
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * listEl = PyList_GetItem(list,i);
        if(listEl == NULL)
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Cannot retrieve list element" << endl;
            exit(1);
        }
        vectorV[i] = (double) PyFloat_AsDouble(listEl);
        if(PyErr_Occurred() != NULL)
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setTime_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}

PyObject * setDopplerCentroidCoefficients_C(PyObject* self, PyObject* args)
{
    int dim1 = 0;
    PyObject * list;
    if(!PyArg_ParseTuple(args, "Oi", &list,&dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                ". Expecting a list type object" << endl;
        exit(1);
    }
    double *  vectorV = new double[dim1];
    for(int i = 0; i  < dim1; ++i)
    {
        PyObject * listEl = PyList_GetItem(list,i);
        if(listEl == NULL)
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Cannot retrieve list element" << endl;
            exit(1);
        }
        vectorV[i] = (double) PyFloat_AsDouble(listEl);
        if(PyErr_Occurred() != NULL)
        {
            cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                    ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setDopplerCentroidCoefficients_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}


PyObject *setPegPoint_C(PyObject *self, PyObject *args)
{
    double latitude;
    double longitude;
    double heading;
    if(!PyArg_ParseTuple(args,"ddd",&latitude,&longitude,&heading))
      {
        return NULL;
      }
    setPegPoint_f(&latitude,&longitude,&heading);
    return Py_BuildValue("i", 0);
}

PyObject *setPlanet_C(PyObject *self, PyObject *args)
{
    double a;
    double e2;
    if(!PyArg_ParseTuple(args,"dd",&a,&e2))
      {
        return NULL;
      }
    setPlanet_f(&a,&e2);
    return Py_BuildValue("i", 0);
}

PyObject *setEllipsoid_C(PyObject *self, PyObject *args)
{
    double spin;
    double gm;
    if(!PyArg_ParseTuple(args,"dd",&spin,&gm))
      {
        return NULL;
      }
    setEllipsoid_f(&spin,&gm);
    return Py_BuildValue("i", 0);
}

PyObject *setSlcWidth_C(PyObject *self, PyObject *args)
{
    int var;
    if(!PyArg_ParseTuple(args, "i", &var))
    {
        return NULL;
    }
    setSlcWidth_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject *setStartingRange_C(PyObject *self, PyObject *args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
    return NULL;
    }
    setStartingRange_f(&var);
    return Py_BuildValue("i", 0);
}

// ML 08-23-2013
PyObject *setShift_C(PyObject* self, PyObject* args)
{
    double var;
    if(!PyArg_ParseTuple(args, "d", &var))
    {
        return NULL;
    }
    setShift_f(&var);
    return Py_BuildValue("d", 0);
}
//ML

PyObject *setMinAmb_C(PyObject *self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args,"i",&var))
    {
    return NULL;
    }
    setMinAmb_f(&var);
    return Py_BuildValue("i",0);
}

PyObject *setMaxAmb_C(PyObject *self, PyObject* args)
{
    int var;
    if(!PyArg_ParseTuple(args,"i",&var))
    {
    return NULL;
    }
    setMaxAmb_f(&var);
    return Py_BuildValue("i",0);
}

PyObject *getEntropy_C(PyObject *self, PyObject *args)
{
    int size;
    if(!PyArg_ParseTuple(args,"i", &size))
    {
    return NULL;
    }
    double *vector = new double[size];
    getEntropy_f(vector, &size);

    PyObject *list = PyList_New(size);
    for(int i=0; i<size; i++)
    {
    PyObject *listEl = PyFloat_FromDouble((double) vector[i]);
    if(listEl == NULL)
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ <<
                ". Cannot set list element. \n";
        exit(1);
    }
    PyList_SetItem(list,i,listEl);
    }
    delete [] vector;
    return Py_BuildValue("N", list);
}

// end of file
