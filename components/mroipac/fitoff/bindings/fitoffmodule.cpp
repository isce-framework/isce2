#include <Python.h>
#include "fitoffmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for fitoff.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "fitoff",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    fitoff_methods,
};

// initialization function for the module
// *must* be called PyInit_formslc
PyMODINIT_FUNC
PyInit_fitoff()
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


PyObject * fitoff_C(PyObject * self, PyObject * args)
{
        fitoff_f();
        return Py_BuildValue("i", 0);

}
PyObject * setStdWriter_C(PyObject* self, PyObject* args)
{
    uint64_t var;
    if(!PyArg_ParseTuple(args, "K", &var))
    {
        return NULL;
    }
    setStdWriter_f(&var);
    return Py_BuildValue("i",0);
}
PyObject * setMinPoint_C(PyObject* self, PyObject* args)
{
        int varInt;
        if(!PyArg_ParseTuple(args, "i", &varInt))
        {
                return NULL;
        }
        setMinPoint_f(&varInt);
        return Py_BuildValue("i", 0);
}
PyObject * setNSig_C(PyObject* self, PyObject* args)
{
        double varDouble;
        if(!PyArg_ParseTuple(args, "d", &varDouble))
        {
                return NULL;
        }
        setNSig_f(&varDouble);
        return Py_BuildValue("i", 0);
}
PyObject * setMaxRms_C(PyObject* self, PyObject* args)
{
        double varDouble;
        if(!PyArg_ParseTuple(args, "d", &varDouble))
        {
                return NULL;
        }
        setMaxRms_f(&varDouble);
        return Py_BuildValue("i", 0);
}
PyObject * setNumberLines_C(PyObject* self, PyObject* args)
{
    int varInt;
    if(!PyArg_ParseTuple(args, "i", &varInt))
    {
        return NULL;
    }
    setNumberLines_f(&varInt);
    return Py_BuildValue("i", 0);
}
PyObject * setMaxIter_C(PyObject* self, PyObject* args)
{
    int varInt;
    if(!PyArg_ParseTuple(args, "i", &varInt))
    {
        return NULL;
    }
    setMaxIter_f(&varInt);
    return Py_BuildValue("i",0);
}
PyObject * setMinIter_C(PyObject* self, PyObject* args)
{
    int varInt;
    if(!PyArg_ParseTuple(args, "i", &varInt))
    {
        return NULL;
    }
    setMinIter_f(&varInt);
    return Py_BuildValue("i",0);
}

PyObject * setL1normFlag_C(PyObject* self, PyObject *args)
{
    int varInt;
    if(!PyArg_ParseTuple(args, "i", &varInt))
    {
        return NULL;
    }
    setL1normFlag_f(&varInt);
    return Py_BuildValue("i",0);
}
PyObject * setLocationAcross_C(PyObject* self, PyObject* args)
{
    int dim1=0;
    PyObject * list;
    if(!PyArg_ParseTuple(args,"Oi", &list, &dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
        exit(1);
    }
    double *vectorV = new double[dim1];
    for(int i=0; i<dim1; ++i)
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
            cout << "Error in file " << __FILE__ << " at line "<< __LINE__ << ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setLocationAcross_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}
PyObject * setLocationDown_C(PyObject* self, PyObject* args)
{
    int dim1=0;
    PyObject * list;
    if(!PyArg_ParseTuple(args,"Oi", &list, &dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
        exit(1);
    }
    double *vectorV = new double[dim1];
    for(int i=0; i<dim1; ++i)
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
            cout << "Error in file " << __FILE__ << " at line "<< __LINE__ << ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setLocationDown_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}
PyObject * setLocationDownOffset_C(PyObject* self, PyObject* args)
{
    int dim1=0;
    PyObject * list;
    if(!PyArg_ParseTuple(args,"Oi", &list, &dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
        exit(1);
    }
    double *vectorV = new double[dim1];
    for(int i=0; i<dim1; ++i)
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
            cout << "Error in file " << __FILE__ << " at line "<< __LINE__ << ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setLocationDownOffset_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}
PyObject * setLocationAcrossOffset_C(PyObject* self, PyObject* args)
{
    int dim1=0;
    PyObject * list;
    if(!PyArg_ParseTuple(args,"Oi", &list, &dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
        exit(1);
    }
    double *vectorV = new double[dim1];
    for(int i=0; i<dim1; ++i)
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
            cout << "Error in file " << __FILE__ << " at line "<< __LINE__ << ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setLocationAcrossOffset_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}
PyObject * setSNR_C(PyObject* self, PyObject* args)
{
    int dim1=0;
    PyObject * list;
    if(!PyArg_ParseTuple(args,"Oi", &list, &dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
        exit(1);
    }
    double *vectorV = new double[dim1];
    for(int i=0; i<dim1; ++i)
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
            cout << "Error in file " << __FILE__ << " at line "<< __LINE__ << ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setSNR_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}
PyObject * setCovDown_C(PyObject* self, PyObject* args)
{
    int dim1=0;
    PyObject * list;
    if(!PyArg_ParseTuple(args,"Oi", &list, &dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
        exit(1);
    }
    double *vectorV = new double[dim1];
    for(int i=0; i<dim1; ++i)
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
            cout << "Error in file " << __FILE__ << " at line "<< __LINE__ << ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setCovDown_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}

PyObject * setCovAcross_C(PyObject* self, PyObject* args)
{
    int dim1=0;
    PyObject * list;
    if(!PyArg_ParseTuple(args,"Oi", &list, &dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
        exit(1);
    }
    double *vectorV = new double[dim1];
    for(int i=0; i<dim1; ++i)
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
            cout << "Error in file " << __FILE__ << " at line "<< __LINE__ << ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setCovAcross_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);
}

PyObject * setCovCross_C(PyObject* self, PyObject* args)
{
    int dim1=0;
    PyObject * list;
    if(!PyArg_ParseTuple(args,"Oi", &list, &dim1))
    {
        return NULL;
    }
    if(!PyList_Check(list))
    {
        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
        exit(1);
    }
    double *vectorV = new double[dim1];
    for(int i=0; i<dim1; ++i)
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
            cout << "Error in file " << __FILE__ << " at line "<< __LINE__ << ". Cannot convert Py Object to C " << endl;
            exit(1);
        }
    }
    setCovCross_f(vectorV, &dim1);
    delete [] vectorV;
    return Py_BuildValue("i", 0);

}

PyObject * getAffineVector_C(PyObject* self, PyObject* args)
{
        int numElements = 6;
        vector<double> affineVec(numElements,0);

        getAffineVector_f(&affineVec[0]);
        PyObject * pyList = PyList_New(numElements);
        if(!pyList)
        {
            cout << "Error at line " << __LINE__ << " in file " << __FILE__ ". Exiting ..."<< endl;
            exit(1);
        }
        for(int i = 0; i < numElements; ++i)
        {
            PyList_SetItem(pyList,i, PyFloat_FromDouble(affineVec[i]));
        }
        return Py_BuildValue("O", pyList);

}

PyObject *getNumberOfRefinedOffsets_C(PyObject* self, PyObject* args)
{
    int numElements = 0;
    getNumberOfRefinedOffsets_f(&numElements);
    return Py_BuildValue("i", numElements);
}

PyObject *getRefinedOffsetField_C(PyObject* self, PyObject* args)
{
    int numElements = 0;
    int nValues = 8;

    if(!PyArg_ParseTuple(args, "i", &numElements))
    {
        return NULL;
    }


    double *acLoc = new double[numElements];
    double *dnLoc = new double[numElements];
    double *acOff = new double[numElements];
    double *dnOff = new double[numElements];
    double *snr   = new double[numElements];
    double *covAc = new double[numElements];
    double *covDn = new double[numElements];
    double *covX  = new double[numElements];

    getRefinedLocationAcross_f(acLoc);
    getRefinedLocationDown_f(dnLoc);
    getRefinedLocationAcrossOffset_f(acOff);
    getRefinedLocationDownOffset_f(dnOff);
    getRefinedSNR_f(snr);
    getRefinedCovAcross_f(covAc);
    getRefinedCovDown_f(covDn);
    getRefinedCovCross_f(covX);

    PyObject *pyList = PyList_New(numElements);
    if(!pyList)
    {
        cout << "Error at line " << __LINE__ << "in file " << __FILE__ ". Exiting ..." << endl;
        exit(1);
    }
    for(int i=0; i<numElements; i++)
    {
        PyObject *nList = PyList_New(nValues);
        if(!nList)
        {
            cout << "Error at line " << __LINE__ << " in file " << __FILE__ ". Exiting ..." << endl;
            exit(1);
        }

        PyList_SetItem(nList, 0, PyFloat_FromDouble(acLoc[i]));
        PyList_SetItem(nList, 1, PyFloat_FromDouble(dnLoc[i]));
        PyList_SetItem(nList, 2, PyFloat_FromDouble(acOff[i]));
        PyList_SetItem(nList, 3, PyFloat_FromDouble(dnOff[i]));
        PyList_SetItem(nList, 4, PyFloat_FromDouble(snr[i]));
        PyList_SetItem(nList, 5, PyFloat_FromDouble(covAc[i]));
        PyList_SetItem(nList, 6, PyFloat_FromDouble(covDn[i]));
        PyList_SetItem(nList, 7, PyFloat_FromDouble(covX[i]));

        PyList_SetItem(pyList, i , nList);
    }

    delete [] acLoc;
    delete [] dnLoc;
    delete [] acOff;
    delete [] dnOff;
    delete [] snr;
    delete [] covAc;
    delete [] covDn;
    delete [] covX;

    return Py_BuildValue("O", pyList);
}

PyObject * allocate_Arrays_C(PyObject* self, PyObject *args)
{
    allocate_LocationAcross_f();
    allocate_LocationDown_f();
    allocate_LocationAcrossOffset_f();
    allocate_LocationDownOffset_f();
    allocate_SNR_f();
    allocate_Covariance_f();
    return Py_BuildValue("i", 0);
}

PyObject * deallocate_Arrays_C(PyObject *self, PyObject *args)
{
    deallocate_LocationAcross_f();
    deallocate_LocationDown_f();
    deallocate_LocationAcrossOffset_f();
    deallocate_LocationDownOffset_f();
    deallocate_SNR_f();
    deallocate_Covariance_f();
    return Py_BuildValue("i", 0);
}
