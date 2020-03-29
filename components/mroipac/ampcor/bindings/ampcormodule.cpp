#include <Python.h>
#include "ampcormodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <fstream>
using namespace std;

static const char * const __doc__ = "Python extension for ampcor.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "ampcor",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    ampcor_methods,
};

// initialization function for the module
PyMODINIT_FUNC
PyInit_ampcor()
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


//set state variable methods

PyObject* setImageDatatype1_C(PyObject* self, PyObject* args)
{
  char* type;
  int len;
  if( !PyArg_ParseTuple(args,"s#",&type,&len) )
  {
    return NULL;
  }
  setImageDatatype1_f(type,len);
  return Py_BuildValue("i",0);
}

PyObject* setLineLength1_C(PyObject* self, PyObject* args)
{
  int samples;
  if( !PyArg_ParseTuple(args,"i",&samples) )
  {
    return NULL;
  }
  setLineLength1_f(&samples);
  return Py_BuildValue("i",0);
}

PyObject* setImageLength1_C(PyObject* self, PyObject* args)
{
  int samples;
  if( !PyArg_ParseTuple(args,"i",&samples) )
  {
    return NULL;
  }
  setImageLength1_f(&samples);
  return Py_BuildValue("i",0);
}


PyObject* setImageDatatype2_C(PyObject* self, PyObject* args)
{
  char* type;
  int len;
  if( !PyArg_ParseTuple(args,"s#",&type,&len) )
  {
    return NULL;
  }
  setImageDatatype2_f(type,len);
  return Py_BuildValue("i",0);
}

PyObject* setLineLength2_C(PyObject* self, PyObject* args)
{
  int samples;
  if( !PyArg_ParseTuple(args,"i",&samples) )
  {
    return NULL;
  }
  setLineLength2_f(&samples);
  return Py_BuildValue("i",0);
}

PyObject* setImageLength2_C(PyObject* self, PyObject* args)
{
  int samples;
  if( !PyArg_ParseTuple(args,"i",&samples) )
  {
    return NULL;
  }
  setImageLength2_f(&samples);
  return Py_BuildValue("i",0);
}

PyObject* setFirstSampleDown_C(PyObject* self, PyObject* args)
{
  int line;
  if( !PyArg_ParseTuple(args,"i",&line) )
  {
    return NULL;
  }
  setFirstSampleDown_f(&line);
  return Py_BuildValue("i",0);
}

PyObject* setLastSampleDown_C(PyObject* self, PyObject* args)
{
  int line;
  if( !PyArg_ParseTuple(args,"i",&line) )
  {
    return NULL;
  }
  setLastSampleDown_f(&line);
  return Py_BuildValue("i",0);
}

PyObject* setSkipSampleDown_C(PyObject* self, PyObject* args)
{
  int line;
  if( !PyArg_ParseTuple(args,"i",&line))
  {
    return NULL;
  }
  setSkipSampleDown_f(&line);
  return Py_BuildValue("i",0);
}

PyObject* setFirstSampleAcross_C(PyObject* self, PyObject* args)
{
  int sample;
  if( !PyArg_ParseTuple(args,"i",&sample) )
  {
    return NULL;
  }
  setFirstSampleAcross_f(&sample);
  return Py_BuildValue("i",0);
}

PyObject* setLastSampleAcross_C(PyObject* self, PyObject* args)
{
  int sample;
  if( !PyArg_ParseTuple(args,"i",&sample))
  {
    return NULL;
  }
  setLastSampleAcross_f(&sample);
  return Py_BuildValue("i",0);
}

PyObject* setSkipSampleAcross_C(PyObject* self, PyObject* args)
{
  int sample;
  if( !PyArg_ParseTuple(args,"i",&sample) )
  {
    return NULL;
  }
  setSkipSampleAcross_f(&sample);
  return Py_BuildValue("i",0);
}

PyObject* setWindowSizeWidth_C(PyObject* self, PyObject* args)
{
  int width;
  if( !PyArg_ParseTuple(args,"i",&width) )
  {
    return NULL;
  }
  setWindowSizeWidth_f(&width);
  return Py_BuildValue("i",0);
}

PyObject* setWindowSizeHeight_C(PyObject* self, PyObject* args)
{
  int height;
  if( !PyArg_ParseTuple(args,"i",&height) )
  {
    return NULL;
  }
  setWindowSizeHeight_f(&height);
  return Py_BuildValue("i",0);
}

PyObject* setSearchWindowSizeWidth_C(PyObject* self, PyObject* args)
{
  int width;
  if( !PyArg_ParseTuple(args,"i",&width) )
  {
    return NULL;
  }
  setSearchWindowSizeWidth_f(&width);
  return Py_BuildValue("i",0);
}

PyObject* setSearchWindowSizeHeight_C(PyObject* self, PyObject* args)
{
  int height;
  if( !PyArg_ParseTuple(args,"i",&height) )
  {
    return NULL;
  }
  setSearchWindowSizeHeight_f(&height);
  return Py_BuildValue("i",0);
}

PyObject* setAcrossLooks_C(PyObject* self, PyObject* args)
{
  int width;
  if( !PyArg_ParseTuple(args,"i",&width) )
  {
    return NULL;
  }
  setAcrossLooks_f(&width);
  return Py_BuildValue("i",0);
}

PyObject* setDownLooks_C(PyObject* self, PyObject* args)
{
  int height;
  if( !PyArg_ParseTuple(args,"i",&height) )
  {
    return NULL;
  }
  setDownLooks_f(&height);
  return Py_BuildValue("i",0);
}

PyObject* setOversamplingFactor_C(PyObject* self, PyObject* args)
{
  int factor;
  if( !PyArg_ParseTuple(args,"i",&factor) )
  {
    return NULL;
  }
  setOversamplingFactor_f(&factor);
  return Py_BuildValue("i",0);
}

PyObject* setZoomWindowSize_C(PyObject* self, PyObject* args)
{
  int size;
  if( !PyArg_ParseTuple(args,"i",&size) )
  {
    return NULL;
  }
  setZoomWindowSize_f(&size);
  return Py_BuildValue("i",0);
}

PyObject* setAcrossGrossOffset_C(PyObject *self, PyObject* args)
{
    int var;
    if (!PyArg_ParseTuple(args,"i",&var))
    {
        return NULL;
    }
    setAcrossGrossOffset_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject* setDownGrossOffset_C(PyObject *self, PyObject* args)
{
    int var;
    if (!PyArg_ParseTuple(args,"i",&var))
    {
        return NULL;
    }
    setDownGrossOffset_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject* setThresholdSNR_C(PyObject *self, PyObject* args)
{
    double var;
    if (!PyArg_ParseTuple(args,"d",&var))
    {
        return NULL;
    }
    setThresholdSNR_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject* setThresholdCov_C(PyObject *self, PyObject* args)
{
    double var;
    if (!PyArg_ParseTuple(args,"d",&var))
    {
        return NULL;
    }
    setThresholdCov_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject* setDebugFlag_C(PyObject *self, PyObject* args)
{
    PyObject *obj;
    int var;
    if (!PyArg_ParseTuple(args,"O",&obj))
    {
        return NULL;
    }
    if (obj == Py_True) var= 1;
    else                var = 0;

    setDebugFlag_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject* setDisplayFlag_C(PyObject *self, PyObject* args)
{
    PyObject *obj;
    int var;
    if (!PyArg_ParseTuple(args,"O",&obj))
    {
        return NULL;
    }
    if (obj==Py_True) var = 1;
    else              var = 0;

    setDisplayFlag_f(&var);
    return Py_BuildValue("i", 0);
}

PyObject* setScaleFactorX_C(PyObject *self, PyObject* args)
{
    float var;
    if (!PyArg_ParseTuple(args,"f",&var))
    {
        return NULL;
    }

    setScaleFactorX_f(&var);
    return Py_BuildValue("i",0);
}

PyObject* setScaleFactorY_C(PyObject *self, PyObject *args)
{
    float var;
    if(!PyArg_ParseTuple(args,"f",&var))
    {
        return NULL;
    }
    setScaleFactorY_f(&var);
    return Py_BuildValue("i",0);
}

//print state method
PyObject* ampcorPrintState_C(PyObject* self, PyObject* args)
{
  ampcorPrintState_f();
  return Py_BuildValue("i",0);
}


//Allocate Deallocate methods
PyObject * allocate_locationAcross_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_locationAcross_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_locationAcross_C(PyObject* self, PyObject* args)
{
        deallocate_locationAcross_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_locationAcrossOffset_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_locationAcrossOffset_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_locationAcrossOffset_C(PyObject* self, PyObject* args)
{
        deallocate_locationAcrossOffset_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_locationDown_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_locationDown_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_locationDown_C(PyObject* self, PyObject* args)
{
        deallocate_locationDown_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_locationDownOffset_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_locationDownOffset_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_locationDownOffset_C(PyObject* self, PyObject* args)
{
        deallocate_locationDownOffset_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_snrRet_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_snrRet_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_snrRet_C(PyObject* self, PyObject* args)
{
        deallocate_snrRet_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_cov1Ret_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_cov1Ret_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_cov1Ret_C(PyObject* self, PyObject* args)
{
        deallocate_cov1Ret_f();
        return Py_BuildValue("i", 0);
}


PyObject * allocate_cov2Ret_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_cov2Ret_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_cov2Ret_C(PyObject* self, PyObject* args)
{
        deallocate_cov2Ret_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_cov3Ret_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_cov3Ret_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_cov3Ret_C(PyObject* self, PyObject* args)
{
        deallocate_cov3Ret_f();
        return Py_BuildValue("i", 0);
}

//Actual driver routine
PyObject * ampcor_C(PyObject* self, PyObject* args)
{
        uint64_t var0;
        uint64_t var1;
        int bnd1;
        int bnd2;
        if(!PyArg_ParseTuple(args, "KKii",&var0,&var1,&bnd1,&bnd2))
        {
                return NULL;
        }
        bnd1 = bnd1 + 1;   //Change band number from C to Fortran
        bnd2 = bnd2 + 1;
        ampcor_f(&var0,&var1,&bnd1,&bnd2);
        return Py_BuildValue("i", 0);
}


//get state variable methods
PyObject* getNumRows_C(PyObject *self, PyObject* args)
{
    int var;
    getNumRows_f(&var);
    return Py_BuildValue("i", var);
}

PyObject * getLocationAcross_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        int *  vectorV = new int[dim1];
        getLocationAcross_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyLong_FromLong((long int) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("N",list);
}

PyObject * getLocationAcrossOffset_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        float *  vectorV = new float[dim1];
        getLocationAcrossOffset_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("N",list);
}

PyObject * getLocationDown_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        int *  vectorV = new int[dim1];
        getLocationDown_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyLong_FromLong((long int) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("N",list);
}

PyObject * getLocationDownOffset_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        float *  vectorV = new float[dim1];
        getLocationDownOffset_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("N",list);
}

PyObject * getSNR_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        float *  vectorV = new float[dim1];
        getSNR_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("N",list);
}

PyObject * getCov1_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        float *  vectorV = new float[dim1];
        getCov1_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("N",list);
}

PyObject * getCov2_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        float *  vectorV = new float[dim1];
        getCov2_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("N",list);
}

PyObject * getCov3_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        float *  vectorV = new float[dim1];
        getCov3_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("N",list);
}



PyObject* setWinsizeFilt_C(PyObject* self, PyObject* args)
{
  int factor;
  if( !PyArg_ParseTuple(args,"i",&factor) )
  {
    return NULL;
  }
  setWinsizeFilt_f(&factor);
  return Py_BuildValue("i",0);
}

PyObject* setOversamplingFactorFilt_C(PyObject* self, PyObject* args)
{
  int factor;
  if( !PyArg_ParseTuple(args,"i",&factor) )
  {
    return NULL;
  }
  setOversamplingFactorFilt_f(&factor);
  return Py_BuildValue("i",0);
}
