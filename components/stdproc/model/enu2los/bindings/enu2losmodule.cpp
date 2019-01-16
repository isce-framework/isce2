//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//# Author: Piyush Agram
//# Copyright 2013, by the California Institute of Technology. ALL RIGHTS RESERVED.
//# United States Government Sponsorship acknowledged.
//# Any commercial use must be negotiated with the Office of Technology Transfer at
//# the California Institute of Technology.
//# This software may be subject to U.S. export control laws.
//# By accepting this software, the user agrees to comply with all applicable U.S.
//# export laws and regulations. User has the responsibility to obtain export licenses,
//# or other export authority as may be required before exporting such information to
//# foreign countries or providing access to foreign persons.
//#
//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <Python.h>
#include "DataAccessor.h"
#include "enu2losmodule.h"
#include "enu2los.h"
#include <iostream>
#include <string>
#include <stdint.h>
using namespace std;

static const char * const __doc__ = "Python extension for enu2los";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "enu2los",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    enu2los_methods,
};

// initialization function for the module
PyMODINIT_FUNC
PyInit_enu2los()
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


PyObject * createENU2LOS_C(PyObject* self, PyObject *args)
{
    enu2los* ptr = new enu2los;
    return Py_BuildValue("K", (uint64_t) ptr);
}

PyObject * destroyENU2LOS_C(PyObject* self, PyObject *args)
{
    uint64_t ptr;
    if(!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }

    if(((enu2los*)(ptr)) != NULL)
    {
        delete ((enu2los*)(ptr));
    }
    return Py_BuildValue("i", 0);
}


PyObject * setDims_C(PyObject * self, PyObject *args)
{
    uint64_t ptr = 0;
    int wid, len;
    if(!PyArg_ParseTuple(args, "Kii", &ptr, &wid, &len))
    {
        return NULL;
    }

    ((enu2los*)(ptr))->setDims(wid,len);
    return Py_BuildValue("i", 0);
}

PyObject * setGeoDims_C(PyObject * self, PyObject * args)
{
    uint64_t ptr=0;
    int wid, len;
    if(!PyArg_ParseTuple(args,"Kii", &ptr, &wid, &len))
    {
        return NULL;
    }

    ((enu2los*)(ptr))->setGeoDims(wid,len);
    return Py_BuildValue("i",0);
}

PyObject * setWavelength_C(PyObject *self, PyObject* args)
{
    uint64_t ptr=0;
    float wvl=0.0;
    if(!PyArg_ParseTuple(args,"Kf", &ptr, &wvl))
    {
        return NULL;
    }

    ((enu2los*)(ptr))->setWavelength(wvl);
    return Py_BuildValue("i",0);
}

PyObject * setScaleFactor_C(PyObject *self, PyObject* args)
{
      uint64_t ptr=0;
      float scl=0.0;
      if(!PyArg_ParseTuple(args,"Kf", &ptr, &scl))
      {
          return NULL;
      }

      ((enu2los*)(ptr))->setScaleFactor(scl);
      return Py_BuildValue("i",0);
}

PyObject * setLatitudeInfo_C(PyObject *self, PyObject* args)
{
      uint64_t ptr=0;
      float fLat,dLat;
      if(!PyArg_ParseTuple(args,"Kff", &ptr, &fLat,&dLat))
      {
          return NULL;
      }

      ((enu2los*)(ptr))->setLatitudeInfo(fLat,dLat);
      return Py_BuildValue("i",0);
}

PyObject * setLongitudeInfo_C(PyObject* self, PyObject* args)
{
    uint64_t ptr=0;
    float fLon, dLon;
    if(!PyArg_ParseTuple(args,"Kff", &ptr, &fLon, &dLon))
    {
        return NULL;
    }

    ((enu2los*)(ptr))->setLongitudeInfo(fLon, dLon);
    return Py_BuildValue("i",0);
}

PyObject* enu2los_C(PyObject* self, PyObject* args)
{
    uint64_t ptr=0;
    uint64_t model=0;
    uint64_t lat=0;
    uint64_t lon=0;
    uint64_t los=0;
    uint64_t out=0;

    if(!PyArg_ParseTuple(args,"KKKKKK", &ptr, &model, &lat, &lon, &los, &out))
    {
        return NULL;
    }

    ((enu2los*)(ptr))->process(model,lat,lon,los,out);
    return Py_BuildValue("i",0);
}
