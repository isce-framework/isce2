/*#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# Author: Piyush Agram
# Copyright 2014, by the California Institute of Technology. ALL RIGHTS RESERVED.
# United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer at
# the California Institute of Technology.
# This software may be subject to U.S. export control laws.
# By accepting this software, the user agrees to comply with all applicable U.S.
# export laws and regulations. User has the responsibility to obtain export licenses,
# or other export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <Python.h>
#include "watermaskmodule.h"
using namespace std;

static const char * const __doc__ = "module for watermask.f";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "watermask",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    watermask_methods,
};

// initialization function for the module
// *must* be called PyInit_watermask
PyMODINIT_FUNC
PyInit_watermask()
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

PyObject* watermask_C(PyObject* self, PyObject* args)
{
    double lat0, lon0;
    double dlat, dlon;
    double x,y;
    int nx, ny;
    char *outname;

    PyObject *shapeList;
    PyObject *poly;
    PyObject *point;
    int nshape, npoly;
    if(!PyArg_ParseTuple(args, "Oddddiis", &shapeList, &lon0, &lat0, &dlon,
        &dlat,&nx,&ny,&outname))
    {
        return NULL;
    }


    nshape = PyList_Size(shapeList);

    //Create waterbody object
    WaterBody waterInfo(nshape);
    waterInfo.setTopLeft(lon0, lat0);
    waterInfo.setSpacing(dlon, dlat);
    waterInfo.setDimensions(nx, ny);

//    printf("Top Left: %f , %f \n", lon0, lat0);
//    printf("Spacing: %f, %f \n", dlon, dlat);
    printf("Number of polygons : %d \n", nshape);

    for(int i=0;i<nshape;i++)
    {
        //Get size of polygon and allocate memory
        poly = PyList_GetItem(shapeList, i);
        npoly = PyList_Size(poly);
        printf("Polygon: %d , Num points: %d \n", i, npoly);
        waterInfo.allocate(i, npoly);

        for(int j=0;j<npoly;j++)
        {
            point = PyList_GetItem(poly, j);
            x = PyFloat_AsDouble(PyList_GetItem(point, 0));
            y = PyFloat_AsDouble(PyList_GetItem(point, 1));
            waterInfo.setShapeData(i, j, x, y);
        }
//        waterInfo.printShape(i);
    }


    waterInfo.fillGrid(outname);

    return Py_BuildValue("i", 0);
}


PyObject* watermaskxy_C(PyObject* self, PyObject* args)
{
    double x,y;
    int nx, ny;
    char *outname;
    char *latname;
    char *lonname;

    PyObject *shapeList;
    PyObject *poly;
    PyObject *point;
    int nshape, npoly;
    if(!PyArg_ParseTuple(args, "Oiisss", &shapeList, &nx, &ny, &lonname,
        &latname,&outname))
    {
        return NULL;
    }


    nshape = PyList_Size(shapeList);

    //Create waterbody object
    WaterBody waterInfo(nshape);
    waterInfo.setDimensions(nx, ny);

    printf("Number of polygons : %d \n", nshape);

    for(int i=0;i<nshape;i++)
    {
        //Get size of polygon and allocate memory
        poly = PyList_GetItem(shapeList, i);
        npoly = PyList_Size(poly);
        printf("Polygon: %d , Num points: %d \n", i, npoly);
        waterInfo.allocate(i, npoly);

        for(int j=0;j<npoly;j++)
        {
            point = PyList_GetItem(poly, j);
            x = PyFloat_AsDouble(PyList_GetItem(point, 0));
            y = PyFloat_AsDouble(PyList_GetItem(point, 1));
            waterInfo.setShapeData(i, j, x, y);
        }
//        waterInfo.printShape(i);
    }


    waterInfo.makemask(lonname, latname, outname);

    return Py_BuildValue("i", 0);
}

// end of file
