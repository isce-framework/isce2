#ifndef combinedlibmodule_h
#define combinedlibmodule_h

#include <Python.h>
#include <stdint.h>


extern "C"
{
    #include "geometry.h"
    #include "poly1d.h"
    #include "poly2d.h"
    #include "orbit.h"

    PyObject* exportOrbitToC(PyObject*, PyObject*);
    PyObject* exportPegToC(PyObject*, PyObject*);
    PyObject* exportPoly1DToC(PyObject*, PyObject*);
    PyObject* exportPoly2DToC(PyObject*, PyObject*);
    PyObject* exportEllipsoidToC(PyObject*, PyObject*);
    PyObject* importOrbitFromC(PyObject*, PyObject*);
    PyObject* importPegFromC(PyObject*, PyObject*);
    PyObject* importPoly1DFromC(PyObject*, PyObject*);
    PyObject* importPoly2DFromC(PyObject*, PyObject*);
    PyObject *freeCPoly1D(PyObject*, PyObject*);
    PyObject *freeCOrbit(PyObject*, PyObject*);
    PyObject *freeCPoly2D(PyObject*, PyObject*);
    PyObject *freeCPeg(PyObject*, PyObject*);
    PyObject *freeCEllipsoid(PyObject*, PyObject*);
}


static PyMethodDef combinedlib_methods[] = 
{
    {"exportOrbitToC", exportOrbitToC, METH_VARARGS, " "},
    {"exportPegToC", exportPegToC, METH_VARARGS, " "},
    {"exportPoly1DToC", exportPoly1DToC, METH_VARARGS, " "},
    {"exportPoly2DToC", exportPoly2DToC, METH_VARARGS, " "},
    {"exportEllipsoidToC", exportEllipsoidToC, METH_VARARGS, " "},
    {"importOrbitFromC", importOrbitFromC, METH_VARARGS, " "},
    {"importPegFromC", importPegFromC, METH_VARARGS, " "},
    {"importPoly1DFromC", importPoly1DFromC, METH_VARARGS, " "},
    {"importPoly2DFromC", importPoly2DFromC, METH_VARARGS, " "},
    {"freeCOrbit", freeCOrbit, METH_VARARGS, " "},
    {"freeCPoly1D", freeCPoly1D, METH_VARARGS, " "},
    {"freeCPoly2D", freeCPoly2D, METH_VARARGS, " "},
    {"freeCPeg", freeCPeg, METH_VARARGS, " "},
    {"freeCEllipsoid", freeCEllipsoid, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};

#endif
