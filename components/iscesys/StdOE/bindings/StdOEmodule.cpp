//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "StdOE.h"
#include "StdOEmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for StdOE";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "StdOE",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    StdOE_methods,
};

// initialization function for the module
// *must* be called PyInit_formslc
PyMODINIT_FUNC
PyInit_StdOE()
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

PyObject * setStdErr_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string message = var;
        StdOE::setStdErr(message);
        return Py_BuildValue("i", 0);
}
PyObject * setStdErrFileTag_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string tag = var;
        StdOE::setStdErrFileTag(tag);
        return Py_BuildValue("i", 0);
}
PyObject * setStdOutFileTag_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string tag = var;
        StdOE::setStdOutFileTag(tag);
        return Py_BuildValue("i", 0);
}
PyObject * setStdLogFileTag_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string tag = var;
        StdOE::setStdLogFileTag(tag);
        return Py_BuildValue("i", 0);
}
PyObject * setStdErrFile_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string filename = var;
        StdOE::setStdErrFile(filename);
        return Py_BuildValue("i", 0);
}
PyObject * setStdOut_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string message = var;
        StdOE::setStdOut(message);
        return Py_BuildValue("i", 0);
}
PyObject * setStdLogFile_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string filename = var;
        StdOE::setStdLogFile(filename);
        return Py_BuildValue("i", 0);
}
PyObject * setStdOutFile_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string filename = var;
        StdOE::setStdOutFile(filename);
        return Py_BuildValue("i", 0);
}
PyObject * getStdOut_C(PyObject* self, PyObject* args)
{
        char var;
        var  = StdOE::getStdOut();
        return Py_BuildValue("c",var);
}
PyObject * getStdErr_C(PyObject* self, PyObject* args)
{
        char var;
        var  = StdOE::getStdErr();
        return Py_BuildValue("c",var);
}
PyObject * writeStd_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string message = var;
        StdOE::writeStd(var);
        return Py_BuildValue("i", 0);
}
PyObject * writeStdLog_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string message = var;
        StdOE::writeStdLog(var);
        return Py_BuildValue("i", 0);
}
PyObject * writeStdOut_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string message = var;
        StdOE::writeStdOut(var);
        return Py_BuildValue("i", 0);
}
PyObject * writeStdErr_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        if(!PyArg_ParseTuple(args, "s#", &var ,&varInt))
        {
                return NULL;
        }
        string message = var;
        StdOE::writeStdOut(var);
        return Py_BuildValue("i", 0);
}
PyObject * writeStdFile_C(PyObject* self, PyObject* args)
{
        char * var;
        Py_ssize_t varInt;
        char * var1;
        Py_ssize_t varInt1;
        if(!PyArg_ParseTuple(args, "s#s#", &var ,&varInt,&var1,&varInt1))
        {
                return NULL;
        }
        string filename = var;
        string message = var1;
        StdOE::writeStdFile(var,var1);
        return Py_BuildValue("i", 0);
}
