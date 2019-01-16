//#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//#
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

#ifndef addsubmodelmodule_h
#define addsubmodelmodule_h

#include <Python.h>

extern "C"
{
    PyObject* createaddsubmodel_C(PyObject *, PyObject *);
    PyObject* destroyaddsubmodel_C(PyObject *, PyObject *);
    PyObject* cpxCpxProcess_C(PyObject *, PyObject *);
    PyObject* unwUnwProcess_C(PyObject *, PyObject *);
    PyObject* cpxUnwProcess_C(PyObject *, PyObject *);
    PyObject* setDims_C(PyObject *, PyObject *);
    PyObject* setScaleFactor_C(PyObject *, PyObject *);
    PyObject* setFlip_C(PyObject *, PyObject *);
}

static PyMethodDef addsubmodel_methods[] =
{
    {"createaddsubmodel", createaddsubmodel_C, METH_VARARGS, " "},
    {"destroyaddsubmodel", destroyaddsubmodel_C, METH_VARARGS, " "},
    {"cpxCpxProcess", cpxCpxProcess_C, METH_VARARGS, " "},
    {"cpxUnwProcess", cpxUnwProcess_C, METH_VARARGS, " "},
    {"unwUnwProcess", unwUnwProcess_C, METH_VARARGS, " "},
    {"setDims", setDims_C, METH_VARARGS, " "},
    {"setFlip", setFlip_C, METH_VARARGS, " "},
    {"setScaleFactor", setScaleFactor_C, METH_VARARGS, " "},
    {NULL, NULL, 0 , NULL}
};
#endif //addsubmodelmodule_h
