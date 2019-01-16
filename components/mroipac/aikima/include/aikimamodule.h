#if !defined(__MROIPAC_AIKIMAMODULE_H__)
#define __MROIPAC_AIKIMAMODULE_H__

#include <Python.h>
#include "aikimamoduleFortTrans.h"

extern "C"
{
    //the fortran engine
    void aikima_f(void*, void*, int*, int*);
    PyObject* aikima_C(PyObject*, PyObject*);

    //fortran routines for setting the module variables
    void setWidth_f(int*);
    PyObject* setWidth_C(PyObject*, PyObject*);

    void setLength_f(int*);
    PyObject* setLength_C(PyObject*, PyObject*);

    void setFirstPixelAcross_f(int*);
    PyObject* setFirstPixelAcross_C(PyObject*, PyObject*);

    void setLastPixelAcross_f(int*);
    PyObject* setLastPixelAcross_C(PyObject*, PyObject*);

    void setFirstLineDown_f(int*);
    PyObject* setFirstLineDown_C(PyObject*, PyObject*);

    void setLastLineDown_f(int*);
    PyObject* setLastLineDown_C(PyObject*, PyObject*);

    void setBlockSize_f(int*);
    PyObject *setBlockSize_C(PyObject*, PyObject*);

    void setPadSize_f(int*);
    PyObject *setPadSize_C(PyObject*, PyObject*);

    void setNumberPtsPartial_f(int*);
    PyObject *setNumberPtsPartial_C(PyObject*, PyObject*);

    void setPrintFlag_f(int*);
    PyObject *setPrintFlag_C(PyObject*, PyObject*);

    void setThreshold_f(float*);
    PyObject *setThreshold_C(PyObject*, PyObject*);
}

//Method Table
static PyMethodDef aikima_methods[]=
{
    {"aikima_Py", aikima_C, METH_VARARGS, " "},
    {"setWidth_Py", setWidth_C, METH_VARARGS, " "},
    {"setLength_Py", setLength_C, METH_VARARGS, " "},
    {"setFirstPixelAcross_Py", setFirstPixelAcross_C, METH_VARARGS, " "},
    {"setLastPixelAcross_Py", setLastPixelAcross_C, METH_VARARGS, " "},
    {"setFirstLineDown_Py", setFirstLineDown_C, METH_VARARGS, " "},
    {"setLastLineDown_Py", setLastLineDown_C, METH_VARARGS, " "},
    {"setBlockSize_Py", setBlockSize_C, METH_VARARGS, " "},
    {"setPadSize_Py", setPadSize_C, METH_VARARGS, " "},
    {"setNumberPtsPartial_Py", setNumberPtsPartial_C, METH_VARARGS, " "},
    {"setPrintFlag_Py", setPrintFlag_C, METH_VARARGS, " "},
    {"setThreshold_Py", setThreshold_C, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};

#endif

//end of file
