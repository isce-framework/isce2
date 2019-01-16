#ifndef dopplermodule_h
#define dopplermodule_h

#include <Python.h>
#include <iostream>
#include "dopplermoduleFortTrans.h"

extern "C"
{
        void doppler_f(uint64_t *);
        PyObject *doppler_C(PyObject *self,PyObject *args);
        void setLines_f(int *);
        PyObject *setLines_C(PyObject *self,PyObject *args);
        void setStartLine_f(int *);
        PyObject *setStartLine_C(PyObject *self,PyObject *args);
        void setSamples_f(int *);
        PyObject *setSamples_C(PyObject *self,PyObject *args);
        void get_r_fd_f(double *,int *);
        PyObject *get_r_fd_C(PyObject *self,PyObject *args);
        void allocate_r_fd_f(int *);
        PyObject *allocate_r_fd_C(PyObject *self,PyObject *args);
        void deallocate_r_fd_f();
        PyObject *deallocate_r_fd_C(PyObject *self,PyObject *args);
}

static PyMethodDef doppler_methods[] =
{
                {"doppler_Py",doppler_C,METH_VARARGS," "},
                {"setLines_Py",setLines_C,METH_VARARGS," "},
                {"setSamples_Py",setSamples_C,METH_VARARGS," "},
                {"setStartLine_Py",setStartLine_C,METH_VARARGS," "},
                {"get_r_fd_Py",get_r_fd_C,METH_VARARGS," "},
                {"allocate_r_fd_Py",allocate_r_fd_C,METH_VARARGS," "},
                {"deallocate_r_fd_Py",deallocate_r_fd_C,METH_VARARGS," "},
                {NULL,NULL,0,NULL}
};

#endif //dopplermodule_h
