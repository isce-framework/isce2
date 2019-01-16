#ifndef dopiqmodule_h
#define dopiqmodule_h

#include <Python.h>
#include <stdint.h>
#include "dopiqmoduleFortTrans.h"

extern "C"
{
        void dopiq_f(uint64_t *);
        PyObject *dopiq_C(PyObject *self,PyObject *args);
        void setLineLength_f(int *);
        PyObject *setLineLength_C(PyObject *self,PyObject *args);
        void setLineHeaderLength_f(int *);
        PyObject *setLineHeaderLength_C(PyObject *self,PyObject *args);
        void setLastSample_f(int *);
        PyObject *setLastSample_C(PyObject *self,PyObject *args);
        void setStartLine_f(int *);
        PyObject *setStartLine_C(PyObject *self,PyObject *args);
        void setNumberOfLines_f(int *);
        PyObject *setNumberOfLines_C(PyObject *self,PyObject *args);
        void setMean_f(double *);
        PyObject *setMean_C(PyObject *self,PyObject *args);
        void setPRF_f(double *);
        PyObject *setPRF_C(PyObject *self,PyObject *args);
        void getAcc_f(double *,int *);
        PyObject *getDoppler_C(PyObject *self,PyObject *args);
        void allocate_acc_f(int *);
        PyObject *allocate_doppler_C(PyObject *self,PyObject *args);
        void deallocate_acc_f();
        PyObject *deallocate_doppler_C(PyObject *self,PyObject *args);
}

static PyMethodDef dopiq_methods[] =
{
                {"dopiq_Py",dopiq_C,METH_VARARGS," "},
                {"setLineLength_Py",setLineLength_C,METH_VARARGS," "},
                {"setLineHeaderLength_Py",setLineHeaderLength_C,METH_VARARGS," "},
                {"setLastSample_Py",setLastSample_C,METH_VARARGS," "},
                {"setStartLine_Py",setStartLine_C,METH_VARARGS," "},
                {"setNumberOfLines_Py",setNumberOfLines_C,METH_VARARGS," "},
                {"setMean_Py",setMean_C,METH_VARARGS," "},
                {"setPRF_Py",setPRF_C,METH_VARARGS," "},
                {"getDoppler_Py",getDoppler_C,METH_VARARGS," "},
                {"allocate_doppler_Py",allocate_doppler_C,METH_VARARGS," "},
                {"deallocate_doppler_Py",deallocate_doppler_C,METH_VARARGS," "},
                {NULL,NULL,0,NULL}
};

#endif //dopiqmodule_h
