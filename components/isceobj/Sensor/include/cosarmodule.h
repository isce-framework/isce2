#ifndef cosarmodule_h
#define cosarmodule_h

#include <Python.h>
#include "Cosar.hh"

extern "C"
{
        PyObject *cosar_C(PyObject *self,PyObject *args);
}

static PyMethodDef cosar_methods[]  =
{
                {"cosar_Py",cosar_C,METH_VARARGS," "},
                {NULL,NULL,0,NULL}
};

#endif //cosarmodule_h
