//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                  (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef cpxlooksmodule_h
#define cpxlooksmodule_h

#include <Python.h>
#include <stdint.h>

extern "C"
{
        PyObject * cpxlooks_C(PyObject *, PyObject *);
}

static PyMethodDef cpxlooks_methods[] =
{
        {"cpxlooks_Py", cpxlooks_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //cpxlooksmodule_h
