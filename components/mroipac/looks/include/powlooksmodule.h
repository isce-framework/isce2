//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                  (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef powlooksmodule_h
#define powlooksmodule_h

#include <Python.h>
#include <stdint.h>

extern "C"
{
        PyObject * powlooks_C(PyObject *, PyObject *);

}

static PyMethodDef powlooks_methods[] =
{
        {"powlooks_Py", powlooks_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //powlooksmodule_h
