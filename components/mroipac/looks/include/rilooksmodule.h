//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                  (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef rilooksmodule_h
#define rilooksmodule_h

#include <Python.h>
#include <stdint.h>

extern "C"
{
        PyObject * rilooks_C(PyObject *, PyObject *);

}

static PyMethodDef rilooks_methods[] =
{
        {"rilooks_Py", rilooks_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //rilooksmodule_h
