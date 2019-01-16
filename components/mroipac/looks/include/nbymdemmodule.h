//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                  (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef nbymdemmodule_h
#define nbymdemmodule_h

#include <Python.h>
#include <stdint.h>

extern "C"
{
        PyObject * nbymdem_C(PyObject *, PyObject *);

}

static PyMethodDef nbymdem_methods[] =
{
        {"nbymdem_Py", nbymdem_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //nbymdemmodule_h
