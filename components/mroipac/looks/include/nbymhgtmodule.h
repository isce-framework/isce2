//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                  (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef nbymhgtmodule_h
#define nbymhgtmodule_h

#include <Python.h>
#include <stdint.h>

extern "C"
{
        PyObject * nbymhgt_C(PyObject *, PyObject *);

}

static PyMethodDef nbymhgt_methods[] =
{
        {"nbymhgt_Py", nbymhgt_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //nbymhgtmodule_h
