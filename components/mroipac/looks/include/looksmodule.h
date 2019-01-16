//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                  (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef looksmodule_h
#define looksmodule_h

#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <Python.h>
#include <stdint.h>

extern "C"
{
        PyObject * looks_C(PyObject *, PyObject *);
}

static PyMethodDef looks_methods[] =
{
        {"looks_Py", looks_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //looksmodule_h
