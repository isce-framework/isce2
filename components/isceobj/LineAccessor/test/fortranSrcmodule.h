//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef fortranSrcmodule_h
#define fortranSrcmodule_h

#include <Python.h>
#include "fortranSrcmoduleFortTrans.h"
#include <stdint.h>

extern "C"
{

        //name of the C function called when fortranSrc.testImageSetGet() is invoked in python.
        PyObject * testImageSetGet_C(PyObject *, PyObject *);


        //name used form C++ to call the testImageSegGet subroutine in fortran
        void testImageSetGet_f(uint64_t *, uint64_t *, int *);
}


static char * moduleDoc = "test module to interface pyhton with the LineAccessor c++ class.";

static PyMethodDef fortranSrc_methods[] =
{
        // when the python call fortranSrc.testImageSetGet() is made, the funtion testImageSetGet_C() is invoked
        {"testImageSetGet", testImageSetGet_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //fortranSrcmodule_h
