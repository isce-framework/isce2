//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef StdOEmodule_h
#define StdOEmodule_h

#include <Python.h>
#include <stdint.h>

extern "C"
{
        PyObject * setStdErr_C(PyObject *, PyObject *);
        PyObject * setStdErrFileTag_C(PyObject *, PyObject *);
        PyObject * setStdOutFileTag_C(PyObject *, PyObject *);
        PyObject * setStdLogFileTag_C(PyObject *, PyObject *);
        PyObject * setStdErrFile_C(PyObject *, PyObject *);
        PyObject * setStdLogFile_C(PyObject *, PyObject *);
        PyObject * setStdOutFile_C(PyObject *, PyObject *);
        PyObject * setStdOut_C(PyObject *, PyObject *);
        PyObject * getStdOut_C(PyObject *, PyObject *);
        PyObject * getStdErr_C(PyObject *, PyObject *);
        PyObject * writeStd_C(PyObject *, PyObject *);
        PyObject * writeStdOut_C(PyObject *, PyObject *);
        PyObject * writeStdLog_C(PyObject *, PyObject *);
        PyObject * writeStdErr_C(PyObject *, PyObject *);
        PyObject * writeStdFile_C(PyObject *, PyObject *);

}

static PyMethodDef StdOE_methods[] =
{
        {"setStdErr_Py", setStdErr_C, METH_VARARGS, " "},
        {"setStdErrFileTag_Py", setStdErrFileTag_C, METH_VARARGS, " "},
        {"setStdOutFileTag_Py", setStdOutFileTag_C, METH_VARARGS, " "},
        {"setStdLogFileTag_Py", setStdLogFileTag_C, METH_VARARGS, " "},
        {"setStdErrFile_Py", setStdErrFile_C, METH_VARARGS, " "},
        {"setStdOutFile_Py", setStdOutFile_C, METH_VARARGS, " "},
        {"setStdLogFile_Py", setStdLogFile_C, METH_VARARGS, " "},
        {"setStdOut_Py", setStdOut_C, METH_VARARGS, " "},
        {"getStdOut_Py", getStdOut_C, METH_VARARGS, " "},
        {"getStdErr_Py", getStdErr_C, METH_VARARGS, " "},
        {"writeStd_Py", writeStd_C, METH_VARARGS, " "},
        {"writeStdOut_Py", writeStdOut_C, METH_VARARGS, " "},
        {"writeStdLog_Py", writeStdLog_C, METH_VARARGS, " "},
        {"writeStdErr_Py", writeStdErr_C, METH_VARARGS, " "},
        {"writeStdFile_Py", writeStdFile_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //StdOEmodule_h
