#include <Python.h>
#include "dopplermodule.h"

using namespace std;

static const char * const __doc__ = "Python extension for doppler.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "doppler",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    doppler_methods,
};

// initialization function for the module
// *must* be called PyInit_formslc
PyMODINIT_FUNC
PyInit_doppler()
{
    // create the module using moduledef struct defined above
    PyObject * module = PyModule_Create(&moduledef);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return module;
    }
    // otherwise, we have an initialized module
    // and return the newly created module
    return module;
}


PyObject *doppler_C(PyObject *self,PyObject *args)
{
        uint64_t var0;
        if (!PyArg_ParseTuple(args,"K",&var0))
        {
                return NULL;
        }
        doppler_f(&var0);
        return Py_BuildValue("i",0);
}

PyObject *allocate_r_fd_C(PyObject *self, PyObject *args)
{
        int dim1;
        if (!PyArg_ParseTuple(args,"i",&dim1))
        {
                return NULL;
        }
        allocate_r_fd_f(&dim1);
        return Py_BuildValue("i",0);
}

PyObject *deallocate_r_fd_C(PyObject *self, PyObject *args)
{
        deallocate_r_fd_f();
        return Py_BuildValue("i",0);
}

PyObject *setLines_C(PyObject *self, PyObject *args)
{
        int var;
        if (!PyArg_ParseTuple(args,"i",&var))
        {
                return NULL;
        }
        setLines_f(&var);
        return Py_BuildValue("i",0);
}

PyObject *setSamples_C(PyObject *self, PyObject *args)
{
        int var;
        if (!PyArg_ParseTuple(args,"i",&var))
        {
                return NULL;
        }
        setSamples_f(&var);
        return Py_BuildValue("i",0);
}

PyObject *setStartLine_C(PyObject *self, PyObject *args)
{
        int var;
        if (!PyArg_ParseTuple(args,"i",&var))
        {
                return NULL;
        }
        setStartLine_f(&var);
        return Py_BuildValue("i",0);
}

PyObject *get_r_fd_C(PyObject *self,PyObject *args)
{
        int dim1;
        if (!PyArg_ParseTuple(args,"i",&dim1))
        {
                return NULL;
        }
        PyObject *list = PyList_New(dim1);
        double *vectorV = new double[dim1];
        get_r_fd_f(vectorV,&dim1);
        for(int i=0;i<dim1;i++)
        {
                PyObject *listEl = PyFloat_FromDouble((double) vectorV[i]);
                if (listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
            exit(1);
                }
                PyList_SetItem(list,i,listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list);
}
