//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#include <Python.h>
#include "get_peg_infomodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

static const char * const __doc__ = "Python extension for get_peg_info.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "get_peg_info",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    get_peg_info_methods,
};

// initialization function for the module
// *must* be called PyInit_formslc
PyMODINIT_FUNC
PyInit_get_peg_info()
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


PyObject * allocate_r_time_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_r_time_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_time_C(PyObject* self, PyObject* args)
{
        deallocate_r_time_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_xyz1_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "ii", &dim1, &dim2))
        {
                return NULL;
        }
        allocate_r_xyz1_f(&dim1, &dim2);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_xyz1_C(PyObject* self, PyObject* args)
{
        deallocate_r_xyz1_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_vxyz1_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "ii", &dim1, &dim2))
        {
                return NULL;
        }
        allocate_r_vxyz1_f(&dim1, &dim2);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_vxyz1_C(PyObject* self, PyObject* args)
{
        deallocate_r_vxyz1_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_axyz1_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "ii", &dim1, &dim2))
        {
                return NULL;
        }
        allocate_r_axyz1_f(&dim1, &dim2);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_axyz1_C(PyObject* self, PyObject* args)
{
        deallocate_r_axyz1_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_af_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_r_af_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_af_C(PyObject* self, PyObject* args)
{
        deallocate_r_af_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_cf_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_r_cf_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_cf_C(PyObject* self, PyObject* args)
{
        deallocate_r_cf_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_afdot_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_r_afdot_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_afdot_C(PyObject* self, PyObject* args)
{
        deallocate_r_afdot_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_cfdot_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_r_cfdot_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_cfdot_C(PyObject* self, PyObject* args)
{
        deallocate_r_cfdot_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_sfdot_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_r_sfdot_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_sfdot_C(PyObject* self, PyObject* args)
{
        deallocate_r_sfdot_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_transVect_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_r_transVect_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_transVect_C(PyObject* self, PyObject* args)
{
        deallocate_r_transVect_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_transfMat_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "ii", &dim1, &dim2))
        {
                return NULL;
        }
        allocate_r_transfMat_f(&dim1, &dim2);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_transfMat_C(PyObject* self, PyObject* args)
{
        deallocate_r_transfMat_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_intPos_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "ii", &dim1, &dim2))
        {
                return NULL;
        }
        allocate_r_intPos_f(&dim1, &dim2);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_intPos_C(PyObject* self, PyObject* args)
{
        deallocate_r_intPos_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_intVel_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "ii", &dim1, &dim2))
        {
                return NULL;
        }
        allocate_r_intVel_f(&dim1, &dim2);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_intVel_C(PyObject* self, PyObject* args)
{
        deallocate_r_intVel_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_vxyzpeg_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_r_vxyzpeg_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_vxyzpeg_C(PyObject* self, PyObject* args)
{
        deallocate_r_vxyzpeg_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_platvel_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_r_platvel_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_platvel_C(PyObject* self, PyObject* args)
{
        deallocate_r_platvel_f();
        return Py_BuildValue("i", 0);
}

PyObject * allocate_r_platacc_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        allocate_r_platacc_f(&dim1);
        return Py_BuildValue("i", 0);
}

PyObject * deallocate_r_platacc_C(PyObject* self, PyObject* args)
{
        deallocate_r_platacc_f();
        return Py_BuildValue("i", 0);
}

PyObject * get_peg_info_C(PyObject* self, PyObject* args)
{
        get_peg_info_f();
        return Py_BuildValue("i", 0);
}
PyObject * setNumObservations_C(PyObject* self, PyObject* args)
{
        int var;
        if(!PyArg_ParseTuple(args, "i", &var))
        {
                return NULL;
        }
        setNumObservations_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * setStartLineSlc_C(PyObject* self, PyObject* args)
{
        int var;
        if(!PyArg_ParseTuple(args, "i", &var))
        {
                return NULL;
        }
        setStartLineSlc_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * setNumLinesInt_C(PyObject* self, PyObject* args)
{
        int var;
        if(!PyArg_ParseTuple(args, "i", &var))
        {
                return NULL;
        }
        setNumLinesInt_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * setNumLinesSlc_C(PyObject* self, PyObject* args)
{
        int var;
        if(!PyArg_ParseTuple(args, "i", &var))
        {
                return NULL;
        }
        setNumLinesSlc_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * setNumAzimuthLooksInt_C(PyObject* self, PyObject* args)
{
        int var;
        if(!PyArg_ParseTuple(args, "i", &var))
        {
                return NULL;
        }
        setNumAzimuthLooksInt_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * setPrfSlc_C(PyObject* self, PyObject* args)
{
        double var;
        if(!PyArg_ParseTuple(args, "d", &var))
        {
                return NULL;
        }
        setPrfSlc_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * setTimeSlc_C(PyObject* self, PyObject* args)
{
        double var;
        if(!PyArg_ParseTuple(args, "d", &var))
        {
                return NULL;
        }
        setTimeSlc_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * setTime_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        PyObject * list;
        if(!PyArg_ParseTuple(args, "Oi", &list,&dim1))
        {
                return NULL;
        }
        if(!PyList_Check(list))
        {
                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                exit(1);
        }
        double *  vectorV = new double[dim1];
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyList_GetItem(list,i);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
                        exit(1);
                }
                vectorV[i] = (double) PyFloat_AsDouble(listEl);
                if(PyErr_Occurred() != NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
                        exit(1);
                }
        }
        setTime_f(vectorV, &dim1);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setPositionVector_C(PyObject* self, PyObject* args)
{
        PyObject * list;
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "Oii", &list, &dim1, &dim2))
        {
                return NULL;
        }
        if(!PyList_Check(list))
        {
                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                exit(1);
        }
        double *  vectorV = new double[dim1*dim2];
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyList_GetItem(list,i);
                if(!PyList_Check(listEl))
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                        exit(1);
                }
                for(int j = 0; j < dim2; ++j)
                {
                        PyObject * listElEl = PyList_GetItem(listEl,j);
                        if(listElEl == NULL)
                        {
                                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
                                exit(1);
                        }
                        vectorV[dim2*i + j] = (double) PyFloat_AsDouble(listElEl);
                        if(PyErr_Occurred() != NULL)
                        {
                                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
                                exit(1);
                        }
                }
        }
        setPositionVector_f(vectorV, &dim1, &dim2);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setVelocityVector_C(PyObject* self, PyObject* args)
{
        PyObject * list;
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "Oii", &list, &dim1, &dim2))
        {
                return NULL;
        }
        if(!PyList_Check(list))
        {
                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                exit(1);
        }
        double *  vectorV = new double[dim1*dim2];
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyList_GetItem(list,i);
                if(!PyList_Check(listEl))
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                        exit(1);
                }
                for(int j = 0; j < dim2; ++j)
                {
                        PyObject * listElEl = PyList_GetItem(listEl,j);
                        if(listElEl == NULL)
                        {
                                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
                                exit(1);
                        }
                        vectorV[dim2*i + j] = (double) PyFloat_AsDouble(listElEl);
                        if(PyErr_Occurred() != NULL)
                        {
                                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
                                exit(1);
                        }
                }
        }
        setVelocityVector_f(vectorV, &dim1, &dim2);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setAccelerationVector_C(PyObject* self, PyObject* args)
{
        PyObject * list;
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "Oii", &list, &dim1, &dim2))
        {
                return NULL;
        }
        if(!PyList_Check(list))
        {
                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                exit(1);
        }
        double *  vectorV = new double[dim1*dim2];
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl = PyList_GetItem(list,i);
                if(!PyList_Check(listEl))
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Expecting a list type object" << endl;
                        exit(1);
                }
                for(int j = 0; j < dim2; ++j)
                {
                        PyObject * listElEl = PyList_GetItem(listEl,j);
                        if(listElEl == NULL)
                        {
                                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot retrieve list element" << endl;
                                exit(1);
                        }
                        vectorV[dim2*i + j] = (double) PyFloat_AsDouble(listElEl);
                        if(PyErr_Occurred() != NULL)
                        {
                                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot convert Py Object to C " << endl;
                                exit(1);
                        }
                }
        }
        setAccelerationVector_f(vectorV, &dim1, &dim2);
        delete [] vectorV;
        return Py_BuildValue("i", 0);
}

PyObject * setPlanetGM_C(PyObject* self, PyObject* args)
{
        double var;
        if(!PyArg_ParseTuple(args, "d", &var))
        {
                return NULL;
        }
        setPlanetGM_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * setPlanetSpinRate_C(PyObject* self, PyObject* args)
{
        double var;
        if(!PyArg_ParseTuple(args, "d", &var))
        {
                return NULL;
        }
        setPlanetSpinRate_f(&var);
        return Py_BuildValue("i", 0);
}
PyObject * getPegLat_C(PyObject* self, PyObject* args)
{
        double var;
        getPegLat_f(&var);
        return Py_BuildValue("d",var);
}
PyObject * getPegLon_C(PyObject* self, PyObject* args)
{
        double var;
        getPegLon_f(&var);
        return Py_BuildValue("d",var);
}
PyObject * getPegHeight_C(PyObject* self, PyObject* args)
{
        double var;
        getPegHeight_f(&var);
        return Py_BuildValue("d",var);
}
PyObject * getPegHeading_C(PyObject* self, PyObject* args)
{
        double var;
        getPegHeading_f(&var);
        return Py_BuildValue("d",var);
}
PyObject * getVerticalFit_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        double *  vectorV = new double[dim1];
        getVerticalFit_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list);
}

PyObject * getHorizontalFit_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        double *  vectorV = new double[dim1];
        getHorizontalFit_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list);
}

PyObject * getVerticalVelocityFit_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        double *  vectorV = new double[dim1];
        getVerticalVelocityFit_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list);
}

PyObject * getCrossTrackVelocityFit_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        double *  vectorV = new double[dim1];
        getCrossTrackVelocityFit_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list);
}

PyObject * getAlongTrackVelocityFit_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        double *  vectorV = new double[dim1];
        getAlongTrackVelocityFit_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list);
}

PyObject * getPegRadius_C(PyObject* self, PyObject* args)
{
        double var;
        getPegRadius_f(&var);
        return Py_BuildValue("d",var);
}
PyObject * getGroundSpacing_C(PyObject* self, PyObject* args)
{
        double var;
        getGroundSpacing_f(&var);
        return Py_BuildValue("d",var);
}
PyObject * getTranslationVector_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        double *  vectorV = new double[dim1];
        getTranslationVector_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list);
}

PyObject * getTransformationMatrix_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "ii", &dim1, &dim2))
        {
                return NULL;
        }
        PyObject * list1 = PyList_New(dim1);
        double *  vectorV = new double[dim1*dim2];
        getTransformationMatrix_f(vectorV, &dim1, &dim2);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * list2 = PyList_New(dim2);
                for(int j = 0; j  < dim2; ++j)
                {
                        PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i*dim2 + j]);
                        if(listEl == NULL)
                        {
                                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                                exit(1);
                        }
                        PyList_SetItem(list2,j,listEl);
                }
                PyList_SetItem(list1,i,list2);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list1);
}

PyObject * getIntPosition_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "ii", &dim1, &dim2))
        {
                return NULL;
        }
        PyObject * list1 = PyList_New(dim1);
        double *  vectorV = new double[dim1*dim2];
        getIntPosition_f(vectorV, &dim1, &dim2);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * list2 = PyList_New(dim2);
                for(int j = 0; j  < dim2; ++j)
                {
                        PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i*dim2 + j]);
                        if(listEl == NULL)
                        {
                                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                                exit(1);
                        }
                        PyList_SetItem(list2,j,listEl);
                }
                PyList_SetItem(list1,i,list2);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list1);
}

PyObject * getIntVelocity_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        int dim2 = 0;
        if(!PyArg_ParseTuple(args, "ii", &dim1, &dim2))
        {
                return NULL;
        }
        PyObject * list1 = PyList_New(dim1);
        double *  vectorV = new double[dim1*dim2];
        getIntVelocity_f(vectorV, &dim1, &dim2);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * list2 = PyList_New(dim2);
                for(int j = 0; j  < dim2; ++j)
                {
                        PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i*dim2 + j]);
                        if(listEl == NULL)
                        {
                                cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                                exit(1);
                        }
                        PyList_SetItem(list2,j,listEl);
                }
                PyList_SetItem(list1,i,list2);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list1);
}

PyObject * getPegVelocity_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        double *  vectorV = new double[dim1];
        getPegVelocity_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list);
}

PyObject * getPlatformSCHVelocity_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        double *  vectorV = new double[dim1];
        getPlatformSCHVelocity_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list);
}

PyObject * getPlatformSCHAcceleration_C(PyObject* self, PyObject* args)
{
        int dim1 = 0;
        if(!PyArg_ParseTuple(args, "i", &dim1))
        {
                return NULL;
        }
        PyObject * list = PyList_New(dim1);
        double *  vectorV = new double[dim1];
        getPlatformSCHAcceleration_f(vectorV, &dim1);
        for(int i = 0; i  < dim1; ++i)
        {
                PyObject * listEl =  PyFloat_FromDouble((double) vectorV[i]);
                if(listEl == NULL)
                {
                        cout << "Error in file " << __FILE__ << " at line " << __LINE__ << ". Cannot set list element" << endl;
                        exit(1);
                }
                PyList_SetItem(list,i, listEl);
        }
        delete [] vectorV;
        return Py_BuildValue("O",list);
}

PyObject * getTimeFirstScene_C(PyObject* self, PyObject* args)
{
        double var;
        getTimeFirstScene_f(&var);
        return Py_BuildValue("d",var);
}
