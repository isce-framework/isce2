#ifndef fitoffmodule_h
#define fitoffmodule_h

#include <Python.h>
#include "fitoffmoduleFortTrans.h"

extern "C"
{
        void fitoff_f();
        void setMinPoint_f(int *);
        void setNSig_f(double *);
        void setMaxRms_f(double *);
        void getAffineVector_f(double *);
        PyObject * fitoff_C(PyObject *, PyObject *);
        PyObject * setMinPoint_C(PyObject *, PyObject *);
        PyObject * setNSig_C(PyObject *, PyObject *);
        PyObject * setMaxRms_C(PyObject *, PyObject *);
        PyObject * getAffineVector_C(PyObject *, PyObject *);

        void setNumberLines_f(int *);
        PyObject * setNumberLines_C(PyObject *, PyObject *);

        void setMaxIter_f(int*);
        PyObject * setMaxIter_C(PyObject *, PyObject *);

        void setMinIter_f(int*);
        PyObject * setMinIter_C(PyObject *, PyObject *);

        void setL1normFlag_f(int*);
        PyObject * setL1normFlag_C(PyObject *, PyObject *);

        void setLocationAcross_f(double *, int *);
        PyObject * setLocationAcross_C(PyObject *, PyObject *);

        void setLocationDown_f(double *, int*);
        PyObject * setLocationDown_C(PyObject *, PyObject *);

        void setLocationAcrossOffset_f(double *, int *);
        PyObject * setLocationAcrossOffset_C(PyObject *, PyObject *);

        void setLocationDownOffset_f(double *, int *);
        PyObject * setLocationDownOffset_C(PyObject *, PyObject *);

        void setSNR_f(double *, int *);
        PyObject * setSNR_C(PyObject *, PyObject *);

        void setCovAcross_f(double *, int *);
        PyObject * setCovAcross_C(PyObject *, PyObject *);

        void setCovDown_f(double *, int *);
        PyObject * setCovDown_C(PyObject *, PyObject *);

        void setCovCross_f(double *, int *);
        PyObject * setCovCross_C(PyObject *, PyObject *);

        void setStdWriter_f(uint64_t *);
        PyObject * setStdWriter_C(PyObject *, PyObject *);

        void allocate_LocationAcross_f();
        void allocate_LocationDown_f();
        void allocate_LocationAcrossOffset_f();
        void allocate_LocationDownOffset_f();
        void allocate_SNR_f();
        void allocate_Covariance_f();

        void deallocate_LocationAcross_f();
        void deallocate_LocationDown_f();
        void deallocate_LocationAcrossOffset_f();
        void deallocate_LocationDownOffset_f();
        void deallocate_SNR_f();
        void deallocate_Covariance_f();

        PyObject * allocate_Arrays_C(PyObject*, PyObject *);
        PyObject * deallocate_Arrays_C(PyObject*, PyObject*);

        void getNumberOfRefinedOffsets_f(int*);
        PyObject * getNumberOfRefinedOffsets_C(PyObject*, PyObject*);

        PyObject * getRefinedOffsetField_C(PyObject*, PyObject*);
        void getRefinedLocationAcross_f(double*);
        void getRefinedLocationDown_f(double*);
        void getRefinedLocationAcrossOffset_f(double*);
        void getRefinedLocationDownOffset_f(double*);
        void getRefinedSNR_f(double*);
        void getRefinedCovAcross_f(double*);
        void getRefinedCovDown_f(double*);
        void getRefinedCovCross_f(double*);
}

static PyMethodDef fitoff_methods[] =
{
        {"fitoff_Py", fitoff_C, METH_VARARGS, " "},
        {"setMinPoint_Py", setMinPoint_C, METH_VARARGS, " "},
        {"setNSig_Py", setNSig_C, METH_VARARGS, " "},
        {"setMaxRms_Py", setMaxRms_C, METH_VARARGS, " "},
        {"setNumberLines_Py", setNumberLines_C, METH_VARARGS, " "},
        {"setMinIter_Py", setMinIter_C, METH_VARARGS, " "},
        {"setMaxIter_Py", setMaxIter_C, METH_VARARGS, " "},
        {"setL1normFlag_Py", setL1normFlag_C, METH_VARARGS, " "},
        {"setLocationAcross_Py", setLocationAcross_C, METH_VARARGS, " "},
        {"setLocationDown_Py", setLocationDown_C, METH_VARARGS, " "},
        {"setLocationAcrossOffset_Py", setLocationAcrossOffset_C, METH_VARARGS, " "},
        {"setLocationDownOffset_Py", setLocationDownOffset_C, METH_VARARGS, " "},
        {"setSNR_Py", setSNR_C, METH_VARARGS, " "},
        {"setCovAcross_Py", setCovAcross_C, METH_VARARGS, " "},
        {"setCovDown_Py", setCovDown_C, METH_VARARGS, " "},
        {"setCovCross_Py", setCovCross_C, METH_VARARGS, " "},
        {"setStdWriter_Py", setStdWriter_C, METH_VARARGS, " "},
        {"getAffineVector_Py", getAffineVector_C, METH_VARARGS, " "},
        {"allocateArrays_Py", allocate_Arrays_C, METH_VARARGS, " "},
        {"deallocateArrays_Py", deallocate_Arrays_C, METH_VARARGS, " "},
        {"getNumberOfRefinedOffsets_Py", getNumberOfRefinedOffsets_C, METH_VARARGS, " "},
        {"getRefinedOffsetField_Py", getRefinedOffsetField_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //fitoffmodule_h
