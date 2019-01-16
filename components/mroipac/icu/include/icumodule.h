#if !defined(__MROIPAC_ICUMODULE_H__)
#define __MROIPAC_ICUMODULE_H__

#include <Python.h>
#include "icumoduleFortTrans.h"


extern "C"
{
  //the fortran engine
  void icu_f(void*,void*,void*,void*,void*,void*,void*,void*);
  PyObject* icu_C(PyObject*, PyObject*);

  //fortran routines for setting the module variables
  void setWidth_f(int*);
  PyObject* setWidth_C(PyObject*, PyObject*);

  void setStartSample_f(int*);
  PyObject* setStartSample_C(PyObject*, PyObject*);

  void setEndSample_f(int*);
  PyObject* setEndSample_C(PyObject*, PyObject*);

  void setStartingLine_f(int*);
  PyObject* setStartingLine_C(PyObject*, PyObject*);

  void setLength_f(int*);
  PyObject* setLength_C(PyObject*, PyObject*);

  void setAzimuthBufferSize_f(int*);
  PyObject* setAzimuthBufferSize_C(PyObject*, PyObject*);

  void setOverlap_f(int*);
  PyObject* setOverlap_C(PyObject*, PyObject*);
 
  void setFilteringFlag_f(int*);
  PyObject* setFilteringFlag_C(PyObject*, PyObject*);

  void setUnwrappingFlag_f(int*);
  PyObject* setUnwrappingFlag_C(PyObject*, PyObject*);

  void setFilterType_f(int*);
  PyObject* setFilterType_C(PyObject*, PyObject*);

  void setLPRangeWinSize_f(float*);
  PyObject* setLPRangeWinSize_C(PyObject*, PyObject*);

  void setLPAzimuthWinSize_f(float*);
  PyObject* setLPAzimuthWinSize_C(PyObject*, PyObject*);

  void setFilterExponent_f(float*);
  PyObject* setFilterExponent_C(PyObject*, PyObject*);

  void setUseAmplitudeFlag_f(int*);
  PyObject* setUseAmplitudeFlag_C(PyObject*,PyObject*);

  void setCorrelationType_f(int*);
  PyObject* setCorrelationType_C(PyObject*, PyObject*);

  void setCorrelationBoxSize_f(int*);
  PyObject* setCorrelationBoxSize_C(PyObject*, PyObject*);

  void setPhaseSigmaBoxSize_f(int*);
  PyObject* setPhaseSigmaBoxSize_C(PyObject*, PyObject*);

  void setPhaseVarThreshold_f(float*);
  PyObject* setPhaseVarThreshold_C(PyObject*, PyObject*);

  void setInitCorrThreshold_f(float*);
  PyObject* setInitCorrThreshold_C(PyObject*, PyObject*);

  void setCorrThreshold_f(float*);
  PyObject* setCorrThreshold_C(PyObject*, PyObject*);

  void setCorrThresholdInc_f(float*);
  PyObject* setCorrThresholdInc_C(PyObject*, PyObject*);

  void setNeuTypes_f(int*, int*);
  PyObject* setNeuTypes_C(PyObject*, PyObject*);

  void setNeuThreshold_f(float*, float*, float*);
  PyObject* setNeuThreshold_C(PyObject*, PyObject*);

  void setBootstrapSize_f(int*, int*);
  PyObject* setBootstrapSize_C(PyObject*, PyObject*);

  void setNumTreeSets_f(int*);
  PyObject* setNumTreeSets_C(PyObject*, PyObject*);

  void setTreeType_f(int*);
  PyObject* setTreeType_C(PyObject*, PyObject*);
}



//Method Table
static PyMethodDef icu_methods[] =
{
 {"icu_Py", icu_C, METH_VARARGS, " "},

 //set state methods
 {"setWidth_Py", setWidth_C, METH_VARARGS, " "},
 {"setStartSample_Py", setStartSample_C, METH_VARARGS, " "},
 {"setEndSample_Py", setEndSample_C, METH_VARARGS, " "},
 {"setStartingLine_Py", setStartingLine_C, METH_VARARGS, " "},
 {"setLength_Py", setLength_C, METH_VARARGS, " "},
 {"setAzimuthBufferSize_Py", setAzimuthBufferSize_C, METH_VARARGS, " "},
 {"setOverlap_Py", setOverlap_C, METH_VARARGS, " "},
 {"setFilteringFlag_Py", setFilteringFlag_C, METH_VARARGS, " "},
 {"setUnwrappingFlag_Py", setUnwrappingFlag_C, METH_VARARGS, " "},
 {"setFilterType_Py", setFilterType_C, METH_VARARGS, " "},
 {"setLPRangeWinSize_Py", setLPRangeWinSize_C, METH_VARARGS, " "},
 {"setLPAzimuthWinSize_Py", setLPAzimuthWinSize_C, METH_VARARGS, " "},
 {"setFilterExponent_Py", setFilterExponent_C, METH_VARARGS, " "},
 {"setUseAmplitudeFlag_Py", setUseAmplitudeFlag_C, METH_VARARGS, " "},
 {"setCorrelationType_Py", setCorrelationType_C, METH_VARARGS, " "},
 {"setCorrelationBoxSize_Py", setCorrelationBoxSize_C, METH_VARARGS, " "},
 {"setPhaseSigmaBoxSize_Py", setPhaseSigmaBoxSize_C, METH_VARARGS, " "},
 {"setPhaseVarThreshold_Py", setPhaseVarThreshold_C, METH_VARARGS, " "},
 {"setInitCorrThreshold_Py", setInitCorrThreshold_C, METH_VARARGS, " "},
 {"setCorrThreshold_Py", setCorrThreshold_C, METH_VARARGS, " "},
 {"setCorrThresholdInc_Py", setCorrThresholdInc_C, METH_VARARGS, " "},
 {"setNeuTypes_Py", setNeuTypes_C, METH_VARARGS, " "},
 {"setNeuThreshold_Py", setNeuThreshold_C, METH_VARARGS, " "},
 {"setBootstrapSize_Py", setBootstrapSize_C, METH_VARARGS, " "},
 {"setNumTreeSets_Py", setNumTreeSets_C, METH_VARARGS, " "},
 {"setTreeType_Py", setTreeType_C, METH_VARARGS, " "},
 {NULL, NULL, 0 , NULL}
};

#endif

//end of file
