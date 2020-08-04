#if !defined(__MROIPAC_AMPCORMODULE_H__)
#define __MROIPAC_AMPCORMODULE_H__

#include <Python.h>
#include "ampcormoduleFortTrans.h"

extern "C"
{
  //the fortran engine
  void ampcor_f(void*,void*,int*,int*);
  PyObject* ampcor_C(PyObject*, PyObject*);
  

  //fortran routines for setting the module variables
  void setImageDatatype1_f(char*, int);
  PyObject* setImageDatatype1_C(PyObject*, PyObject*);

  void setLineLength1_f(int*);
  PyObject* setLineLength1_C(PyObject*, PyObject*);

  void setImageLength1_f(int*);
  PyObject* setImageLength1_C(PyObject*, PyObject*);

  void setImageDatatype2_f(char*, int);
  PyObject* setImageDatatype2_C(PyObject*, PyObject*);

  void setLineLength2_f(int*);
  PyObject* setLineLength2_C(PyObject*, PyObject*);

  void setImageLength2_f(int*);
  PyObject* setImageLength2_C(PyObject*, PyObject*);

  void setFirstSampleDown_f(int*);
  void setLastSampleDown_f(int*);
  void setSkipSampleDown_f(int*);
  void setFirstSampleAcross_f(int*);
  void setLastSampleAcross_f(int*);
  void setSkipSampleAcross_f(int*);
  PyObject* setFirstSampleDown_C(PyObject*, PyObject*);
  PyObject* setLastSampleDown_C(PyObject*, PyObject*);
  PyObject* setSkipSampleDown_C(PyObject*, PyObject*);
  PyObject* setFirstSampleAcross_C(PyObject*, PyObject*);
  PyObject* setLastSampleAcross_C(PyObject*, PyObject*);
  PyObject* setSkipSampleAcross_C(PyObject*, PyObject*);


  void setWindowSizeWidth_f(int*);
  void setWindowSizeHeight_f(int*);
  PyObject* setWindowSizeWidth_C(PyObject*, PyObject*);
  PyObject* setWindowSizeHeight_C(PyObject*, PyObject*);

  void setSearchWindowSizeWidth_f(int*);
  void setSearchWindowSizeHeight_f(int*);
  PyObject* setSearchWindowSizeWidth_C(PyObject*, PyObject*);
  PyObject* setSearchWindowSizeHeight_C(PyObject*, PyObject*);

  void setAcrossLooks_f(int*);
  void setDownLooks_f(int*);
  PyObject* setAcrossLooks_C(PyObject*, PyObject*);
  PyObject* setDownLooks_C(PyObject*, PyObject*);

  void setOversamplingFactor_f(int*);
  void setZoomWindowSize_f(int*);
  PyObject* setOversamplingFactor_C(PyObject*, PyObject*);
  PyObject* setZoomWindowSize_C(PyObject*, PyObject*);

  void setAcrossGrossOffset_f(int*);
  void setDownGrossOffset_f(int*);
  PyObject* setAcrossGrossOffset_C(PyObject*, PyObject*);
  PyObject* setDownGrossOffset_C(PyObject*, PyObject*);
  
  void setThresholdSNR_f(double*);
  void setThresholdCov_f(double*);
  PyObject* setThresholdSNR_C(PyObject*, PyObject*);
  PyObject* setThresholdCov_C(PyObject*, PyObject*);

  void setDebugFlag_f(int*);
  void setDisplayFlag_f(int*);
  PyObject* setDebugFlag_C(PyObject*, PyObject*);
  PyObject* setDisplayFlag_C(PyObject*, PyObject*);

  
  //print the module variables
  void ampcorPrintState_f();
  PyObject* ampcorPrintState_C(PyObject*, PyObject*);


  //fortran routines for getting the module variables
  void getNumRows_f(int *);
  PyObject *getNumRows_C(PyObject*, PyObject*);


  void getLocationAcross_f(int *, int *);
  void allocate_locationAcross_f(int *);
  void deallocate_locationAcross_f();
  PyObject * allocate_locationAcross_C(PyObject *, PyObject *);
  PyObject * deallocate_locationAcross_C(PyObject *, PyObject *);
  PyObject * getLocationAcross_C(PyObject *, PyObject *);


  void getLocationAcrossOffset_f(float *, int *);
  void allocate_locationAcrossOffset_f(int *);
  void deallocate_locationAcrossOffset_f();
  PyObject * allocate_locationAcrossOffset_C(PyObject *, PyObject *);
  PyObject * deallocate_locationAcrossOffset_C(PyObject *, PyObject *);
  PyObject * getLocationAcrossOffset_C(PyObject *, PyObject *);

  void getLocationDown_f(int *, int *);
  void allocate_locationDown_f(int *);
  void deallocate_locationDown_f();
  PyObject * allocate_locationDown_C(PyObject *, PyObject *);
  PyObject * deallocate_locationDown_C(PyObject *, PyObject *);
  PyObject * getLocationDown_C(PyObject *, PyObject *);

  void getLocationDownOffset_f(float *, int *);
  void allocate_locationDownOffset_f(int *);
  void deallocate_locationDownOffset_f();
  PyObject * allocate_locationDownOffset_C(PyObject *, PyObject *);
  PyObject * deallocate_locationDownOffset_C(PyObject *, PyObject *);
  PyObject * getLocationDownOffset_C(PyObject *, PyObject *);

  void getSNR_f(float *, int *);
  void allocate_snrRet_f(int *);
  void deallocate_snrRet_f();
  PyObject * allocate_snrRet_C(PyObject *, PyObject *);
  PyObject * deallocate_snrRet_C(PyObject *, PyObject *);
  PyObject * getSNR_C(PyObject *, PyObject *);

  void getCov1_f(float *, int *);
  void allocate_cov1Ret_f(int *);
  void deallocate_cov1Ret_f();
  PyObject * allocate_cov1Ret_C(PyObject *, PyObject *);
  PyObject * deallocate_cov1Ret_C(PyObject *, PyObject *);
  PyObject * getCov1_C(PyObject *, PyObject *);

  void getCov2_f(float *, int *);
  void allocate_cov2Ret_f(int *);
  void deallocate_cov2Ret_f();
  PyObject * allocate_cov2Ret_C(PyObject *, PyObject *);
  PyObject * deallocate_cov2Ret_C(PyObject *, PyObject *);
  PyObject * getCov2_C(PyObject *, PyObject *);

  void getCov3_f(float *, int *);
  void allocate_cov3Ret_f(int *);
  void deallocate_cov3Ret_f();
  PyObject * allocate_cov3Ret_C(PyObject *, PyObject *);
  PyObject * deallocate_cov3Ret_C(PyObject *, PyObject *);
  PyObject * getCov3_C(PyObject *, PyObject *);

  void setScaleFactorX_f(float*);
  void setScaleFactorY_f(float*);
  PyObject * setScaleFactorX_C(PyObject*, PyObject*);
  PyObject * setScaleFactorY_C(PyObject*, PyObject*);

  void setOversamplingFactorFilt_f(int*);
  PyObject* setOversamplingFactorFilt_C(PyObject*, PyObject*);
  void setWinsizeFilt_f(int*);
  PyObject* setWinsizeFilt_C(PyObject*, PyObject*);
}



//Method Table
static PyMethodDef ampcor_methods[] =
{
 {"ampcor_Py", ampcor_C, METH_VARARGS, " "},

 //set state methods

 { "setImageDataType1_Py", setImageDatatype1_C, METH_VARARGS," "},
 { "setLineLength1_Py", setLineLength1_C, METH_VARARGS," "},
 { "setImageLength1_Py", setImageLength1_C, METH_VARARGS, " "},
 { "setImageDataType2_Py", setImageDatatype2_C, METH_VARARGS," "},
 { "setLineLength2_Py", setLineLength2_C, METH_VARARGS," "},
 { "setImageLength2_Py", setImageLength2_C, METH_VARARGS, " "},
 { "setFirstSampleDown_Py", setFirstSampleDown_C, METH_VARARGS, " "},
 { "setLastSampleDown_Py", setLastSampleDown_C, METH_VARARGS, " "},
 { "setSkipSampleDown_Py", setSkipSampleDown_C, METH_VARARGS, " "},
 { "setFirstSampleAcross_Py", setFirstSampleAcross_C, METH_VARARGS, " "},
 { "setLastSampleAcross_Py", setLastSampleAcross_C, METH_VARARGS, " "},
 { "setSkipSampleAcross_Py", setSkipSampleAcross_C, METH_VARARGS, " "},
 { "setWindowSizeWidth_Py", setWindowSizeWidth_C, METH_VARARGS, " "},
 { "setWindowSizeHeight_Py", setWindowSizeHeight_C, METH_VARARGS, " "},
 { "setSearchWindowSizeWidth_Py", setSearchWindowSizeWidth_C, METH_VARARGS, " "},
 { "setSearchWindowSizeHeight_Py", setSearchWindowSizeHeight_C, METH_VARARGS, " "},
 { "setAcrossLooks_Py", setAcrossLooks_C, METH_VARARGS, " "},
 { "setDownLooks_Py", setDownLooks_C, METH_VARARGS, " "},
 { "setOversamplingFactor_Py", setOversamplingFactor_C, METH_VARARGS, " "},
 { "setZoomWindowSize_Py", setZoomWindowSize_C, METH_VARARGS, " "},
 { "setAcrossGrossOffset_Py", setAcrossGrossOffset_C, METH_VARARGS, " "},
 { "setDownGrossOffset_Py", setDownGrossOffset_C, METH_VARARGS, " "},
 { "setThresholdSNR_Py", setThresholdSNR_C, METH_VARARGS, " "},
 { "setThresholdCov_Py", setThresholdCov_C, METH_VARARGS, " "},
 { "setDebugFlag_Py", setDebugFlag_C, METH_VARARGS, " "},
 { "setDisplayFlag_Py", setDisplayFlag_C, METH_VARARGS, " "},
 { "setScaleFactorX_Py", setScaleFactorX_C, METH_VARARGS, " "},
 { "setScaleFactorY_Py", setScaleFactorY_C, METH_VARARGS, " "},

 //print state method
 { "ampcorPrintState_Py", ampcorPrintState_C, METH_VARARGS, " "},

 //get state methods
 { "getNumRows_Py", getNumRows_C, METH_VARARGS, " "},
 { "getCov1_Py", getCov1_C, METH_VARARGS, " "},
 { "getCov2_Py", getCov2_C, METH_VARARGS, " "},
 { "getCov3_Py", getCov3_C, METH_VARARGS, " "},
 { "getSNR_Py", getSNR_C, METH_VARARGS, " "},
 { "getLocationAcross_Py", getLocationAcross_C, METH_VARARGS, " "},
 { "getLocationAcrossOffset_Py", getLocationAcrossOffset_C, METH_VARARGS, " "},
 { "getLocationDown_Py", getLocationDown_C, METH_VARARGS, " "},
 { "getLocationDownOffset_Py", getLocationDownOffset_C, METH_VARARGS, " "},

 //allocate methods
 { "allocate_locationAcross_Py", allocate_locationAcross_C, METH_VARARGS, " "},
 { "allocate_locationAcrossOffset_Py", allocate_locationAcrossOffset_C, METH_VARARGS, " "},
 { "allocate_locationDown_Py", allocate_locationDown_C, METH_VARARGS, " "},
 { "allocate_locationDownOffset_Py", allocate_locationDownOffset_C, METH_VARARGS, " "},
 { "allocate_snrRet_Py", allocate_snrRet_C, METH_VARARGS, " "},
 { "allocate_cov1Ret_Py", allocate_cov1Ret_C, METH_VARARGS, " "},
 { "allocate_cov2Ret_Py", allocate_cov2Ret_C, METH_VARARGS, " "},
 { "allocate_cov3Ret_Py", allocate_cov3Ret_C, METH_VARARGS, " "},

 //deallocate methods
 { "deallocate_locationAcross_Py", deallocate_locationAcross_C, METH_VARARGS, " "},
 { "deallocate_locationAcrossOffset_Py", deallocate_locationAcrossOffset_C, METH_VARARGS, " "},
 { "deallocate_locationDown_Py", deallocate_locationDown_C, METH_VARARGS, " "},
 { "deallocate_locationDownOffset_Py", deallocate_locationDownOffset_C, METH_VARARGS, " "},
 { "deallocate_snrRet_Py", deallocate_snrRet_C, METH_VARARGS, " "},
 { "deallocate_cov1Ret_Py", deallocate_cov1Ret_C, METH_VARARGS, " "},
 { "deallocate_cov2Ret_Py", deallocate_cov2Ret_C, METH_VARARGS, " "},
 { "deallocate_cov3Ret_Py", deallocate_cov3Ret_C, METH_VARARGS, " "},

 { "setWinsizeFilt_Py", setWinsizeFilt_C, METH_VARARGS, " "},
 { "setOversamplingFactorFilt_Py", setOversamplingFactorFilt_C, METH_VARARGS, " "},

 {NULL, NULL, 0 , NULL}
};

#endif

//end of file
