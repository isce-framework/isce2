//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// United States Government Sponsorship acknowledged. This software is subject to
// U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
// (No [Export] License Required except when exporting to an embargoed country,
// end user, or in support of a prohibited end use). By downloading this software,
// the user agrees to comply with all applicable U.S. export laws and regulations.
// The user has the responsibility to obtain export licenses, or other export
// authority as may be required before exporting this software to any 'EAR99'
// embargoed foreign country or citizen of those countries.
//
// Author: Giangi Sacco
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





#ifndef denseoffsetsmodule_h
#define denseoffsetsmodule_h

#include <Python.h>
#include <stdint.h>
#include "denseoffsetsmoduleFortTrans.h"

extern "C"
{
        void denseoffsets_f(uint64_t *, uint64_t *, uint64_t *, uint64_t *);
        PyObject * denseoffsets_C(PyObject *, PyObject *);
        void setLineLength1_f(int *);
        PyObject * setLineLength1_C(PyObject *, PyObject *);
        void setLineLength2_f(int *);
        PyObject * setLineLength2_C(PyObject *, PyObject *);
        void setFileLength1_f(int *);
        PyObject * setFileLength1_C(PyObject *, PyObject *);
        void setFileLength2_f(int *);
        PyObject * setFileLength2_C(PyObject *, PyObject *);
        void setFirstSampleAcross_f(int *);
        PyObject * setFirstSampleAcross_C(PyObject *, PyObject *);
        void setLastSampleAcross_f(int *);
        PyObject * setLastSampleAcross_C(PyObject *, PyObject *);
        void setSkipSampleAcross_f(int *);
        PyObject * setSkipSampleAcross_C(PyObject *, PyObject *);
        void setFirstSampleDown_f(int *);
        PyObject * setFirstSampleDown_C(PyObject *, PyObject *);
        void setLastSampleDown_f(int *);
        PyObject * setLastSampleDown_C(PyObject *, PyObject *);
        void setSkipSampleDown_f(int *);
        PyObject * setSkipSampleDown_C(PyObject *, PyObject *);
        void setAcrossGrossOffset_f(int *);
        PyObject * setAcrossGrossOffset_C(PyObject *, PyObject *);
        void setDownGrossOffset_f(int *);
        PyObject * setDownGrossOffset_C(PyObject *, PyObject *);
        void setScaleFactorX_f(float *);
        PyObject * setScaleFactorX_C(PyObject *, PyObject *);
        void setScaleFactorY_f(float *);
        PyObject * setScaleFactorY_C(PyObject *, PyObject *);
        void setDebugFlag_f(char *, int *);
        PyObject * setDebugFlag_C(PyObject *, PyObject *);
        void setWindowSizeWidth_f(int *);
        PyObject * setWindowSizeWidth_C(PyObject *, PyObject *);
        void setWindowSizeHeight_f(int *);
        PyObject * setWindowSizeHeight_C(PyObject *, PyObject *);
        void setSearchWindowSizeWidth_f(int *);
        PyObject * setSearchWindowSizeWidth_C(PyObject *, PyObject *);
        void setSearchWindowSizeHeight_f(int *);
        PyObject *setSearchWindowSizeHeight_C(PyObject *, PyObject *);
        void setZoomWindowSize_f(int *);
        PyObject * setZoomWindowSize_C(PyObject *, PyObject *);
        void setOversamplingFactor_f(int *);
        PyObject * setOversamplingFactor_C(PyObject *, PyObject *);
        void setIsComplex1_f(int *);
        PyObject * setIsComplex1_C(PyObject *, PyObject *);
        void setIsComplex2_f(int *);
        PyObject * setIsComplex2_C(PyObject *, PyObject *);
        void setBand1_f(int *);
        PyObject * setBand1_C(PyObject *, PyObject *);
        void setBand2_f(int *);
        PyObject * setBand2_C(PyObject *, PyObject *);
        void setNormalizeFlag_f(int *);
        PyObject *setNormalizeFlag_C(PyObject*, PyObject*);
}


static PyMethodDef denseoffsets_methods[] =
{
        {"denseoffsets_Py", denseoffsets_C, METH_VARARGS, " "},
        {"setLineLength1_Py", setLineLength1_C, METH_VARARGS, " "},
        {"setLineLength2_Py", setLineLength2_C, METH_VARARGS, " "},
        {"setFileLength1_Py", setFileLength1_C, METH_VARARGS, " "},
        {"setFileLength2_Py", setFileLength2_C, METH_VARARGS, " "},
        {"setFirstSampleAcross_Py", setFirstSampleAcross_C, METH_VARARGS, " "},
        {"setLastSampleAcross_Py", setLastSampleAcross_C, METH_VARARGS, " "},
        {"setSkipSampleAcross_Py", setSkipSampleAcross_C, METH_VARARGS, " "},
        {"setFirstSampleDown_Py", setFirstSampleDown_C, METH_VARARGS, " "},
        {"setLastSampleDown_Py", setLastSampleDown_C, METH_VARARGS, " "},
        {"setSkipSampleDown_Py", setSkipSampleDown_C, METH_VARARGS, " "},
        {"setAcrossGrossOffset_Py", setAcrossGrossOffset_C, METH_VARARGS, " "},
        {"setDownGrossOffset_Py", setDownGrossOffset_C, METH_VARARGS, " "},
        {"setScaleFactorX_Py", setScaleFactorX_C, METH_VARARGS, " "},
        {"setScaleFactorY_Py", setScaleFactorY_C, METH_VARARGS, " "},
        {"setDebugFlag_Py", setDebugFlag_C, METH_VARARGS, " "},
        {"setWindowSizeWidth_Py", setWindowSizeWidth_C, METH_VARARGS, " "},
        {"setWindowSizeHeight_Py", setWindowSizeHeight_C, METH_VARARGS, " "},
        {"setSearchWindowSizeWidth_Py", setSearchWindowSizeWidth_C, METH_VARARGS, " "},
        {"setSearchWindowSizeHeight_Py", setSearchWindowSizeHeight_C, METH_VARARGS, " "},
        {"setZoomWindowSize_Py", setZoomWindowSize_C, METH_VARARGS, " "},
        {"setOversamplingFactor_Py", setOversamplingFactor_C, METH_VARARGS, " "},
        {"setIsComplex1_Py", setIsComplex1_C, METH_VARARGS, " "},
        {"setIsComplex2_Py", setIsComplex2_C, METH_VARARGS, " "},
        {"setBand1_Py", setBand1_C, METH_VARARGS, " "},
        {"setBand2_Py", setBand2_C, METH_VARARGS, " "},
        {"setNormalizeFlag_Py", setNormalizeFlag_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //denseoffsetsmodule_h
