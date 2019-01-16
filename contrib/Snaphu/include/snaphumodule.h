//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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




#ifndef snaphumodule_h
#define snaphumodule_h

#include <Python.h>

extern "C"
{
    #include "snaphu.h"
    int snaphu(infileT *infiles,outfileT *outfiles, paramT *params,
        long linelen);
    PyObject *setDefaults_C(PyObject *self,PyObject *args);
    PyObject *snaphu_C(PyObject *self,PyObject *args);
    PyObject *setInput_C(PyObject *self,PyObject *args);
    PyObject *setOutput_C(PyObject *self,PyObject *args);
    PyObject *setConnectedComponents_C(PyObject *self,PyObject *args);
    PyObject *setCostMode_C(PyObject *self,PyObject *args);
    PyObject *setWavelength_C(PyObject *self,PyObject *args);
    PyObject *setAltitude_C(PyObject *self,PyObject *args);
    PyObject *setEarthRadius_C(PyObject *self,PyObject *args);
    PyObject *setCorrfile_C(PyObject *self,PyObject *args);
    PyObject *setCorrLooks_C(PyObject *self,PyObject *args);
    PyObject *setDefoMaxCycles_C(PyObject *self, PyObject *args);
    PyObject *setInitMethod_C(PyObject *self, PyObject *args);
    PyObject *setMaxComponents_C(PyObject *self, PyObject *args);
    PyObject *setRangeLooks_C(PyObject *self, PyObject *args);
    PyObject *setAzimuthLooks_C(PyObject *self, PyObject *args);
    PyObject *setInitOnly_C(PyObject *self, PyObject *args);
    PyObject *setRegrowComponents_C(PyObject *self, PyObject *args);
    PyObject *setUnwrappedInput_C(PyObject *self, PyObject *args);
    PyObject *setMinConnectedComponentFraction_C(PyObject *self, PyObject *args);
    PyObject *setConnectedComponentThreshold_C(PyObject *self, PyObject *args);
    PyObject *setMagnitude_C(PyObject *self, PyObject *args);
    PyObject *setIntFileFormat_C(PyObject *self, PyObject *args);
    PyObject *setUnwFileFormat_C(PyObject *self, PyObject *args);
    PyObject *setCorFileFormat_C(PyObject *self, PyObject *args);
}

static PyMethodDef snaphu_methods[] =
{
    {"snaphu_Py",snaphu_C,METH_VARARGS," "},
    {"setInput_Py",setInput_C,METH_VARARGS," "},
    {"setOutput_Py",setOutput_C,METH_VARARGS," "},
    {"setConnectedComponents_Py",setConnectedComponents_C,METH_VARARGS," "},
    {"setCostMode_Py",setCostMode_C,METH_VARARGS," "},
    {"setWavelength_Py",setWavelength_C,METH_VARARGS," "},
    {"setAltitude_Py",setAltitude_C,METH_VARARGS," "},
    {"setEarthRadius_Py",setEarthRadius_C,METH_VARARGS," "},
    {"setDefaults_Py",setDefaults_C,METH_VARARGS," "},
    {"setCorrfile_Py",setCorrfile_C,METH_VARARGS," "},
    {"setCorrLooks_Py",setCorrLooks_C,METH_VARARGS," "},
    {"setDefoMaxCycles_Py",setDefoMaxCycles_C,METH_VARARGS," "},
    {"setInitMethod_Py",setInitMethod_C,METH_VARARGS," "},
    {"setMaxComponents_Py", setMaxComponents_C,METH_VARARGS," "},
    {"setRangeLooks_Py", setRangeLooks_C, METH_VARARGS, " "},
    {"setAzimuthLooks_Py", setAzimuthLooks_C, METH_VARARGS, " "},
    {"setInitOnly_Py", setInitOnly_C, METH_VARARGS, " "},
    {"setRegrowComponents_Py", setRegrowComponents_C, METH_VARARGS, " "},
    {"setUnwrappedInput_Py", setUnwrappedInput_C, METH_VARARGS, " "},
    {"setMinConnectedComponentFraction_Py", setMinConnectedComponentFraction_C, METH_VARARGS, " "},
    {"setConnectedComponentThreshold_Py", setConnectedComponentThreshold_C, METH_VARARGS, " "},
    {"setIntFileFormat_Py", setIntFileFormat_C, METH_VARARGS, " "},
    {"setCorFileFormat_Py", setCorFileFormat_C, METH_VARARGS, " "},
    {"setUnwFileFormat_Py", setUnwFileFormat_C, METH_VARARGS, " "},
    {"setMagnitude_Py", setMagnitude_C, METH_VARARGS, " "},
    {NULL,NULL,0,NULL}
};

#endif
// end of file
