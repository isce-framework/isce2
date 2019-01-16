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





#ifndef readOrbitPulseERSmodule_h
#define readOrbitPulseERSmodule_h

#include <Python.h>
#include <stdint.h>
#include "readOrbitPulseERSmoduleFortTrans.h"

extern "C"
{
        void readOrbitPulseERS_f();
        PyObject * readOrbitPulseERS_C(PyObject *, PyObject *);
        void setEncodedBinaryTimeCode_f(uint64_t *);
        PyObject * setEncodedBinaryTimeCode_C(PyObject *, PyObject *);
        void setWidth_f(int *);
        PyObject * setWidth_C(PyObject *, PyObject *);
        void setICUoffset_f(int *);
        PyObject * setICUoffset_C(PyObject *, PyObject *);
        void setNumberLines_f(int *);
        PyObject * setNumberLines_C(PyObject *, PyObject *);
        void setSatelliteUTC_f(double *);
        PyObject * setSatelliteUTC_C(PyObject *, PyObject *);
        void setPRF_f(double *);
        PyObject * setPRF_C(PyObject *, PyObject *);
        void setDeltaClock_f(double *);
        PyObject * setDeltaClock_C(PyObject *, PyObject *);
        void getStartingTime_f(double *);
        PyObject * getStartingTime_C(PyObject *, PyObject *);

}

static char * moduleDoc = "module for readOrbitPulseERS.F";

static PyMethodDef readOrbitPulseERS_methods[] =
{
        {"readOrbitPulseERS_Py", readOrbitPulseERS_C, METH_VARARGS, " "},
        {"setEncodedBinaryTimeCode_Py", setEncodedBinaryTimeCode_C, METH_VARARGS, " "},
        {"setWidth_Py", setWidth_C, METH_VARARGS, " "},
        {"setICUoffset_Py", setICUoffset_C, METH_VARARGS, " "},
        {"setNumberLines_Py", setNumberLines_C, METH_VARARGS, " "},
        {"setSatelliteUTC_Py", setSatelliteUTC_C, METH_VARARGS, " "},
        {"setPRF_Py", setPRF_C, METH_VARARGS, " "},
        {"setDeltaClock_Py", setDeltaClock_C, METH_VARARGS, " "},
        {"getStartingTime_Py", getStartingTime_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //readOrbitPulseERSmodule_h
