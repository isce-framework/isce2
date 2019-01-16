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





#ifndef cpxmag2rgmodule_h
#define cpxmag2rgmodule_h

#include <Python.h>
#include <stdint.h>
#include "cpxmag2rgmoduleFortTrans.h"

extern "C"
{
    void setStdWriter_f(uint64_t *);
    PyObject * setStdWriter_C(PyObject *, PyObject *);
        void cpxmag2rg_f(uint64_t *,uint64_t *,uint64_t *);
        PyObject * cpxmag2rg_C(PyObject *, PyObject *);
        void setLineLength_f(int *);
        PyObject * setLineLength_C(PyObject *, PyObject *);
        void setFileLength_f(int *);
        PyObject * setFileLength_C(PyObject *, PyObject *);
        void setAcOffset_f(int *);
        PyObject * setAcOffset_C(PyObject *, PyObject *);
        void setDnOffset_f(int *);
        PyObject * setDnOffset_C(PyObject *, PyObject *);

}

static char * moduleDoc = "module for cpxmag2rg.F";

static PyMethodDef cpxmag2rg_methods[] =
{
    {"setStdWriter_Py", setStdWriter_C, METH_VARARGS, " "},
        {"cpxmag2rg_Py", cpxmag2rg_C, METH_VARARGS, " "},
        {"setLineLength_Py", setLineLength_C, METH_VARARGS, " "},
        {"setFileLength_Py", setFileLength_C, METH_VARARGS, " "},
        {"setAcOffset_Py", setAcOffset_C, METH_VARARGS, " "},
        {"setDnOffset_Py", setDnOffset_C, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //cpxmag2rgmodule_h
