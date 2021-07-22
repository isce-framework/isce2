//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copyright 2009 California Institute of Technology. ALL RIGHTS RESERVED.
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

#ifndef alosmodule_h
#define alosmodule_h

#include <Python.h>
#include <stdint.h>
#include "image_sio.h"
#include "alosglobals.h"

extern "C"
{
    PyObject *alos_C(PyObject *self,PyObject *args);
    PyObject *alose_C(PyObject *self,PyObject *args);
    PyObject *createDictionaryOutput(struct PRM *prm,PyObject *dict);
    int ALOS_pre_process(struct PRM inputPRM, struct PRM *outputPRM,
        struct GLOBALS globals, int image_i);
}

static PyMethodDef alos_methods[]  =
{
    {"alos_Py",alos_C,METH_VARARGS," "},
    {"alose_Py",alose_C,METH_VARARGS," "},
    {NULL,NULL,0,NULL}
};

#endif
// end of file

