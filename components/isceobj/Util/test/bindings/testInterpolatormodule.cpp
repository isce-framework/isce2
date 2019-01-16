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




#include <Python.h>
#include "testInterpolatormodule.h"
using namespace std;

static const char * const __doc__ = "Python extension for image API data accessors";

PyModuleDef moduledef =
{
// header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "testInterpolator",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1, testInterpolator_methods, };

// initialization function for the module
// *must* be called PyInit_formslc
PyMODINIT_FUNC
PyInit_testInterpolator()
{
  // create the module using moduledef struct defined above
  PyObject * module = PyModule_Create(&moduledef);
  // check whether module creation succeeded and raise an exception if not
  if (!module)
  {
    return module;
  }
  // otherwise, we have an initialized module
  // and return the newly created module
  return module;
}

PyObject *
testInterpolator_C(PyObject* self, PyObject* args)
{
  uint64_t ptPoly2 = 0;
  uint64_t ptPoly1 = 0;

  if (!PyArg_ParseTuple(args, "KK", &ptPoly1, &ptPoly2))
  {
    return NULL;
  }

 testinterpolator_(&ptPoly1, &ptPoly2);

  return Py_BuildValue("i", 0);
}
