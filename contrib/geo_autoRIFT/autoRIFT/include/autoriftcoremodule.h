/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright 2019 California Institute of Technology. ALL RIGHTS RESERVED.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * United States Government Sponsorship acknowledged. This software is subject to
 * U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
 * (No [Export] License Required except when exporting to an embargoed country,
 * end user, or in support of a prohibited end use). By downloading this software,
 * the user agrees to comply with all applicable U.S. export laws and regulations.
 * The user has the responsibility to obtain export licenses, or other export
 * authority as may be required before exporting this software to any 'EAR99'
 * embargoed foreign country or citizen of those countries.
 *
 * Author: Yang Lei
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */


#ifndef autoriftcoremodule_h
#define autoriftcoremodule_h

#include <Python.h>
#include <stdint.h>


extern "C"
{
        PyObject * createAutoRiftCore(PyObject*, PyObject*);
        PyObject * destroyAutoRiftCore(PyObject*, PyObject*);
        PyObject * arPixDisp_u(PyObject *, PyObject *);
        PyObject * arSubPixDisp_u(PyObject *, PyObject *);
        PyObject * arPixDisp_s(PyObject *, PyObject *);
        PyObject * arSubPixDisp_s(PyObject *, PyObject *);
}

static PyMethodDef autoriftcore_methods[] =
{
        {"createAutoRiftCore_Py", createAutoRiftCore, METH_VARARGS, " "},
        {"destroyAutoRiftCore_Py", destroyAutoRiftCore, METH_VARARGS, " "},
        {"arPixDisp_u_Py", arPixDisp_u, METH_VARARGS, " "},
        {"arSubPixDisp_u_Py", arSubPixDisp_u, METH_VARARGS, " "},
        {"arPixDisp_s_Py", arPixDisp_s, METH_VARARGS, " "},
        {"arSubPixDisp_s_Py", arSubPixDisp_s, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //autoRiftCoremodule_h

