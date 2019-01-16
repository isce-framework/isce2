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





#include <Python.h>
#include "readOrbitPulseERSmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;
extern "C" void initreadOrbitPulseERS()
{
 	Py_InitModule3("readOrbitPulseERS", readOrbitPulseERS_methods, moduleDoc);
}
PyObject * readOrbitPulseERS_C(PyObject* self, PyObject* args) 
{
	readOrbitPulseERS_f();
	return Py_BuildValue("i", 0);
}
PyObject * setEncodedBinaryTimeCode_C(PyObject* self, PyObject* args) 
{
	uint64_t var;
	if(!PyArg_ParseTuple(args, "i", &var)) 
	{
		return NULL;  
	}  
	setEncodedBinaryTimeCode_f(&var);
	return Py_BuildValue("i", 0);
}
PyObject * setWidth_C(PyObject* self, PyObject* args) 
{
	int var;
	if(!PyArg_ParseTuple(args, "i", &var)) 
	{
		return NULL;  
	}  
	setWidth_f(&var);
	return Py_BuildValue("i", 0);
}
PyObject * setICUoffset_C(PyObject* self, PyObject* args) 
{
	int var;
	if(!PyArg_ParseTuple(args, "i", &var)) 
	{
		return NULL;  
	}  
	setICUoffset_f(&var);
	return Py_BuildValue("i", 0);
}
PyObject * setNumberLines_C(PyObject* self, PyObject* args) 
{
	int var;
	if(!PyArg_ParseTuple(args, "i", &var)) 
	{
		return NULL;  
	}  
	setNumberLines_f(&var);
	return Py_BuildValue("i", 0);
}
PyObject * setSatelliteUTC_C(PyObject* self, PyObject* args) 
{
	double var;
	if(!PyArg_ParseTuple(args, "d", &var)) 
	{
		return NULL;  
	}  
	setSatelliteUTC_f(&var);
	return Py_BuildValue("i", 0);
}
PyObject * setPRF_C(PyObject* self, PyObject* args) 
{
	double var;
	if(!PyArg_ParseTuple(args, "d", &var)) 
	{
		return NULL;  
	}  
	setPRF_f(&var);
	return Py_BuildValue("i", 0);
}
PyObject * setDeltaClock_C(PyObject* self, PyObject* args) 
{
	double var;
	if(!PyArg_ParseTuple(args, "d", &var)) 
	{
		return NULL;  
	}  
	setDeltaClock_f(&var);
	return Py_BuildValue("i", 0);
}
PyObject * getStartingTime_C(PyObject* self, PyObject* args) 
{
	double var;
	getStartingTime_f(&var);
	return Py_BuildValue("d",var);
}
