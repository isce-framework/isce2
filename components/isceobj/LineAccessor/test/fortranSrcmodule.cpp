//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#include <Python.h>
#include "fortranSrcmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
using namespace std;
extern "C" void initfortranSrc()
{
 	//fortranSrc = module name as imported in python
	Py_InitModule3("fortranSrc", fortranSrc_methods, moduleDoc);
}

// interface function from python to fortran. when calling the function fortranSrc.testImageSetGet() the following function gets called in C++
PyObject * testImageSetGet_C(PyObject* self, PyObject* args) 
{
	uint64_t ptLAGet = 0;
	uint64_t ptLASet = 0;
	int choice = 0;
	//get the arguments passed to fortranSrc.testImageSetGet()
	if(!PyArg_ParseTuple(args, "KKi", &ptLAGet, &ptLASet, &choice)) 
	{
		return NULL;  
	}  
	// call the fortan subtoutine testImageSetGet
	testImageSetGet_f(&ptLAGet, &ptLASet, &choice);
	return Py_BuildValue("i",0);
}
