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
#include "AccessorFactory.h"
#include "DataAccessormodule.h"
#include <iostream>
#include <fstream>
#include <string>
#include <complex>
#include <stdint.h>
#include <cstdio>
using namespace std;

static const char * const __doc__ = "Python extension for image API data accessors";

PyModuleDef moduledef =
{
// header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "DataAccessor",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1, DataAccessor_methods, };

// initialization function for the module
// *must* be called PyInit_formslc
PyMODINIT_FUNC
PyInit_DataAccessor()
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
createPolyAccessor_C(PyObject* self, PyObject* args)
{
  string polytype;
  char * polytypeCh;
  uint64_t ptPoly = 0;
  int width = 0;
  int length = 0;
  int dataSize = 0;
  if (!PyArg_ParseTuple(args, "Ksiii", &ptPoly, &polytypeCh, &width, &length,
      &dataSize))
  {
    return NULL;
  }
  polytype = polytypeCh;
  AccessorFactory * AF = new AccessorFactory();
  uint64_t ptAccessor = (uint64_t) AF->createAccessor((void *) ptPoly, polytype,
      width, length, dataSize);

  return Py_BuildValue("KK", ptAccessor, (uint64_t) AF);
}
string getString(PyObject * key)
{
   PyObject * utf8string;
   utf8string = PyUnicode_AsUTF8String (key);
   string ret = PyBytes_AsString(utf8string);
   Py_XDECREF(utf8string);
   return ret;
}
PyObject *
createAccessor_C(PyObject* self, PyObject* args)
{
  string filename;
  char * filenameCh;
  string filemode;
  char * filemodeCh;
  string scheme;
  char * schemeCh;
  string caster;
  char * casterCh;
  int size = 0;
  int bands = 0;
  int width = 0;
  int len = 0;
  // In order to allow multiple type of casters that might need different initialization pass
  // an optional dictionary which will have the extra parameters to create the correct accessor
  PyObject * dict = NULL;
  if (!PyArg_ParseTuple(args, "ssiiis|sO", &filenameCh, &filemodeCh, &size,
      &bands, &width, &schemeCh, &casterCh, &dict))
  {
    return NULL;
  }
  filename = filenameCh;
  filemode = filemodeCh;
  scheme = schemeCh;
  AccessorFactory * AF = new AccessorFactory();
  uint64_t ptDataAccessor = 0;

  if (!PyDict_Check(dict))
  {
    cout << "Error in file " << __FILE__ << " at line " << __LINE__
        << ". Expecting a dictionary type object" << endl;
    exit(1);
  }
  if (casterCh[0] == '\0')
  {
    try
    {
      ptDataAccessor = (uint64_t) AF->createAccessor(filename, filemode, size,
          bands, width, scheme);
    }
    catch(const std::exception& e)
    {
      PyErr_SetString(PyExc_OSError, e.what());
      return NULL;
    }

  }
  else if (casterCh[0] != '\0' && PyDict_Size(dict) == 0)
  {

    caster = casterCh;
    ptDataAccessor = (uint64_t) AF->createAccessor(filename, filemode, size,
        bands, width, scheme, caster);
  }
  else if (casterCh[0] != '\0' && PyDict_Size(dict) != 0)
  {

    PyObject * pyobj = PyDict_GetItemString(dict, "type");
    if (PyErr_Occurred() != NULL)
    {
      cout << "Error in file " << __FILE__ << " at line " << __LINE__
          << ". Error reading caster type" << endl;
      exit(1);
    }


    string type_s = getString(pyobj);

    if (type_s == "iq")
    {

      pyobj = PyDict_GetItemString(dict, "xmi");
      float xmi = PyFloat_AsDouble(pyobj);

      pyobj = PyDict_GetItemString(dict, "xmq");
      float xmq = PyFloat_AsDouble(pyobj);

      pyobj = PyDict_GetItemString(dict, "iqflip");
      long iqflip = PyLong_AsLong(pyobj);

      caster = casterCh;

      ptDataAccessor = (uint64_t) AF->createAccessor(filename, filemode, size,
          bands, width, scheme, caster, xmi, xmq, (int) iqflip);

    }
  }
  else
  {
    cout << "Error in file " << __FILE__ << " at line " << __LINE__
        << ". Cannot parse inputs " << endl;
    exit(1);
  }
  return Py_BuildValue("KK", ptDataAccessor, (uint64_t) AF);
}
PyObject *
finalizeAccessor_C(PyObject* self, PyObject* args)
{
  uint64_t ptDataAccessor = 0;
  uint64_t ptFactory = 0;
  if (!PyArg_ParseTuple(args, "KK", &ptDataAccessor, &ptFactory))
  {
    return NULL;
  }
  AccessorFactory * tmp = (AccessorFactory *) (ptFactory);
  tmp->finalize((DataAccessor *) (ptDataAccessor));

  delete tmp;
  return Py_BuildValue("i", 0);
}
PyObject *
getFileLength_C(PyObject* self, PyObject* args)
{
  uint64_t ptDataAccessor = 0;
  if (!PyArg_ParseTuple(args, "K", &ptDataAccessor))
  {
    return NULL;
  }
  int length =
      ((DataAccessor *) (ptDataAccessor))->getInterleavedAccessor()->getFileLength();
  return Py_BuildValue("i", length);
}
PyObject *
rewind_C(PyObject* self, PyObject* args)
{
  uint64_t ptDataAccessor = 0;
  if (!PyArg_ParseTuple(args, "K", &ptDataAccessor))
  {
    return NULL;
  }
  DataAccessor * tmp = (DataAccessor *) (ptDataAccessor);
  tmp->rewindAccessor();
  return Py_BuildValue("i", 0);
}
PyObject *
createFile_C(PyObject* self, PyObject* args)
{
  uint64_t ptDataAccessor = 0;
  int length = 0;
  if (!PyArg_ParseTuple(args, "Ki", &ptDataAccessor, &length))
  {
    return NULL;
  }
  DataAccessor * tmp = (DataAccessor *) (ptDataAccessor);
  tmp->createFile(length);
  return Py_BuildValue("i", 0);
}
PyObject *
getTypeSize_C(PyObject* self, PyObject* args)
{
  char * typeCh;
  string type;
  if (!PyArg_ParseTuple(args, "s", &typeCh))
  {
    return NULL;
  }
  type = typeCh;
  int retVal = -1;
  if (type == "byte" || type == "BYTE" || type == "char" || type == "CHAR")
  {
    retVal = sizeof(char);
  }
  else if (type == "short" || type == "SHORT")
  {
    retVal = sizeof(short);
  }
  else if (type == "int" || type == "INT")
  {
    retVal = sizeof(int);
  }
  else if (type == "long" || type == "LONG")
  {
    retVal = sizeof(long);
  }
  else if (type == "float" || type == "FLOAT")
  {
    retVal = sizeof(float);
  }
  else if (type == "double" || type == "DOUBLE")
  {
    retVal = sizeof(double);
  }
  else if (type == "cbyte" || type == "CBYTE" || type == "cchar"
      || type == "CCHAR" || type == "ciqbyte" || type == "CIQBYTE")
  {
    retVal = sizeof(complex<char> );
  }
  else if (type == "cshort" || type == "CSHORT")
  {
    retVal = sizeof(complex<short> );
  }
  else if (type == "cint" || type == "CINT")
  {
    retVal = sizeof(complex<int> );
  }
  else if (type == "clong" || type == "CLONG")
  {
    retVal = sizeof(complex<long> );
  }
  else if (type == "cfloat" || type == "CFLOAT")
  {
    retVal = sizeof(complex<float> );
  }
  else if (type == "cdouble" || type == "CDOUBLE")
  {
    retVal = sizeof(complex<double> );
  }
  else
  {
    cout << "Error. Unrecognized data type " << type << endl;

    ERR_MESSAGE
    ;
  }
  return Py_BuildValue("i", retVal);
}
