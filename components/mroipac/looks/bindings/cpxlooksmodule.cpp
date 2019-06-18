//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                  (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#include <Python.h>
#include "cpxlooksmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
#include "ImageAccessor.h"
using namespace std;

static const char * const __doc__ = "Python extension for cpxlooks.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "cpxlooks",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    cpxlooks_methods,
};

// initialization function for the module
// *must* be called PyInit_formslc
PyMODINIT_FUNC
PyInit_cpxlooks()
{
    // create the module using moduledef struct defined above
    PyObject * module = PyModule_Create(&moduledef);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return module;
    }
    // otherwise, we have an initialized module
    // and return the newly created module
    return module;
}


PyObject * cpxlooks_C(PyObject* self, PyObject* args)
{
    char * inputImage;
    char * outputImage;
    int lenIn = 0;
    int lenOut = 0;
    char enIn = 'n';
    char enOut = 'n';
    string inmode = "read";
    string typeIn = "CFLOAT";
    string typeOut = "CFLOAT";
    string outmode = "write";
    int la = 0;//across = range
    int ld = 0;//down = azimuth
    int na = 0;//width get from image object
    int nd = -1;//lenght. get as optional argument just in case don;t want to do all the lines
    float pa = 0;
    float pd = 0;
    PyObject * dictionary = NULL;
    //put explicity the mandatory args and in a dictionary the optionals
    if(!PyArg_ParseTuple(args, "s#s#iii|O", &inputImage,&lenIn,&outputImage,&lenOut,&na,&la,&ld,&dictionary))
    {
        return NULL;
    }
    if((dictionary != NULL))
    {
        PyObject * lengthPy = PyDict_GetItemString(dictionary,"LENGTH");
        if(lengthPy != NULL)
        {
            nd = (int) PyLong_AsLong(lengthPy);
        }
        PyObject * paPy = PyDict_GetItemString(dictionary,"PHASE_RANGE");
        if(paPy != NULL)
        {
            pa = (float) PyFloat_AsDouble(paPy);
        }
        PyObject * pdPy = PyDict_GetItemString(dictionary,"PHASE_AZIMUTH");
        if(pdPy != NULL)
        {
            pd = (float) PyFloat_AsDouble(pdPy);
        }
        PyObject * enInPy = PyDict_GetItemString(dictionary,"INPUT_ENDIANNESS");
        if(enInPy != NULL)
        {
            char * inEndian = PyBytes_AsString(enInPy);
            enIn = inEndian[0];
        }
        PyObject * enOutPy = PyDict_GetItemString(dictionary,"OUTPUT_ENDIANNESS");
        if(enOutPy != NULL)
        {
            char * outEndian = PyBytes_AsString(enOutPy);
            enOut = outEndian[0];
        }

    }
    int sizeC = na/la;
    string infile = inputImage;
    string outfile = outputImage;
    ImageAccessor IAIn;
    ImageAccessor IAOut;
    if( enIn == 'n')//use as default the machine endianness
    {
        enIn = IAIn.getMachineEndianness();
    }
    if( enOut == 'n')//use as default the machine endianness
    {
        enOut = IAOut.getMachineEndianness();
    }
    IAIn.initImageAccessor(infile,inmode,enIn,typeIn,na);
    IAOut.initImageAccessor(outfile,outmode,enOut,typeOut,sizeC);
    if(nd == -1)//use as default the whole file
    {
        nd = IAIn.getFileLength();
    }
    complex<float> pha(cos(pa),sin(pa));
    complex<float> phd(cos(pd),sin(pd));
    vector<complex<float> > b(na,0);
    vector<complex<float> > b1(sizeC,0);
    vector<complex<float> > a(na,0);
    bool eofReached = false;
    for(int line = 0; line < nd; line += ld)
    {
        for(int i = 0; i < ld; ++i)
        {
            int lineToGet = (line + i + 1);
            IAIn.getLine((char *) &a[0], lineToGet);

            if(lineToGet == -1)
            {
                eofReached = true;
                break;
            }
            for(int j = 0; j < na; ++j)
            {
                b[j] = b[j] + a[j]*pow(pha,j+1.0f)*pow(phd,lineToGet*1.0f);
            }
        }
        if(eofReached)
        {
            break;
        }
        int jpix = 0;
        for(int j = 0; j < na; j += la)
        {
            complex<float> sum(0,0);
            for(int k = 0; k < la; ++k)
            {
                sum = sum + b[j+k];
            }
            b1[jpix] = sum;
            ++jpix;
        }
        IAOut.setLineSequential((char *) &b1[0]);
        b.assign(na,complex<float>(0,0));
    }
    IAIn.finalizeImageAccessor();
    IAOut.finalizeImageAccessor();
    return Py_BuildValue("i", 0);
}
