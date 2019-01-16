//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                  (C) 2009-2010  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#include <Python.h>
#include "powlooksmodule.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
#include "ImageAccessor.h"
using namespace std;

static  const char * const __doc__ = "Python extension for powlooks.F";

PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "powlooks",
    // module documentation string
    __doc__,
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    powlooks_methods,
};

// initialization function for the module
// *must* be called PyInit_formslc
PyMODINIT_FUNC
PyInit_powlooks()
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


PyObject * powlooks_C(PyObject* self, PyObject* args)
{
    char * inputImage;
    char * outputImage;
    int lenIn = 0;
    int lenOut = 0;
    char enIn = 'n';
    char enOut = 'n';
    string inmode = "read";
    string typeIn = "CFLOAT";
    string typeOut = "FLOAT";
    string outmode = "write";
    int la = 0;//across = range
    int ld = 0;//down = azimuth
    int na = 0;//width get from image object
    int nd = -1;//lenght. get as optional argument just in case don;t want to do all the lines

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
    int sizeC = 2*na/la;
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

    vector<float>  b(na,0);
    vector<float> c(sizeC,0);
    vector<complex<float> > a(na*ld,0);
    int indX = 0;
    int indY = 0;
    int numEl = 0;
    for(int line = 0; line < nd; line += ld)
    {
        indY= line + 1;
        indX = 1;
        numEl = na*ld;
        IAIn.getSequentialElements((char *) &a[0],indY,indX,numEl);
        if(numEl < na*ld)//numEl at return is the number of elements actually read. if they differ then the eof is reached.
        {
            break;
        }
        for(int j = 0; j < na; ++j)
        {
            for(int i = 0; i < ld; ++i)
            {
                b[j] += real(a[j+i*na])*real(a[j+i*na]) + imag(a[j+i*na])*imag(a[j+i*na]);

            }
        }
        for(int j = 0; j < na/la; ++j)
        {
            for(int k = 0; k < la; ++k)
            {
                c[2*j] += b[j*la+k];
            }
        }
        IAOut.setLineSequential((char *) &c[0]);
        b.assign(na,0);
        c.assign(sizeC,0);
    }
    IAIn.finalizeImageAccessor();
    IAOut.finalizeImageAccessor();

    return Py_BuildValue("i", 0);
}
