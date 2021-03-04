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
#include "image_sio.h"
#include "siocomplex.h"
#include "lib_functions.h"
#include "cfft1d_jpl_c.h"
#define clip127(A) ( ((A) > 127) ? 127 : (((A) < 0) ? 0 : A) )

#include "ALOS_fbs2fbdmodule.h"

const* __doc__ =  "Python extension for ALOS_fbs2fbd.c";

static struct PyModuleDef moduledef = {
    // header
    PyModuleDef_HEAD_INIT,
    // name of the module
    "ALOS_fbs2fbd",
    // module documentation string
    "Python extension for ALOS_fbs2fbd.c",
    // size of the per-interpreter state of the module;
    // -1 if this state is global
    -1,
    ALOS_fbs2fbd_methods,
};

// initialization function for the module
// *must* be called PyInit_ALOS_fbs2fbd
PyMODINIT_FUNC
PyInit_ALOS_fbs2fbd()
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

//globals to be used by setters
int ALOS_fbs2fbd_bytes_per_line = 0;
int ALOS_fbs2fbd_good_bytes = 0;
int ALOS_fbs2fbd_first_sample = 0;
int ALOS_fbs2fbd_number_lines = 0;
double ALOS_fbs2fbd_inphase = 0;
double ALOS_fbs2fbd_quadrature = 0;
char * ALOS_fbs2fbd_InputFilename;
char * ALOS_fbs2fbd_OutputFilename;
PyObject * ALOS_fbs2fbd_C(PyObject* self, PyObject* args)
{
    FILE        *prmfile, *datafile, *prmout, *dataout;
    unsigned char       *indata, *outdata;
    fcomplex *cin, *cout;
    float rtest, itest;
    int i, j, k, np, nffti, nffto, i0, headsize;
    int         ibufsize, obufsize, fbdsamp, fbssamp;
    int dir, n4;
    size_t      n;
    struct      PRM r;

    r.good_bytes = ALOS_fbs2fbd_good_bytes;
    r.first_sample = ALOS_fbs2fbd_first_sample;
    r.num_lines = ALOS_fbs2fbd_number_lines;
    r.xmi = ALOS_fbs2fbd_inphase;
    r.xmq = ALOS_fbs2fbd_quadrature;
    r.bytes_per_line = ALOS_fbs2fbd_bytes_per_line;
    printf("ALOS_fbs2fbdmodule.c: r.good_bytes = %d\n", r.good_bytes);
    printf("ALOS_fbs2fbdmodule.c: r.first_sample = %d\n", r.first_sample);
    printf("ALOS_fbs2fbdmodule.c: r.num_lines = %d\n", r.num_lines);
    printf("ALOS_fbs2fbdmodule.c: r.xmi = %f\n", r.xmi);
    printf("ALOS_fbs2fbdmodule.c: r.xmq = %f\n", r.xmq);
    printf("ALOS_fbs2fbdmodule.c: r.bytes_per_line = %d\n", r.bytes_per_line);
    printf("ALOS_fbs2fbdmodule.c: ALOS_fbs2fbd_InputFilename = %s\n", ALOS_fbs2fbd_InputFilename);
    printf("ALOS_fbs2fbdmodule.c: ALOS_fbs2fbd_OutputFilename = %s\n", ALOS_fbs2fbd_OutputFilename);

    /* open input raw data file */
    if ((datafile = fopen(ALOS_fbs2fbd_InputFilename,"r")) == NULL)
    {

        fprintf(stderr,"Can't open %s \n", ALOS_fbs2fbd_InputFilename);
        exit(1);
    }
    /* open output file for single look complex  image */
    if ((dataout = fopen(ALOS_fbs2fbd_OutputFilename,"w")) == NULL)
    {
        fprintf(stderr,"Can't open %s \n",ALOS_fbs2fbd_OutputFilename);
        exit(1);
    }

    ibufsize = r.bytes_per_line;
    if((indata = (unsigned char *) malloc(ibufsize*sizeof(unsigned char))) ==
        NULL){
        fprintf(stderr, "Sorry, couldn't allocate memory for input indata.\n");
        exit(-1);
    }
    fbssamp = r.good_bytes/2 - r.first_sample;
    fbdsamp = fbssamp/2;
    headsize = 2 * r.first_sample;
    obufsize = 2*(fbdsamp+r.first_sample);
    if((outdata = (unsigned char *) malloc(obufsize*sizeof(unsigned char))) ==
        NULL){
        fprintf(stderr,
            "Sorry, couldn't allocate memory for output outdata.\n");
        exit(-1);
    }


    /* find best length of fft (use power of two) for both input and output  */
    nffti = find_fft_length(fbssamp);
    nffto = find_fft_length(fbdsamp);
    printf("ALOS_fbs2fbdmodule.c: fbssamp %d fbdsamp %d \n",fbssamp,fbdsamp);
    printf("ALOS_fbs2fbdmodule.c: nffti %d nffto %d \n",nffti,nffto);
    if (debug) fprintf(stderr," nffti %d nffto %d \n",nffti,nffto);

    /* allocate the memory for the complex arrays */
    if((cin = (fcomplex*) malloc(nffti*sizeof(fcomplex))) == NULL){
        fprintf(stderr,"Sorry, couldn't allocate memory for fbd \n");
        exit(-1);
    }


    if((cout = (fcomplex *) malloc(nffto*sizeof(fcomplex))) == NULL){
        fprintf(stderr,"Sorry, couldn't allocate memory for fbs \n");
        exit(-1);
    }

    //Initialize FFT plans
    j = nffti; i=0;
    cfft1d_jpl(&j, (float*) cin, &i);
    j = nffto; i=0;
    cfft1d_jpl(&j, (float*) cout, &i);


    /* read and write the input and output raw files */
    for (k=0; k< r.num_lines; k++) {
        for(i=0; i<nffti;i++)
        {
            cin[i].r =  0.0;
            cin[i].i =  0.0;
        }
        for(i=0;i<nffto;i++)
        {
            cout[i].r=0.0;
            cout[i].i=0.0;
        }

        fread((void *)indata,sizeof(unsigned char),ibufsize,datafile);
        fwrite((void *)indata,sizeof(unsigned char),headsize,dataout);

        /* fill the complex array with complex indata */
        for (j=0; j< fbssamp; j++) {
            i = j + r.first_sample;

            if((((int) indata[2*i]) != NULL_DATA) && (((int) indata[2*i+1]) !=
                NULL_DATA)){
                cin[j].r = (float)(indata[2*i]-r.xmi);
                cin[j].i = (float)(indata[2*i+1]-r.xmq);
            }
        }


        /*****Forward FFT ***/
        dir = -1; i=nffti;
        cfft1d_jpl(&i, (float*) cin, &dir);

        n4 = nffti/4;
        for(i=0; i<n4;i++)
        {
            cout[i].r = 0.5*cin[i].r;
            cout[i].i = 0.5*cin[i].i;
            cout[i+n4].r = 0.5*cin[i+3*n4].r;
            cout[i+n4].i = 0.5*cin[i+3*n4].i;
        }

        /*****Inverse FFT*****/
        dir = 1; i=nffto;
        cfft1d_jpl(&i, (float*) cout, &dir);
        
        for(i=0; i<nffto;i++)
        {
            cout[i].r = cout[i].r/(1.0*nffto);
            cout[i].i = cout[i].i/(1.0*nffto);
        }

        /* convert the complex back to bytes  */

        for (j=0; j< fbdsamp; j++) {
            i = j + r.first_sample;

            /* increase dynamic range by 2 and set the mean value to 63.5 */
            rtest = rintf(cout[j].r+r.xmi);
            itest = rintf(cout[j].i+r.xmq);

            /* sometimes the range can exceed 0-127 so
               clip the numbers to be in the correct range */
            outdata[2*i] = (unsigned char) clip127(rtest);
            outdata[2*i+1] = (unsigned char) clip127(itest);
        }
        fwrite((void *)(outdata+headsize), sizeof(unsigned char),
            obufsize-headsize, dataout);
    }

    //Destroy plans
    j = nffti; i = 2;
    cfft1d_jpl(&j, (float*) cin, &i);

    j = nffto; i=2;
    cfft1d_jpl(&j, (float*) cout, &i);

    fclose(datafile);
    fclose(dataout);
    free(cin);
    free(cout);
    free(outdata);
    free(indata);
    return Py_BuildValue("i", 0);
}
PyObject * setInputFilename_C(PyObject* self, PyObject* args)
{
    if(!PyArg_ParseTuple(args, "s", &ALOS_fbs2fbd_InputFilename))
    {
        return NULL;
    }

    return Py_BuildValue("i", 0);
}
PyObject * setOutputFilename_C(PyObject* self, PyObject* args)
{
    if(!PyArg_ParseTuple(args, "s", &ALOS_fbs2fbd_OutputFilename))
    {
        return NULL;
    }

    return Py_BuildValue("i", 0);
}
PyObject * setNumberGoodBytes_C(PyObject* self, PyObject* args)
{
    if(!PyArg_ParseTuple(args, "i", &ALOS_fbs2fbd_good_bytes))
    {
        return NULL;
    }
    return Py_BuildValue("i", 0);
}
PyObject * setNumberBytesPerLine_C(PyObject* self, PyObject* args)
{
    if(!PyArg_ParseTuple(args, "i", &ALOS_fbs2fbd_bytes_per_line))
    {
        return NULL;
    }
    return Py_BuildValue("i", 0);
}
PyObject * setNumberLines_C(PyObject* self, PyObject* args)
{
    if(!PyArg_ParseTuple(args, "i", &ALOS_fbs2fbd_number_lines))
    {
        return NULL;
    }
    return Py_BuildValue("i", 0);
}
PyObject * setFirstSample_C(PyObject* self, PyObject* args)
{
    if(!PyArg_ParseTuple(args, "i", &ALOS_fbs2fbd_first_sample))
    {
        return NULL;
    }
    return Py_BuildValue("i", 0);
}
PyObject * setInPhaseValue_C(PyObject* self, PyObject* args)
{
    if(!PyArg_ParseTuple(args, "d", &ALOS_fbs2fbd_inphase))
    {
        return NULL;
    }
    return Py_BuildValue("i", 0);
}
PyObject * setQuadratureValue_C(PyObject* self, PyObject* args)
{
    if(!PyArg_ParseTuple(args, "d", &ALOS_fbs2fbd_quadrature))
    {
        return NULL;
    }
    return Py_BuildValue("i", 0);
}
