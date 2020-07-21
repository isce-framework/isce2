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






#include <Python.h>
#include <string>
//#include "autoriftcore.h"
#include "autoriftcoremodule.h"


#include "stdio.h"
#include "iostream"
#include "numpy/arrayobject.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"


using namespace cv;
using namespace std;

struct autoRiftCore{
//  This empty structure "autoRiftCore" in C++ is assgined to "self._autoriftcore" in python, which can take a set of variables in this file (declare here or in "autoriftcore.h" and set states below). For example,
//    ((autoRiftCore*)(ptr))->widC = widC;
//    ((autoRiftCore*)(ptr))->arPixDisp()
//  If taking all the variables here in the structure, the complicated computation can be performed in another C++ file, "autoriftcore.cpp" (that includes functions like void autoRiftCore::arPixDisp()).
};


static const char * const __doc__ = "Python extension for autoriftcore";


PyModuleDef moduledef = {
    //header
    PyModuleDef_HEAD_INIT,
    //name of the module
    "autoriftcore",
    //module documentation string
    __doc__,
    //size of the per-interpreter state of the module;
    -1,
    autoriftcore_methods,
};

//Initialization function for the module
PyMODINIT_FUNC
PyInit_autoriftcore()
{
    PyObject* module = PyModule_Create(&moduledef);
    if (!module)
    {
        return module;
    }
    return module;
}

PyObject* createAutoRiftCore(PyObject* self, PyObject *args)
{
    autoRiftCore* ptr = new autoRiftCore;
    return Py_BuildValue("K", (uint64_t) ptr);
}

PyObject* destroyAutoRiftCore(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    if (!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }

    if (((autoRiftCore*)(ptr))!=NULL)
    {
        delete ((autoRiftCore*)(ptr));
    }
    return Py_BuildValue("i", 0);
}



PyObject* arPixDisp_u(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    PyArrayObject *ChipI, *RefI;
    int widC, lenC;
    int widR, lenR;
    if (!PyArg_ParseTuple(args, "KiiOiiO", &ptr, &widC, &lenC, &ChipI, &widR, &lenR, &RefI))
    {
        return NULL;
    }

    uint8_t my_arrC[widC * lenC];
    
    for(int i=0; i<(widC*lenC); i++){
        my_arrC[i] = (*(uint8_t *)PyArray_GETPTR1(ChipI,i));
    }
    
    uint8_t my_arrR[widR * lenR];
    
    for(int i=0; i<(widR*lenR); i++){
        my_arrR[i] = (*(uint8_t *)PyArray_GETPTR1(RefI,i));
    }
    

    cv::Mat my_imgC = cv::Mat(cv::Size(widC, lenC), CV_8UC1, &my_arrC);
    cv::Mat my_imgR = cv::Mat(cv::Size(widR, lenR), CV_8UC1, &my_arrR);
    
    int result_cols =  widR - widC + 1;
    int result_rows = lenR - lenC + 1;
    
    cv::Mat result;
    result.create( result_rows, result_cols, CV_32FC1 );
    
    cv::matchTemplate( my_imgR, my_imgC, result, CV_TM_CCOEFF_NORMED );
    
    cv::Point maxLoc;
    cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
    
    double x = maxLoc.x;
    double y = maxLoc.y;
    
    
    return Py_BuildValue("dd", x, y);
}





PyObject* arSubPixDisp_u(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    PyArrayObject *ChipI, *RefI;
    int widC, lenC;
    int widR, lenR;
    int overSampleNC;
    if (!PyArg_ParseTuple(args, "KiiOiiOi", &ptr, &widC, &lenC, &ChipI, &widR, &lenR, &RefI, &overSampleNC))
    {
        return NULL;
    }
    
    uint8_t my_arrC[widC * lenC];
    
    for(int i=0; i<(widC*lenC); i++){
        my_arrC[i] = (*(uint8_t *)PyArray_GETPTR1(ChipI,i));
    }
    
    uint8_t my_arrR[widR * lenR];
    
    for(int i=0; i<(widR*lenR); i++){
        my_arrR[i] = (*(uint8_t *)PyArray_GETPTR1(RefI,i));
    }
    
    
    cv::Mat my_imgC = cv::Mat(cv::Size(widC, lenC), CV_8UC1, &my_arrC);
    cv::Mat my_imgR = cv::Mat(cv::Size(widR, lenR), CV_8UC1, &my_arrR);
    
    int result_cols =  widR - widC + 1;
    int result_rows = lenR - lenC + 1;
    
    cv::Mat result;
    result.create( result_rows, result_cols, CV_32FC1 );
    
    cv::matchTemplate( my_imgR, my_imgC, result, CV_TM_CCOEFF_NORMED );
    
    cv::Point maxLoc;
    cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
    
    
    // refine the offset at the sub-pixel level using image upsampling (pyramid algorithm): extract 5x5 small image at the coarse offset location
    int x_start, y_start, x_count, y_count;
    
    x_start = cv::max(maxLoc.x-2, 0);
    x_start = cv::min(x_start, result_cols-5);
    x_count = 5;
    
    y_start = cv::max(maxLoc.y-2, 0);
    y_start = cv::min(y_start, result_rows-5);
    y_count = 5;
    
    
    cv::Mat result_small (result, cv::Rect(x_start, y_start, x_count, y_count));
    
    int cols = result_small.cols;
    int rows = result_small.rows;
    int overSampleFlag = 1;
    cv::Mat predecessor_small = result_small;
    cv::Mat foo;
    
    while (overSampleFlag < overSampleNC){
        cols *= 2;
        rows *= 2;
        overSampleFlag *= 2;
        foo.create(cols, rows, CV_32FC1);
        cv::pyrUp(predecessor_small, foo, cv::Size(cols, rows));
        predecessor_small = foo;
    }
    
    cv::Point maxLoc_small;
    cv::minMaxLoc(foo, NULL, NULL, NULL, &maxLoc_small);
    
    
    double x = ((maxLoc_small.x + 0.0)/overSampleNC + x_start);
    double y = ((maxLoc_small.y + 0.0)/overSampleNC + y_start);
    
    
    return Py_BuildValue("dd", x, y);
}




PyObject* arPixDisp_s(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    PyArrayObject *ChipI, *RefI;
    int widC, lenC;
    int widR, lenR;
    if (!PyArg_ParseTuple(args, "KiiOiiO", &ptr, &widC, &lenC, &ChipI, &widR, &lenR, &RefI))
    {
        return NULL;
    }
    
    float my_arrC[widC * lenC];
    
    for(int i=0; i<(widC*lenC); i++){
        my_arrC[i] = (*(float *)PyArray_GETPTR1(ChipI,i));
    }
    
    float my_arrR[widR * lenR];
    
    for(int i=0; i<(widR*lenR); i++){
        my_arrR[i] = (*(float *)PyArray_GETPTR1(RefI,i));
    }
    
    
    cv::Mat my_imgC = cv::Mat(cv::Size(widC, lenC), CV_32FC1, &my_arrC);
    cv::Mat my_imgR = cv::Mat(cv::Size(widR, lenR), CV_32FC1, &my_arrR);
    
    int result_cols =  widR - widC + 1;
    int result_rows = lenR - lenC + 1;
    
    cv::Mat result;
    result.create( result_rows, result_cols, CV_32FC1 );
    
    cv::matchTemplate( my_imgR, my_imgC, result, CV_TM_CCORR_NORMED );
    
    cv::Point maxLoc;
    cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
    
    double x = maxLoc.x;
    double y = maxLoc.y;
    
    
    return Py_BuildValue("dd", x, y);
}





PyObject* arSubPixDisp_s(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    PyArrayObject *ChipI, *RefI;
    int widC, lenC;
    int widR, lenR;
    int overSampleNC;
    if (!PyArg_ParseTuple(args, "KiiOiiOi", &ptr, &widC, &lenC, &ChipI, &widR, &lenR, &RefI, &overSampleNC))
    {
        return NULL;
    }
    
    float my_arrC[widC * lenC];
    
    for(int i=0; i<(widC*lenC); i++){
        my_arrC[i] = (*(float *)PyArray_GETPTR1(ChipI,i));
    }
    
    float my_arrR[widR * lenR];
    
    for(int i=0; i<(widR*lenR); i++){
        my_arrR[i] = (*(float *)PyArray_GETPTR1(RefI,i));
    }
    
    
    cv::Mat my_imgC = cv::Mat(cv::Size(widC, lenC), CV_32FC1, &my_arrC);
    cv::Mat my_imgR = cv::Mat(cv::Size(widR, lenR), CV_32FC1, &my_arrR);
    
    int result_cols =  widR - widC + 1;
    int result_rows = lenR - lenC + 1;
    
    cv::Mat result;
    result.create( result_rows, result_cols, CV_32FC1 );
    
    cv::matchTemplate( my_imgR, my_imgC, result, CV_TM_CCORR_NORMED );
    
    cv::Point maxLoc;
    cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
    
    
    // refine the offset at the sub-pixel level using image upsampling (pyramid algorithm): extract 5x5 small image at the coarse offset location
    int x_start, y_start, x_count, y_count;
    
    x_start = cv::max(maxLoc.x-2, 0);
    x_start = cv::min(x_start, result_cols-5);
    x_count = 5;
    
    y_start = cv::max(maxLoc.y-2, 0);
    y_start = cv::min(y_start, result_rows-5);
    y_count = 5;
    
    
    cv::Mat result_small (result, cv::Rect(x_start, y_start, x_count, y_count));
    
    int cols = result_small.cols;
    int rows = result_small.rows;
    int overSampleFlag = 1;
    cv::Mat predecessor_small = result_small;
    cv::Mat foo;
    
    while (overSampleFlag < overSampleNC){
        cols *= 2;
        rows *= 2;
        overSampleFlag *= 2;
        foo.create(cols, rows, CV_32FC1);
        cv::pyrUp(predecessor_small, foo, cv::Size(cols, rows));
        predecessor_small = foo;
    }
    
    cv::Point maxLoc_small;
    cv::minMaxLoc(foo, NULL, NULL, NULL, &maxLoc_small);
    
    
    double x = ((maxLoc_small.x + 0.0)/overSampleNC + x_start);
    double y = ((maxLoc_small.y + 0.0)/overSampleNC + y_start);
    
    
    
    return Py_BuildValue("dd", x, y);
}
