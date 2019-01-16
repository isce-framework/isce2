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



#include <iostream>
#include <limits>
#include <cmath>
#include "PhaseExtractor.h"

using namespace std;

void PhaseExtractor::extract()
{
    int eof = 1;
    //loop through the image. The DataType size is ImageIn->DataSizeIn.
    int width = ImageIn->getWidth();  
    int bands = ImageIn->getBands();  
    int sizeIn = ImageIn->getSizeIn();  
    char * bufIn = new char[width*sizeIn*bands];
    char * bufOut = new char[width*(sizeIn/2)*bands];
    int cnt = StartLine;
    ImageIn->initSequentialAccessor(StartLine);
    while(true)
    {
        eof = ImageIn->getLineSequential(bufIn);
        ++cnt;
        if(eof < 0 || cnt > EndLine)
        {
            break;
        }
        for(int i = 0; i < width*bands; ++i)
        {
            //we don't know a priori the data type so at this point try to figure the right casting
            if(sizeIn/2 == sizeof(float))
            {
                float * x = (float *) &bufIn[i*sizeIn];
                float * y = (float *) &bufIn[i*sizeIn + sizeIn/2];
                float phase = 0;
                if(abs((*y)) < numeric_limits<float>::min() && abs((*x)) < numeric_limits<float>::min())//assume y=0/x=0. not defined but return pi/2 anyway
                {
                    
                    phase = atan2(1,0);
                }
                else
                {
                    phase = atan2((*y),(*x));
                }
                (* (float *) &bufOut[i*sizeIn/2]) = phase;
            }   
            else if(sizeIn/2 == sizeof(double))
            {
                double * x = (double *) &bufIn[i*sizeIn];
                double * y = (double *) &bufIn[i*sizeIn + sizeIn/2];
                double phase = 0;
                if(abs((*y)) < numeric_limits<double>::min() && abs(*(x)) < numeric_limits<double>::min())//assume y=0/x=0. not defined but return pi/2 anyway
                {
                    phase = atan2(1,0);
                }
                else
                {
                    phase = atan2((*y),(*x));
                }
                (* (double *) &bufOut[i*sizeIn/2]) = phase;
            }
            else
            {
                cout << "Datatype size not supported." << endl;
                ERR_MESSAGE;
            }
        }
        ImageOut->setLineSequential(bufOut);
    }
    delete [] bufIn;
    delete [] bufOut;
}

