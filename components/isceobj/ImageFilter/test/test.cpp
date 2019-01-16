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
#include <fstream>
#include <complex>
#include <cmath>
using namespace std;
int main(int argc,char ** argv)
{
    double pi = atan2(0,-1);
    if(argv[1][0] == '1')
    {
        ofstream fout1("complexPolarBIP");
        ofstream fout("complexXYBIP");
        int cnt = 0;
        complex<double> arr[24];//12 complex elements 2 bands
        complex<double> arr1[24];//12 complex elements 2 bands
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 4; ++j)
            {
                double x = 1.23*(i + 1);
                double y = 4.2*(j + 1);
                arr[cnt] = complex<double>(x,y);
                arr[cnt+1] = complex<double>(2*x,3*y);
                arr1[cnt] = complex<double>(sqrt(x*x + y*y),atan2(y,x));
                arr1[cnt+1] = complex<double>(sqrt(4*x*x + 9*y*y),atan2(3*y,2*x));
                ++cnt;
                ++cnt;
            }
        }
        fout.write((char *) &arr[0],24*sizeof(complex<double>));
        fout1.write((char *) &arr1[0],24*sizeof(complex<double>));
        fout.close();
        fout1.close();
    }
}
