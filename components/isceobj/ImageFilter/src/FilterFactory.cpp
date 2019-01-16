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



#include "FilterFactory.h"
#include "MagnitudeExtractor.h"
#include "PhaseExtractor.h"
#include "RealExtractor.h"
#include "ImagExtractor.h"
#include "ComponentExtractor.h"
#include "BandExtractor.h"
using namespace std;

Filter * FilterFactory::createFilter(string type,int selector)
{
        Filter * filter;
        if(type == "MagnitudeExtractor")
        {
            //Magnitude from cartesian
            filter = new MagnitudeExtractor;
        }
        else if(type == "ComponentExtractor")
        {
            //Magnitude from polar or Real from cartesian selector = 0
            //Phase from polar or Imag from cartesian selector = 1
            filter = new ComponentExtractor;
            filter->selectComponent(selector);
        }
        else if(type == "PhaseExtractor")
        {
            //Phase from cartesian
            filter = new PhaseExtractor;
        }
        else if(type == "RealExtractor")
        {
            //Real from Polar
            filter = new RealExtractor;
        }
        else if(type == "ImagExtractor")
        {
            //Imag from polar
            filter = new ImagExtractor;
        }
        else if(type == "BandExtractor")
        {
            //Extract Band = selector
            filter = new BandExtractor;
            filter->selectBand(selector);
        }
        else
        {
            cout << "Filter " << type << " not implemented." << endl;
            ERR_MESSAGE;
        }
        return filter;
}
