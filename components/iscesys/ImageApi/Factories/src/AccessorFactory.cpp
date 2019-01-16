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



#include "AccessorFactory.h"
#include "CasterFactory.h"
#include "InterleavedFactory.h"
#include "DataCaster.h"
#include "InterleavedBase.h"
#include "DataAccessor.h"
#include "DataAccessorCaster.h"
#include "DataAccessorNoCaster.h"
#include "IQByteToFloatCpxCaster.h"
using namespace std;

DataAccessor *
AccessorFactory::createAccessor(string filename, string accessMode, int size,
    int bands, int width, string interleaved, string caster)
{

  CasterFactory CF;
  InterleavedFactory IF;
  InterleavedBase * interleavedAcc = IF.createInterleaved(interleaved);
  interleavedAcc->init(filename, accessMode, size, bands, width);
  DataCaster * casterD = CF.createCaster(caster);
  return new DataAccessorCaster(interleavedAcc, casterD);
}
DataAccessor *
AccessorFactory::createAccessor(string filename, string accessMode, int size,
    int bands, int width, string interleaved, string caster, float xmi, float xmq, int iqflip)
{

  CasterFactory CF;
  InterleavedFactory IF;
  InterleavedBase * interleavedAcc = IF.createInterleaved(interleaved);
  interleavedAcc->init(filename, accessMode, size, bands, width);
  DataCaster * casterD = CF.createCaster(caster);
  ((IQByteToFloatCpxCaster *) casterD)->setXmi(xmi);
  ((IQByteToFloatCpxCaster *) casterD)->setXmq(xmq);
  ((IQByteToFloatCpxCaster *) casterD)->setIQflip(iqflip);

  return new DataAccessorCaster(interleavedAcc, casterD);

}
DataAccessor *
AccessorFactory::createAccessor(string filename, string accessMode, int size,
    int bands, int width, string interleaved)
{

  InterleavedFactory IF;
  InterleavedBase * interleavedAcc = IF.createInterleaved(interleaved);
  interleavedAcc->init(filename, accessMode, size, bands, width);
  return new DataAccessorNoCaster(interleavedAcc);
}
DataAccessor *
AccessorFactory::createAccessor(void * poly, string interleaved, int width,
    int length, int dataSize)
{
  InterleavedFactory IF;
  InterleavedBase * interleavedAcc = IF.createInterleaved(interleaved);
  interleavedAcc->init(poly);
  interleavedAcc->setLineWidth(width);
  interleavedAcc->setNumberOfLines(length);
  interleavedAcc->setBands(1);
  interleavedAcc->setDataSize(dataSize);

  return new DataAccessorNoCaster(interleavedAcc);
}

void
AccessorFactory::finalize(DataAccessor * dataAccessor)
{
  dataAccessor->finalize();
}
