/*
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *                        NASA Jet Propulsion Laboratory
 *                      California Institute of Technology
 *                      (C) 2004-2005  All Rights Reserved
 *
 * <LicenseText>
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include "PowerOfTwo.h"

bool isaPowerOfTwo(int x)
{
  // Test if an integer is a power of two
  // x & (x-1) = 0 iff x is a power of 2
  // !(x&(x-1)) = 1 iff x is a power of 2
  // !(x&(x-1)) && (x>0) = 1 iff x is a power of 2 and x > 0

    return !(x & (x-1)) && (x>0);
}

int whichPowerOfTwo(unsigned int x)
{
  // Find log2 of an integer, assuming it is a power of 2
  // If x is not a power of 2, the log will be int-truncated
  // If the value passed in was negative the log will always
  // equal sizeof(int)*8-1

  int p = 0;
  while (x >>= 1){ p++; }
  return p;
}
