#include "InterleavedFactory.h"
#include "InterleavedBase.h"
#include "BIPAccessor.h"
#include "BILAccessor.h"
#include "BSQAccessor.h"
#include "GDALAccessor.h"
#include "Poly2dInterpolator.h"
#include "Poly1dInterpolator.h"

#include <algorithm>
using namespace std;

InterleavedBase *
InterleavedFactory::createInterleaved (string sel)
{
    transform (sel.begin (), sel.end (), sel.begin (), ::toupper);
    if (sel == "BIL")
    {
	return new BILAccessor ();
    }
    else if (sel == "BIP")
    {
	return new BIPAccessor ();
    }
    else if (sel == "BSQ")
    {
	return new BSQAccessor ();
    }
    else if (sel == "GDAL")
    {
	return new GDALAccessor ();
    }
    else if (sel == "POLY2D")
    {
	return new Poly2dInterpolator ();
    }
    else if (sel == "POLY1D")
    {
	return new Poly1dInterpolator ();
    }
    else
    {
	cout << "Error. " << sel << " is an unrecognized Interleaved Scheme"
		<< endl;
	ERR_MESSAGE
	;
    }
}
