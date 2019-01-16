#include <math.h>
#include <string.h>
#include <iostream>
#include <stdexcept>
#include "p2.h"

using namespace std;

p2_t::p2_t( )
{
	count = 0;

	add_end_markers( );
}

p2_t::~p2_t( )
{
    delete [] q;
    delete [] dn;
    delete [] np;
    delete [] n;
}

p2_t::p2_t( double quant )
{
	count = 0;

	add_end_markers( );

	add_quantile( quant );
}

void p2_t::add_end_markers( void )
{
	marker_count = 2;
	q = new double[ marker_count ];
	dn = new double[ marker_count ];
	np = new double[ marker_count ];
	n = new int[ marker_count ];
	dn[0] = 0.0;
	dn[1] = 1.0;

	update_markers( );
}

double * p2_t::allocate_markers( int count )
{
	double *newq = new double[ marker_count + count ];
	double *newdn = new double[ marker_count + count ];
	double *newnp = new double[ marker_count + count ];
	int *newn = new int[ marker_count + count ];

	memcpy( newq, q, sizeof(double) * marker_count );
	memcpy( newdn, dn, sizeof(double) * marker_count );
	memcpy( newnp, np, sizeof(double) * marker_count );
	memcpy( newn, n, sizeof(int) * marker_count );

	delete [] q;
	delete [] dn;
	delete [] np;
	delete [] n;

	q = newq;
	dn = newdn;
	np = newnp;
	n = newn;

	marker_count += count;

	return dn + marker_count - count;
}

void p2_t::update_markers( )
{
	p2_sort( dn, marker_count );

	/* Then entirely reset np markers, since the marker count changed */
	for( int i = 0; i < marker_count; i ++ ) {
		np[ i ] = (marker_count - 1) * dn[ i ] + 1;
	}
}

void p2_t::add_quantile( double quant )
{
	double *markers = allocate_markers( 3 );

	/* Add in appropriate dn markers */
	markers[0] = quant;
	markers[1] = quant/2.0;
	markers[2] = (1.0+quant)/2.0;

	update_markers( );
}

void p2_t::add_equal_spacing( int count )
{
	double *markers = allocate_markers( count - 1 );

	/* Add in appropriate dn markers */
	for( int i = 1; i < count; i ++ ) {
		markers[ i - 1 ] = 1.0 * i / count;
	}

	update_markers( );
}

inline int sign( double d )
{
	if( d >= 0.0 ) {
		return 1.0;
	} else {
		return -1.0;
	}
}

// Simple bubblesort, because bubblesort is efficient for small count, and
// count is likely to be small
void p2_t::p2_sort( double *q, int count )
{
	double k;
	int i, j;
	for( j = 1; j < count; j ++ ) {
		k = q[ j ];
		i = j - 1;
		
		while( i >= 0 && q[ i ] > k ) {
			q[ i + 1 ] = q[ i ];
			i --;
		}
		q[ i + 1 ] = k;
	}
}

double p2_t::parabolic( int i, int d )
{
	return q[ i ] + d / (double)(n[ i + 1 ] - n[ i - 1 ]) * ((n[ i ] - n[ i - 1 ] + d) * (q[ i + 1 ] - q[ i ] ) / (n[ i + 1] - n[ i ] ) + (n[ i + 1 ] - n[ i ] - d) * (q[ i ] - q[ i - 1 ]) / (n[ i ] - n[ i - 1 ]) );
}

double p2_t::linear( int i, int d )
{
	return q[ i ] + d * (q[ i + d ] - q[ i ] ) / (n[ i + d ] - n[ i ] );
}

void p2_t::add( double data )
{
	int i;
	int k;
	double d;
	double newq;

	if( count >= marker_count ) {
		count ++;

		// B1
		if( data < q[0] ) {
			q[0] = data;
			k = 1;
		} else if( data >= q[marker_count - 1] ) {
			q[marker_count - 1] = data;
			k = marker_count - 1;
		} else {
			for( i = 1; i < marker_count; i ++ ) {
				if( data < q[ i ] ) {
					k = i;
					break;
				}
			}
		}

		// B2
		for( i = k; i < marker_count; i ++ ) {
			n[ i ] ++;
			np[ i ] = np[ i ] + dn[ i ];
		}
		for( i = 0; i < k; i ++ ) {
			np[ i ] = np[ i ] + dn[ i ];
		}

		// B3
		for( i = 1; i < marker_count - 1; i ++ ) {
			d = np[ i ] - n[ i ];
			if( (d >= 1.0 && n[ i + 1 ] - n[ i ] > 1)
			 || ( d <= -1.0 && n[ i - 1 ] - n[ i ] < -1.0)) {
				newq = parabolic( i, sign( d ) );
				if( q[ i - 1 ] < newq && newq < q[ i + 1 ] ) {
					q[ i ] = newq;
				} else {
					q[ i ] = linear( i, sign( d ) );
				}
				n[ i ] += sign(d);
			}
		}
	} else {
		q[ count ] = data;
		count ++;

		if( count == marker_count ) {
			// We have enough to start the algorithm, initialize
			p2_sort( q, marker_count );

			for( i = 0; i < marker_count; i ++ ) {
				n[ i ] = i + 1;
			}
		}
	}
}

double p2_t::result( )
{
	if( marker_count != 5 ) {
		throw std::runtime_error("Multiple quantiles in use");
	}
    return result( dn[(marker_count - 1) / 2] );
}

double p2_t::result( double quantile )
{
	if( count < marker_count ) {
		int closest = 1;
        p2_sort(q, count);
		for( int i = 2; i < count; i ++ ) {
			if( fabs(((double)i)/count - quantile) < fabs(((double)closest)/marker_count - quantile ) ) {
				closest = i;
			}
		}
		return q[ closest ];
	} else {
		// Figure out which quantile is the one we're looking for by nearest dn
		int closest = 1;
		for( int i = 2; i < marker_count -1; i ++ ) {
			if( fabs(dn[ i ] - quantile) < fabs(dn[ closest ] - quantile ) ) {
				closest = i;
			}
		}
		return q[ closest ];
	}
}

void p2_t::report()
{
    std::cout << "QUANTILE: \t" << "VALUE \n";

    for(int i=0;i<marker_count;i++)
    {
        std::cout << dn[i] << " : \t" << q[i] << "\n";
    }
}

void p2_t::getStats(double *quant, double *val)
{
    for(int i=0; i<marker_count; i++)
    {
        quant[i] = dn[i];
        val[i] = q[i];
    }
}
