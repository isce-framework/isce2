#ifndef __P2_H__
#define __P2_H__

class p2_t
{
public:
	p2_t( );
	~p2_t( );
	// Initialize a p^2 structure to target a particular quantile
	p2_t( double quantile );
	// Set a p^2 structure to target a particular quantile
	void add_quantile( double quant );
	// Set a p^2 structure to target n equally spaced quantiles
	void add_equal_spacing( int n );
    // Call to add a data point into the structure
	void add( double data );
    // Retrieve the value of the quantile. This function may only be called if only one quantile is targetted by the p2_t structure
	double result( );
    // Retrieve the value at a particular quantile.
	double result( double quantile );
    //Report the histogram
        void report();

    //Return stats
        void getStats(double *, double*);
private:
	void add_end_markers( );
	double *allocate_markers( int count );
	void update_markers( );
	void p2_sort( double *q, int count );
	double parabolic( int i, int d );
	double linear( int i, int d );
	double *q;
	double *dn;
	double *np;
	int *n;
	int count;
	int marker_count;
};

#endif
