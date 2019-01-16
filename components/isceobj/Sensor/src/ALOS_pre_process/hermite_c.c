/*******************************************************************************
 * Hermite orbit interpolator based on fortran code of Werner Gunter           *
 * 13th International Workshop on Laser Ranging, 2002, Washington, DC          *
 *******************************************************************************/
/********************************************************************************
 * Creator:  David T. Sandwell and Rob Mellors                                  *
 *           (San Diego State University, Scripps Institution of Oceanography)  *
 * Date   :  06/07/2007                                                         *
 ********************************************************************************/
/********************************************************************************
 * Modification history:                                                        *
 * Date:                                                                        *
 * 10/03/2007   -   converted from FORTRAN to C                                 *
 * *****************************************************************************/

#include "image_sio.h"
#include"lib_functions.h"

void hermite_c(double *x, double *y, double *z, int nmax, int nval, double xp, double *yp, int *ir)
{
/*

  interpolation by a polynomial using nval out of nmax given data points
 
  input:  x(i)  - arguments of given values (i=1,...,nmax)
          y(i)  - functional values y=f(x)
          z(i)  - derivatives       z=f'(x) 
          nmax  - number of given points in list
          nval  - number of points to use for interpolation
          xp    - interpolation argument
 
  output: yp    - interpolated value at xp
          ir    - return code
                  0 = ok
                  1 = interpolation not in center interval
                  2 = argument out of range

***** calls no other routines
*/
int	n, i, j, i0;
double	sj, hj, f0, f1;

/*  check to see if interpolation point is inside data range */

      	*yp = 0.0;
      	n = nval - 1;
      	*ir = 0;

	/* reduced index by 1 */
      	if (xp < x[0] || xp > x[nmax-1]) { 
      		fprintf(stderr,"interpolation point outside of data constraints\n");
      		*ir = 2;
      		exit(1);	
      		}

/*  look for given value immediately preceeding interpolation argument */

      	for (i=0; i<nmax; i++) {
      		if (x[i] >= xp) break; 
	}
/*  check to see if interpolation point is centered in  data range */
 	i0 = i - (n+1)/2;

      	if (i0 <= 0) { 
      		fprintf(stderr,"hermite: interpolation not in center interval\n");
      		i0 = 0;
      		*ir = 0;
      		}

	/* reduced index by 1 */
      	if (i0 + n > nmax) {
      		fprintf(stderr,"hermite: interpolation not in center interval\n");
      		i0 = nmax - n - 1;
      		*ir = 0;
      		}

	/*  do Hermite interpolation */
      	for (i = 0; i<=n; i++){
      		sj = 0.0;
      		hj = 1.0;
      		for (j=0; j<=n; j++){
      			if (j != i) {
				hj = hj*(xp - x[j + i0])/(x[i + i0] - x[j + i0]);
      				sj = sj + 1.0/(x[i + i0] - x[j + i0]);
      			}
   		}

      		f0 = 1.0 - 2.0*(xp - x[i + i0])*sj;
      		f1 = xp - x[i + i0];

      		*yp = *yp + (y[i + i0]*f0 + z[i + i0]*f1)*hj*hj;
		if (isnan(*yp) != 0){
			fprintf(stderr,"nan!\n");
			exit(1);
			}

 	}

/*	done 	*/
}
