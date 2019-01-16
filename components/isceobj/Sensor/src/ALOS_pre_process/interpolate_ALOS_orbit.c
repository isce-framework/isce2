/********************************************************************************
 * Creator:  Rob Mellors and David T. Sandwell                                  *
 *           (San Diego State University, Scripps Institution of Oceanography)  *
 * Date   :  10/03/2007                                                         *
 ********************************************************************************/
/********************************************************************************
 * Modification history:                                                        *
 * Date:                                                                        *
 * *****************************************************************************/

#include "image_sio.h"
#include "lib_functions.h"
#define FACTOR 1000000

/*
void interpolate_ALOS_orbit_slow(struct ALOS_ORB *, double, double *, double *, double *, int *);
void interpolate_ALOS_orbit(struct ALOS_ORB *, double *, double *, double *, double, double *, double *, double *, int *);
*/

/*---------------------------------------------------------------*/
/* from David Sandwell's code */
void interpolate_ALOS_orbit_slow(struct ALOS_ORB *orb, double time, double *x, double *y, double *z, int *ir)
{
int	k;
double	pt0;
double *p, *pt, *pv;

	p = (double *) malloc(orb->nd*sizeof(double));
	pv = (double *) malloc(orb->nd*sizeof(double));
	pt = (double *) malloc(orb->nd*sizeof(double));

	/* seconds from Jan 1 */
	pt0 = (24.0*60.0*60.0)*orb->id + orb->sec;
	for (k=0; k<orb->nd; k++) pt[k] = pt0 + k*orb->dsec;

	interpolate_ALOS_orbit(orb, pt, p, pv, time, x, y, z, ir);

	free((double *) p);
	free((double *) pt);
	free((double *) pv);
}
/*---------------------------------------------------------------*/
void interpolate_ALOS_orbit(struct ALOS_ORB *orb, double *pt, double *p, double *pv, double time, double *x, double *y, double *z, int *ir)
{
/* ir; 			return code 		*/
/* time;		seconds since Jan 1 	*/
/* x, y, z;		position		*/
int	k, nval, nd;

	nval = 6; /* number of points to use in interpolation */
	nd = orb->nd;

	if (debug) fprintf(stderr," time %lf nd %d\n",time,nd);

	/* interpolate for each coordinate direction 	*/

	/* hermite_c c version 				*/
	for (k=0; k<nd; k++) {
		p[k] = orb->points[k].px;
		pv[k] = orb->points[k].vx;
		}

	hermite_c(pt, p, pv, nd, nval, time, x, ir);

	for (k=0; k<nd; k++) {
		p[k] = orb->points[k].py;
		pv[k] = orb->points[k].vy;
		}
	hermite_c(pt, p, pv, nd, nval, time, y, ir);
	if (debug) fprintf(stderr, "C pt %lf py %lf pvy %lf time %lf y %lf ir %d \n",*pt,p[0],pv[0],time,*y,*ir);

	for (k=0; k<nd; k++) {
		p[k] = orb->points[k].pz;
		pv[k] = orb->points[k].vz;
		}
	hermite_c(pt, p, pv, nd, nval, time, z, ir);
	if (debug) fprintf(stderr, "C pt %lf pz %lf pvz %lf time %lf z %lf ir %d \n",*pt,p[0],pv[0],time,*z,*ir);

}
/*---------------------------------------------------------------*/
