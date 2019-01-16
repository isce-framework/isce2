#include "BVector.h"

/**
 * Calculate the value of the Earth's magnetic B-Field at a particular spatial and temporal location.
 *
 * @param year the decimal year at which the value is desired [years]
 * @param lat the latitude of the point at which the value is desired [degrees]
 * @param lon the longitude of the point at which the value is desired [degrees]
 * @param alt the altitude of the point at which the value is desired [km]
 * @param beast on return, the value of the east component of the B Field [gauss]
 * @param bnorth on return, the value of the north component of the B Field [gauss]
 * @param bdown on return, the value of the down component of the B Field [gauss]
 * @param babs on return, the absolute value of the B Field [gauss]
 * @param dataPath the path to the data files containing the definitions of the magnetic field coefficients
 */
int
calculateBVector(float year, float lat, float lon, float alt, float *beast, float *bnorth, float *bdown, float *babs,char *dataPath)
{
  int flag;
  float lshell;

  igrf_bvector_(&lat,&lon,&year,&alt, &lshell,
	        &flag, beast, bnorth, bdown, babs,dataPath);
}
