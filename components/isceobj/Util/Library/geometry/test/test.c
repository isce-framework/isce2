#include "linalg3.h"
#include "geometry.h"

const double d2r = 0.017453292519943295;

void printPoint(double x[3])
{
    printf("%f , %f , %f \n", x[0], x[1], x[2]);
}

int main()
{

    cPeg peg;
    cPegtrans trans;
    cEllipsoid elp;
    double llh1[3];
    double llh2[3];
    double xyz1[3];
    double xyz2[3];

    double dist, hdg, rad;
    

    elp.a = 6378137.0;
    elp.e2 = 0.0066943799901;

    printf("Testing LLH to XYZ conversion:  \n");
    llh1[0] = 40.15*d2r; llh1[1] = -104.97*d2r; llh1[2]=2119.0;
    xyz2[0] = -1261499.8108277766; 
    xyz2[1] = -4717861.0677524200; 
    xyz2[2] = 4092096.6400047773;

    latlon_C(&elp, xyz1, llh1, LLH_2_XYZ);

    printf("Pt1 : ");
    printPoint(xyz1);
    printf("Pt2 : ");
    printPoint(xyz2);

    xyz2[0] -= xyz1[0]; 
    xyz2[1] -= xyz1[1];
    xyz2[2] -= xyz1[2];
    printf("Vector: ");
    printPoint(xyz2);
    
    dist = norm_C(xyz2);
    printf("Estimated Error: %f \n", dist); 
    printf("\n \n");

    printf("Testing XYZ to LLH conversion : \n");
    latlon_C(&elp, xyz1, llh2, XYZ_2_LLH);
    printf("Pt1 : ");
    printPoint(llh2);
    printf("Pt2 : ");
    printPoint(llh1);

    llh1[0] -= llh2[0];
    llh1[1] -= llh2[1];
    llh1[2] -= llh2[2];

    printf("Vector: ");
    printPoint(llh1);
    dist = norm_C(llh1);
    printf("Estimated error : %f \n", dist);
    printf("\n \n");


    printf("Testing radius of curvature: ");
    llh1[0] = 40.0*d2r; llh1[1] = -105.0*d2r; llh1[2]=2000.0;
    xyz1[0] = 6386976.165976;
    xyz1[1] = 6361815.825934;
    xyz1[2] = 6386976.165976;
    xyz2[0] = reast_C(&elp, llh1[0]);
    xyz2[1] = rnorth_C(&elp, llh1[0]);
    xyz2[2] = rdir_C(&elp, 90.0*d2r, llh1[0]);
    printf("Pt 1: ");
    printPoint(llh1);
    printf("Radii : ");
    printPoint(xyz2);
    printf("Ref: ");
    printPoint(xyz1);


    return 0;
}

