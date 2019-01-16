#include <stdio.h>
#include "orbit.h"

int main(int argc, char* argv[])
{

    double rds;
    double pos[3], vel[3];
    char schname[] = "hdr_SCH.rsc";
    char wgsname[] = "hdr_WGS84.rsc";
    double tintp = 0.0;
    double PI = atan(1.0) * 4.0;
    
   
    cPeg peg;
    cOrbit *orb;
    cOrbit *orbw;


    peg.lat = 18.62780383174511*PI/180.0;
    peg.lon = -159.35143391445047*PI/180.0;
    peg.hdg = 11.892607445876507*PI/180.0;
    rds = 6343556.266401461;
    
    orb = loadFromHDR(schname, SCH_ORBIT);
    orbw = loadFromHDR(wgsname, WGS84_ORBIT);

    printOrbit(orb);
    printOrbit(orbw);

    tintp = orb->UTCtime[0] + 55.0;

    interpolateSCHOrbit(orb, tintp, pos, vel);
    printf("Interpolated vector: \n");
    printf("%lf %lf %lf \n", pos[0], pos[1], pos[2]);
    printf("%lf %lf %lf \n", vel[0], vel[1], vel[2]);


    interpolateWGS84Orbit(orbw, tintp, pos, vel);
    printf("Interpolated vector: \n");
    printf("%lf %lf %lf \n", pos[0], pos[1], pos[2]);
    printf("%lf %lf %lf \n", vel[0], vel[1], vel[2]);


    deleteOrbit(orb);
    deleteOrbit(orbw);
}
