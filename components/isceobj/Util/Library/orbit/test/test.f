        program testing
            use, intrinsic :: iso_c_binding
            use geometryModule
            use orbitModule
            implicit none

            double precision :: rds
            double precision, dimension(3) :: pos, vel
            character*256 schname, wgsname
            double precision :: tintp, PI

            type(pegType) :: peg
            type(orbitType) :: orb, orbw
            
            
            
            schname = "hdr_SCH.rsc"
            wgsname = "hdr_WGS84.rsc"
            PI = atan(1.0) * 4.0;
    
   
            peg%r_lat = 18.62780383174511*PI/180.0
            peg%r_lon = -159.35143391445047*PI/180.0
            peg%r_hdg = 11.892607445876507*PI/180.0
            rds = 6343556.266401461
    
            orb = loadFromHDR_f(schname, SCH_ORBIT);
            orbw = loadFromHDR_f(wgsname, WGS84_ORBIT);

            call printOrbit_f(orb);
            call printOrbit_f(orbw);


            call getStateVector_f(orb, 0, tintp, pos, vel)
            tintp = tintp + 55.0;

            call interpolateSCHOrbit_f(orb, tintp, pos, vel);
            print *, "Interpolated vector"
            print *, pos(1), pos(2), pos(3)
            print *, vel(1), vel(2), vel(3)


            call interpolateWGS84Orbit_f(orbw, tintp, pos, vel);
            print *, "Interpolated vector"
            print *, pos(1), pos(2), pos(3)
            print *, vel(1), vel(2), vel(3)

            call cleanOrbit_f(orb);
            call cleanOrbit_f(orbw);
        end program testing
