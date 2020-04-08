c****************************************************************

        subroutine enubasis(r_lat,r_lon,r_enumat)

c****************************************************************
c**
c**     FILE NAME: enubasis.f
c**
c**     DATE WRITTEN: 7/22/93
c**
c**     PROGRAMMER:Scott Hensley
c**
c**     FUNCTIONAL DESCRIPTION:Takes a lat and lon and returns a 
c**     change of basis matrix from ENU to geocentric coordinates.  
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c****************************************************************

        implicit none

c       INPUT VARIABLES:
        real*8 r_lat                   !latitude (deg)
        real*8 r_lon                   !longitude (deg)
   
c       OUTPUT VARIABLES:
        real*8 r_enumat(3,3)

c       LOCAL VARIABLES:
        real*8 r_slt,r_clt,r_clo,r_slo

c       DATA STATEMENTS:

C       FUNCTION STATEMENTS:

c       PROCESSING STEPS:

        r_clt = cos(r_lat)
        r_slt = sin(r_lat)
        r_clo = cos(r_lon)
        r_slo = sin(r_lon)

c     North  vector

        r_enumat(1,2) = -r_slt*r_clo        
        r_enumat(2,2) = -r_slt*r_slo
        r_enumat(3,2) = r_clt
      
c     East vector

        r_enumat(1,1) = -r_slo        
        r_enumat(2,1) = r_clo
        r_enumat(3,1) = 0.d0

c     Up vector 
      
        r_enumat(1,3) = r_clt*r_clo        
        r_enumat(2,3) = r_clt*r_slo
        r_enumat(3,3) = r_slt

        end  
 
