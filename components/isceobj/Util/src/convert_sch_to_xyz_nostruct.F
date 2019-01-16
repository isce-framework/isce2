c****************************************************************

	subroutine convert_sch_to_xyz(r_mat,r_ov,r_radcur,
     1					r_schv,r_xyzv,i_type)

c****************************************************************
c**
c**	FILE NAME: convert_sch_to_xyz.for
c**
c**     DATE WRITTEN:1/15/93 
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: This routine applies the affine matrix 
c**     provided to convert the sch coordinates xyz WGS-84 coordintes or
c**     the inverse transformation.
c**
c**     ROUTINES CALLED:latlon,matvec
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       	implicit none

c	INPUT VARIABLES:
        real*8 r_mat(3,3)             !rotation matrix
        real*8 r_ov(3)                !translation vector
	  real*8 r_radcur               !radius of approximating sphere
        real*8 r_schv(3)              !sch coordinates of a point
        real*8 r_xyzv(3)              !WGS-84 coordinates of a point
        integer i_type                !i_type = 0 sch => xyz ; 
                                      !i_type = 1 xyz => sch
   
c   	OUTPUT VARIABLES:see input

c	LOCAL VARIABLES:
        integer i_f,i_t,i,j
        real*8 r_schvt(3),r_e2u,r_lats,r_lons,r_matinv(3,3),r_h

c	DATA STATEMENTS:
        data r_e2u /0.0d0/
        data i_f /1/

c       SAVE STATEMENTS:  (needed on Freebie only)
        save i_f,r_matinv 

C	FUNCTION STATEMENTS:none

c  	PROCESSING STEPS:

c       compute the linear portion of the transformation 

	if(i_type .eq. 0)then

	   r_lats = r_schv(2)/r_radcur
	   r_lons = r_schv(1)/r_radcur

           i_t = 1
c           call latlon(r_radcur,r_e2u,r_schvt,r_lats,
c     1			r_lons,r_schv(3),i_t)
           call latlon_elp(r_radcur,r_e2u,r_schvt,r_lats,
     1			r_lons,r_schv(3),i_t)
           call matvec(r_mat,r_schvt,r_xyzv)
           call lincomb(1.d0,r_xyzv,1.d0,r_ov,r_xyzv)           

        elseif(i_type .eq. 1)then

c	   if(i_f .eq. 1)then
	      do i=1,3
		 do j=1,3
		    r_matinv(i,j) = r_mat(j,i)
		 enddo
	      enddo
	      i_f = 0
c	   endif

	   call lincomb(1.d0,r_xyzv,-1.d0,r_ov,r_schvt)
           call matvec(r_matinv,r_schvt,r_schv)
           i_t = 2
c           call latlon(r_radcur,r_e2u,r_schv,r_lats,r_lons,r_h,i_t)
           call latlon_elp(r_radcur,r_e2u,r_schv,r_lats,r_lons,r_h,i_t)
 
           r_schv(1) = r_radcur*r_lons
           r_schv(2) = r_radcur*r_lats
           r_schv(3) = r_h

	endif

	end



