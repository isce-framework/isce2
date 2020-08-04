	subroutine bilinear(r_pnt1,r_pnt2,r_pnt3,r_pnt4,r_x,r_y,r_h)

c****************************************************************
c**
c**	FILE NAME: bilinear.for
c**
c**     DATE WRITTEN: 2/16/91
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION:This routine will take four points
c**	and do a bilinear interpolation to get the value for a point
c**	assumed to lie in the interior of the 4 points.
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       	implicit none

c	INPUT VARIABLES:
	real r_pnt1(3)		!point in quadrant 1
	real r_pnt2(3)		!point in quadrant 2
	real r_pnt3(3)		!point in quadrant 3
	real r_pnt4(3)		!point in quadrant 4
	real r_x		!x coordinate of point
	real r_y		!y coordinate of point
   
c   	OUTPUT VARIABLES:
        real r_h			!interpolated vaule

c	LOCAL VARIABLES:
        real r_t1,r_t2,r_h1b,r_h2b,r_y1b,r_y2b
        real r_diff

c	DATA STATEMENTS:none

C	FUNCTION STATEMENTS:none

c  	PROCESSING STEPS:

c	first find interpolation points in x direction

        r_diff=(r_pnt2(1)-r_pnt1(1))
        if ( r_diff .ne. 0 ) then
	  r_t1 = (r_x - r_pnt1(1))/r_diff
        else
          r_t1 = r_pnt1(1)
        endif
        r_diff=(r_pnt4(1)-r_pnt3(1))
        if ( r_diff .ne. 0 ) then
	  r_t2 = (r_x - r_pnt3(1))/r_diff
        else
          r_t2 = r_pnt4(1)
        endif
	r_h1b = (1.-r_t1)*r_pnt1(3) + r_t1*r_pnt2(3)
	r_h2b = (1.-r_t2)*r_pnt3(3) + r_t2*r_pnt4(3)

c	now interpolate in y direction

	r_y1b = r_t1*(r_pnt2(2)-r_pnt1(2)) + r_pnt1(2)
	r_y2b = r_t2*(r_pnt4(2)-r_pnt3(2)) + r_pnt3(2)

        r_diff=r_y2b-r_y1b
        if ( r_diff .ne. 0 ) then
	  r_h = ((r_h2b-r_h1b)/r_diff)*(r_y-r_y1b) + r_h1b
        else
          r_h = r_y2b
        endif
        end  
