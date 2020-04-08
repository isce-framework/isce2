c****************************************************************

	subroutine look_coord_conv(l0,x0,l,x)

c****************************************************************
c**
c**	FILE NAME: look_coord_conv.f
c**
c**     DATE WRITTEN: 4/20/2017
c**
c**     PROGRAMMER:Cunren Liang
c**
c** 	FUNCTIONAL DESCRIPTION: The subroutine calculates the
c**     coordinate x with number of looks l corresponding to
c**     x0 with number of looks l0
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
        integer l0
        real*8  x0
        integer l

   
c   	OUTPUT VARIABLES:see input

c	LOCAL VARIABLES:
	real*8 x

c  	PROCESSING STEPS:

c       compute x

        x = x0 * l0 / l + (l0-l)/(2.0*l)
      
        end  
