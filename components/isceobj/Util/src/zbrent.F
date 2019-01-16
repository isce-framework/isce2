c****************************************************************

	function zbrent(func,x1,x2,tol)

c****************************************************************
c**
c**	FILE NAME: zbrent.f
c**
c**     DATE WRITTEN: 8/3/90
c**
c**     PROGRAMMER:Scott Hensley
c**
c** 	FUNCTIONAL DESCRIPTION: The subroutine is a routine taken 
c**     from "Numerical Recipes" for finding the zero of a function
c**     to within a given tolerance. This routine requires that
c**     the function and desired tolerance be passed in as arguements
c**     along with two values which bracket the zero (i.e. the 
c**     function values at these points must be of opposite sign).    
c**
c**     ROUTINES CALLED:none
c**  
c**     NOTES: none
c**
c**     UPDATE LOG:
c**
c*****************************************************************

       IMPLICIT real*8 (a-h,o-z) 

c	INPUT VARIABLES:
 	real*8  x1                              !bracketing parameter
        real*8  x2                              !bracketing parameter
        real*8  tol                             !tolerance
        real*8  zbrent                          !value of zero
   
c   	OUTPUT VARIABLES:see input

c	LOCAL VARIABLES:
        parameter (itmax=100,eps=3.d-8)

c  	PROCESSING STEPS:

c       for details see "Numerical Recipes"
 
	a = x1
        b = x2
        fa = func(a)
        fb = func(b)

c	write(*,*) 'fa fb ',a,b,fa,fb
        if(fb*fa .gt. 0)then
		zbrent=1.e9 	!no zero in the range
		return
	end if

        fc = fb

        do iter=1,itmax

          if(fb*fc .gt. 0)then
             c = a
             fc = fa
             d = b - a
             e = d
          endif

          if(dabs(fc) .lt. dabs(fb))then
             a = b
             b = c
             c = a
             fa = fb
             fb = fc
             fc = fa
          endif

          tol1 = 2*eps*dabs(b) + .5*tol
          xm = .5*(c-b)

          if(dabs(xm) .le. tol1 .or. fb .eq. 0)then
             zbrent = b
             return
          endif

          if(dabs(e) .ge. tol1 .and. dabs(fa) .gt. dabs(fb))then
             
             sa = fb/fa
             if(a .eq. c)then
                p = 2*xm*s
                q = 1 - s
             else
                q = fa/fc
                r = fb/fc
                p = s*(2*xm*q*(q-r) - (b-a)*(r-1))
                q = (q-1)*(r-1)*(s-1)
             endif

             if(p .gt. 0) q = -q
             p = dabs(p)

             if(2*p .lt. dmin1(3.*xm*q - dabs(tol1*q),dabs(e*q)))then
                e = d
                d = p/q
             else
                d = xm
                e = d
             endif   
                  
	  else 

             d = xm
             e = d

          endif

          a = b
          fa = fb

          if(dabs(d) .gt. tol1)then
             b = b + d
          else     
	     b = b + sign(tol1,xm)
          endif
           
          fb = func(b)

     	enddo

        pause 'it ain t gonna work this time'

        zbrent = b

        return

        end
