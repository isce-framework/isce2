      subroutine poly_funcs(x,y,afunc,ma)
      
      real*8 afunc(ma),x,y
      real*8 cf(10)
      integer i_fitparam(10),i_coef(10)
      
      common /fred/ i_fitparam,i_coef
      
      data cf /10*0./
      
      do i=1,ma
         cf(i_coef(i))=1.
         afunc(i) = cf(1) + x*(cf(2) + x*(cf(5) + x*cf(9))) + &
           y*(cf(3) + y*(cf(6) + y*cf(10))) +  x*y*(cf(4) + y*cf(7) + x*cf(8))  
         cf(i_coef(i))=0.
      end do
      
      return
      end    
