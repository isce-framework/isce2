      subroutine intp_coef(nfilter,xintp)
      
      use fortranUtils

      implicit none
      integer*4     i,j,nfilter
      real*8        x,y
      real*4        xintp(0:65544)
      real*8 pi

      pi = getPi()

!c     compute the interpolation factors
      do i=0,nfilter
         j = i*8
         x = real(i)/real(nfilter)
         y = sin(pi*x)/pi
         if(x.ne.0.0  .and. x.ne.1.0) then
            xintp(j  ) = -y/(3.0+x)
            xintp(j+1) =  y/(2.0+x)
            xintp(j+2) = -y/(1.0+x)
            xintp(j+3) =  y/x
            xintp(j+4) =  y/(1.0-x)
            xintp(j+5) = -y/(2.0-x)
            xintp(j+6) =  y/(3.0-x)
            xintp(j+7) = -y/(4.0-x)
         else if( x.eq.0.0) then
            xintp(j  ) = 0.0
            xintp(j+1) = 0.0
            xintp(j+2) = 0.0
            xintp(j+3) = 1.0
            xintp(j+4) = 0.0
            xintp(j+5) = 0.0
            xintp(j+6) = 0.0
            xintp(j+7) = 0.0
         else if( x.eq.1.0) then
            xintp(j  ) = 0.0
            xintp(j+1) = 0.0
            xintp(j+2) = 0.0
            xintp(j+3) = 0.0
            xintp(j+4) = 1.0
            xintp(j+5) = 0.0
            xintp(j+6) = 0.0
            xintp(j+7) = 0.0
         end if
      end do
      
      return
      end
