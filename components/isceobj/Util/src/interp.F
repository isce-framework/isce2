      real*4 function interp(ix, iy, xfrac, yfrac, rvs, iac, ioff)

c      integer*4 CMAX
c      parameter (CMAX = 7000)
      integer*4 ix, iy, n, first, iac, ioff
      real*8 xfrac, yfrac
      real*4 rvs(0:2*iac-1,*)
      complex temp(8)
      real*4 xintp(0:65544)
      data first/1/
      save xintp, first
      
c*    we want to do a 8192 x (8-pt sinc) interpolation of the original data, choosing
c*    the resulting nearest neighbor.

      if(first .eq. 1) then
         call intp_coef_norm(8192,xintp)
         first = 0
      end if
      n = 0

      ifrac = 8 * nint(xfrac * 8192.)
c      write (*,*)  '1 ',frac,ifrac
      do i = iy - 3 , iy + 4
         n = n + 1
         temp(n) = cmplx(0.0,0.0)
         do k = -3 , 4
            temp(n) = temp(n)  + rvs(ix+k+ioff,i) * xintp(ifrac+k+3)
         end do
      enddo

      ifrac = 8 * nint(yfrac * 8192.)
c      write (*,*)  '2 ', frac,ifrac
      cinterp = cmplx(0.,0.)
      do k = -3 , 4
         cinterp = cinterp + temp(k+4) * xintp(ifrac+k+3)
      end do
      interp = real(cinterp)
      return
      end

