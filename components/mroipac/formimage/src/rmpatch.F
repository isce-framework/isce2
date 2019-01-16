      subroutine rmpatch(trans,slope,inter,nnn,nl,nls,r0,delr,wavl,vel
     $     ,ht,re,fd,fdd,fddd,fdddd,prf,ideskew)

        implicit none
        real*4 pi, pi2
        parameter (pi=3.14159625265359,pi2=6.28318530718)
        real*8     r0,delr,wavl,fd,fdd,fddd,fdddd, slope, inter, tmpd
        real*8     rd0v1(nl),ht,re, gm, th, sinsqref, acc
        real*4        vel,prf, v1, veleff
        integer*4     nfilter, nnn, nl, nls,ideskew
        real*4        xintp, freq
	real*4        frac, ratio
	integer*4     na,i,ifrac, n, k
	complex*8     trans(nnn,nl)

        integer firsttime
        real*4 f0(nl), f_rate(nl),bdel(nl)
	real*8 r(nl)
        real*4 vtmp(nl)
        integer nvtmp(nl)

        complex*8 c_ctmpb(8)
        real*4    c_xintp(8)
        complex*8 c_ctmpa(nl)

        common /intp/ xintp(0:65544), nfilter

        data gm/3.9858528e14/
        data firsttime/1/
        save firsttime

!c initializations        

!c     load the interpolation array

        if(firsttime .eq. 1) then
           nfilter = 8192
           call intp_coef(nfilter,xintp)
           firsttime=0
!c          write(6,*) 'sinc initialized '
        end if

        acc = gm/(re+ht)**2

        do i = 1, nl
           r(i)      = r0 + (i-1)*delr !range to the line
           f0(i) = fd + ( fdd + ( fddd+fdddd*r(i) ) *r(i) )*r(i)
           th=dacos(((ht+re)**2+r(i)*r(i)-re**2)/(2.d0*r(i)*(re
     $          +ht)))
           sinsqref = f0(i) * wavl/(2.d0*vel*sqrt(re/(re+ht))*sin(th))
           f_rate(i) = (2.d0/wavl)*(acc*cos(th)+((vel*sinsqref)**2-vel
     $          **2)/r(i))
           veleff = sqrt(wavl*abs(f_rate(i))*r(i)/(2.d0))
           v1 =wavl**2/(8.*(veleff/prf)**2)
           rd0v1(i)    = v1*r(i)/(1 + v1*(f0(i)/prf)**2)
!c f_rate replaced with a more exact expression for chirp rate
!c           f_rate(i) = -2.d0 * vel**2*(rd0v1(i)/v1/r(i))**2/(wavl*r(i))

!c           write(6,*) 'f_rates ', f_rate(i), -2.d0 * (vel*sqrt(re/(re+ht)))
!c     $          **2*(rd0v1(i)/v1/r(i))**2/(wavl*r(i)),veleff

           bdel(i)   = slope * r(i) + inter
        end do
        
!c        write(6,*) 'linear arrays computed '
	do na = 1,nnn

!c get the interpolation amounts for a given azimuth pixel na as f(line)
              freq=(na-1)/float(nnn)*prf 
              do i = 1,nl
!c     frequencies must be within 0.5*prf of centroid
                 ratio = (freq-f0(i))/prf
                 n = nint(ratio)
                 freq = freq - n * prf
!c     range of a pixel at freq f, bdel is range correction for interferogram
                 
                 if(ideskew.eq.1)then
!c  deskewing
                    tmpd = bdel(i)+ ((r(i)-(wavl/4.)*f0(i)**2/
     $                f_rate(i))-r(1))/delr +
     $                rd0v1(i)*(1.d0/delr)*(freq**2-f0(i)**2)/prf**2
                 else
!c  not deskewing
                    tmpd  = i + rd0v1(i)*(1.d0/delr)*(freq**2-f0(i)**2)/prf**2
     $                + bdel(i)
                 end if
                 nvtmp(i) = int(tmpd)
                 vtmp(i) = tmpd - int(tmpd)
              enddo
!c           write(6,*) 'vtmp computed'

!c  interpolate that line according to coeffs determined above
              do i=1,nl
                 c_ctmpa(i)=cmplx(0.,0.)
                 if(nvtmp(i).ge.4 .and. nvtmp(i).lt.(nl-4)) then
                       frac = vtmp(i)
                       ifrac= 8*nint(frac*float(nfilter))
                       do k = 1 , 8
                          c_xintp(k) = xintp(ifrac+k-1)
                          c_ctmpb(k) = trans(na,nvtmp(i)-3+k-1)
                      end do
                       c_ctmpa(i)=c_ctmpb(1)*c_xintp(1)
     2                      +c_ctmpb(2)*c_xintp(2)
     3                      +c_ctmpb(3)*c_xintp(3)
     4                      +c_ctmpb(4)*c_xintp(4)
     5                      +c_ctmpb(5)*c_xintp(5)
     6                      +c_ctmpb(6)*c_xintp(6)
     7                      +c_ctmpb(7)*c_xintp(7)
     8                      +c_ctmpb(8)*c_xintp(8)
                    endif
              enddo
              do i = 1, nl
                 trans(na,i) = c_ctmpa(i)
              end do
!c           write(6,*) 'interpolation computed'

	enddo                   ! na-loop

        return
        end subroutine rmpatch
