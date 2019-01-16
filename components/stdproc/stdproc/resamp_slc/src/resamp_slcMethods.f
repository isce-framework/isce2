!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!#
!# Author: Piyush Agram
!# Copyright 2013, by the California Institute of Technology. ALL RIGHTS RESERVED.
!# United States Government Sponsorship acknowledged.
!# Any commercial use must be negotiated with the Office of Technology Transfer at
!# the California Institute of Technology.
!# This software may be subject to U.S. export control laws.
!# By accepting this software, the user agrees to comply with all applicable U.S.
!# export laws and regulations. User has the responsibility to obtain export licenses,
!# or other export authority as may be required before exporting such information to
!# foreign countries or providing access to foreign persons.
!#
!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


            module resamp_slcMethods
                use uniform_interp
                use akimaLib
                implicit none

                real*8, dimension(:), allocatable :: r_filter
                real*4, dimension(:), allocatable :: fintp
                real*4 :: f_delay, BADVALUE 

                integer :: sinc_len,sinc_sub,sinchalf,sincone
                integer :: SINC_METHOD, BILINEAR_METHOD
                integer :: BICUBIC_METHOD, NEAREST_METHOD
                integer :: AKIMA_METHOD, BIQUINTIC_METHOD
                parameter(SINC_METHOD=0,BILINEAR_METHOD=1)
                parameter(BICUBIC_METHOD=2,NEAREST_METHOD=3)
                parameter(AKIMA_METHOD=4, BIQUINTIC_METHOD=5)
                parameter(BADVALUE=-1000.0)
                parameter(sinc_sub=8192,sinc_len=8)
                parameter(sinchalf=sinc_len/2, sincone=sinc_len+1)
            
                interface
                    real*4 function intpTemplate(dem,i_x,i_y,f_x,f_y,nx,ny)
                        real*4, dimension(:,:) :: dem
                        integer :: i_x,i_y,nx,ny
                        real*8:: f_x,f_y
                    end function intpTemplate
               end interface

                contains
                    subroutine prepareMethods(method)
                        implicit none
                        integer method
                        integer i_intplength,i_filtercoef
                        integer i,j
                        real*8 ONE,ZERO
                        parameter(ONE=1.0,ZERO=0.0)
                        real*8 ssum

                        if (method.eq.SINC_METHOD) then
                            print *, 'Initializing Sinc interpolator'
                            allocate(r_filter(0:(sinc_sub*sinc_len)))
                            allocate(fintp(0:(sinc_sub*sinc_len-1)))

                            call sinc_coef(ONE,ONE*sinc_len,sinc_sub,ZERO,1,i_intplength,i_filtercoef,r_filter)

!                            print *, i_intplength, sinc_len
!                            print *, i_filtercoef, sinc_len*sinc_sub

                            !!!!Normalize rfilter here
                            do i=0,sinc_sub-1
                                ssum = 0.0d0
                                do j=0,sinc_len-1
                                    ssum = ssum + r_filter(i+j*sinc_sub)
                                end do
                                do j=0,sinc_len-1
                                    r_filter(i+j*sinc_sub) = r_filter(i+j*sinc_sub)/ssum
                                end do
                            enddo

                            do i=0,sinc_len-1
                                do j=0, sinc_sub-1
                                   fintp(i+j*sinc_len) = r_filter(j+i*sinc_sub)
                                enddo
                            enddo

!                            open(31, file='fintp', access='stream', status='unknown')
!                            write(31) fintp(0:sinc_sub*sinc_len-1)
!                            close(31)
                            !open(32, file='rfilter', access='stream',status='unknown')
                            !write(32) r_filter(0:sinc_sub*sinc_len-1)
                            !close(32)

                            f_delay = sinc_len/2.0

                        else if (method.eq.BILINEAR_METHOD) then
                            print *, 'Initializing Bilinear interpolator'
                            f_delay = 2.0
                        else if (method.eq.BICUBIC_METHOD) then
                            print *, 'Initializing Bicubic interpolator'
                            f_delay=3.0
                        else if (method.eq.NEAREST_METHOD) then
                            print *, 'Initializing Nearest Neighbor interpolator'
                            f_delay=2.0
                        else if (method.eq.AKIMA_METHOD) then
                            print *, 'Initializing Akima interpolator'
                            f_delay=2.0
                        else if (method.eq.BIQUINTIC_METHOD) then
                            print *, 'Initializing biquintic interpolator'
                            f_delay=3.0
                        else
                            print *, 'Unknown method type.'
                            stop
                        endif

                    end subroutine prepareMethods

                    subroutine unprepareMethods(method)
                        implicit none
                        integer method

                        if (method.eq.SINC_METHOD) then
                            deallocate(r_filter)
                            deallocate(fintp)
                        endif
                    end subroutine unprepareMethods

                    real*4 function intp_sinc(dem,i_x,i_y,f_x,f_y,nx,ny)
                        implicit none
                        real*4, dimension(:,:) :: dem
                        integer:: i_x,i_y,nx,ny
                        real*8 :: f_x,f_y

                        integer :: i_xx, i_yy

                        if ((i_x.lt.sinchalf) .or. (i_x.gt.(nx-sinchalf))) then
                            intp_sinc = BADVALUE
                            return
                        endif

                        if ((i_y.lt.sinchalf) .or. (i_y.gt.(ny-sinchalf))) then
                            intp_sinc = BADVALUE
                            return
                        endif

                        i_xx = i_x + sinchalf - 1 
                        i_yy = i_y + sinchalf - 1

                        intp_sinc=sinc_eval_2d_f(dem,fintp,sinc_sub,sinc_len,i_xx,i_yy,f_x,f_y,nx,ny)
                    end function intp_sinc


                    complex function intp_sinc_cx(ifg, i_x, i_y, f_x, f_y, nx,ny)
                        implicit none
                        complex, dimension(:,:) :: ifg
                        integer :: i_x,i_y,nx,ny
                        real*8 :: f_x, f_y

                        integer :: i_xx,i_yy

                        if ((i_x.lt.sinchalf) .or. (i_x.gt.(nx-sinchalf))) then
                            intp_sinc_cx = cmplx(0.,0.)
                            return
                        endif

                        if((i_y.lt.sinchalf) .or. (i_y.gt.(ny-sinchalf))) then
                            intp_sinc_cx = cmplx(0., 0.)
                            return
                        endif

                        i_xx = i_x + sinchalf - 1
                        i_yy = i_y + sinchalf - 1
                    
                        intp_sinc_cx = sinc_eval_2d_cx(ifg,fintp,sinc_sub,sinc_len,i_xx,i_yy,f_x,f_y,nx,ny)
                    end function intp_sinc_cx

                    real*4 function intp_bilinear(dem,i_x,i_y,f_x,f_y,nx,ny)
                        implicit none
                        real*4,dimension(:,:) :: dem
                        integer :: i_x,i_y,nx,ny
                        real*8 :: f_x,f_y,temp

                        real*8 :: dx,dy

                        dx = i_x + f_x
                        dy = i_y + f_y

                        if ((i_x.lt.1).or.(i_x.ge.nx)) then
                            intp_bilinear=BADVALUE
                            return
                        endif

                        if ((i_y.lt.1).or.(i_y.ge.ny)) then
                            intp_bilinear=BADVALUE
                            return
                        endif
                        
                        temp = bilinear(dy,dx,dem)
                        intp_bilinear = sngl(temp)

                    end function intp_bilinear

                    real*4 function intp_bicubic(dem,i_x,i_y,f_x,f_y,nx,ny)
                        implicit none
                        real*4,dimension(:,:) :: dem
                        integer :: i_x,i_y,nx,ny
                        real*8 :: f_x,f_y

                        real*8 :: dx,dy,temp

                        dx = i_x + f_x
                        dy = i_y + f_y

                        if ((i_x.lt.2).or.(i_x.ge.(nx-1))) then
                            intp_bicubic = BADVALUE
                            return
                        endif

                        if ((i_y.lt.2).or.(i_y.ge.(ny-1))) then
                            intp_bicubic = BADVALUE
                            return
                        endif

                        temp = bicubic(dy,dx,dem)
                        intp_bicubic = sngl(temp) 
                    end function intp_bicubic

                    real*4 function intp_biquintic(dem,i_x,i_y,f_x,f_y,nx,ny)
                        implicit none
                        real*4,dimension(:,:) :: dem
                        integer :: i_x,i_y,nx,ny
                        real*8 :: f_x,f_y

                        real*8 :: dx,dy
                        real*4 :: interp2Dspline

                        dx = i_x + f_x
                        dy = i_y + f_y

                        if ((i_x.lt.3).or.(i_x.ge.(nx-2))) then
                            intp_biquintic = BADVALUE
                            return
                        endif

                        if ((i_y.lt.3).or.(i_y.ge.(ny-2))) then
                            intp_biquintic = BADVALUE
                            return
                        endif

                        intp_biquintic = interp2DSpline(6,ny,nx,dem,dy,dx) 
                    end function intp_biquintic

                    real*4 function intp_nearest(dem,i_x,i_y,f_x,f_y,nx,ny)
                        implicit none
                        real*4,dimension(:,:) :: dem
                        integer :: i_x,i_y,nx,ny
                        real*8 :: f_x,f_y
                        integer :: dx,dy

                        dx = nint(i_x+f_x)
                        dy = nint(i_y+f_y)

                        if ((dx.lt.1) .or. (dx.gt.nx)) then
                            intp_nearest = BADVALUE
                            return
                        endif

                        if ((dy.lt.1) .or. (dy.gt.ny)) then
                            intp_nearest = BADVALUE
                            return
                        endif

                        intp_nearest = dem(dx,dy)
                    end function intp_nearest

                    real*4 function intp_akima(dem,i_x,i_y,f_x,f_y,nx,ny)
                        implicit none
                        real*4, dimension(:,:) :: dem
                        integer :: i_x,i_y,nx,ny
                        real*8 :: f_x, f_y
                        real*8 :: dx, dy, temp
                        double precision, dimension(aki_nsys) :: poly

                        dx = i_x + f_x
                        dy = i_y + f_y
                        
                        if ((i_x.lt.1).or.(i_x.ge.(nx-1))) then
                            intp_akima = BADVALUE
                            return
                        endif

                        if ((i_y.lt.1).or.(i_y.ge.(ny-1))) then
                            intp_akima = BADVALUE
                            return
                        endif

                        call polyfitAkima(nx,ny,dem,i_x,i_y,poly)
                        temp = polyvalAkima(i_x,i_y,dx,dy,poly)
!!                        temp = akima_intp(ny,nx,dem,dy,dx)
                        intp_akima = sngl(temp)
                    end function intp_akima

            end module resamp_slcMethods
