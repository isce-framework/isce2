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


            module geocodeMethods
                use uniform_interp
                implicit none

                real*8, dimension(:), allocatable :: r_filter
                real*4, dimension(:), allocatable :: fintp
                real*4 :: f_delay 

                integer :: sinc_len,sinc_sub
                integer :: SINC_METHOD, BILINEAR_METHOD
                integer :: BICUBIC_METHOD, NEAREST_METHOD
                parameter(SINC_METHOD=0,BILINEAR_METHOD=1)
                parameter(BICUBIC_METHOD=2,NEAREST_METHOD=3)
                parameter(sinc_sub=8192,sinc_len=8)
            
                interface
                    complex function intpTemplate(ifg,i_x,i_y,f_x,f_y,nx,ny)
                        complex, dimension(:,:) :: ifg
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

                        if (method.eq.SINC_METHOD) then
                            allocate(r_filter(0:(sinc_sub*sinc_len)))
                            allocate(fintp(0:(sinc_sub*sinc_len-1)))

                            call sinc_coef(ONE,ONE*sinc_len,sinc_sub,ZERO,1,i_intplength,i_filtercoef,r_filter)

                            do i=0,sinc_len-1
                                do j=0, sinc_sub-1
                                   fintp(i+j*sinc_len) = r_filter(j+i*sinc_sub)
                                enddo
                            enddo

                            f_delay = sinc_len/2.0

                        else if (method.eq.BILINEAR_METHOD) then
                            f_delay = 2.0
                        else if (method.eq.BICUBIC_METHOD) then
                            f_delay=3.0
                        else if (method.eq.NEAREST_METHOD) then
                            f_delay=2.0
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

                    complex function intp_sinc(ifg,i_x,i_y,f_x,f_y,nx,ny)
                        implicit none
                        complex, dimension(:,:) :: ifg
                        integer:: i_x,i_y,nx,ny
                        real*8 :: f_x,f_y

                        intp_sinc=sinc_eval_2d_cx(ifg,fintp,sinc_sub,sinc_len,i_x,i_y,f_x,f_y,nx,ny)
                    end function intp_sinc

                    complex function intp_bilinear(ifg,i_x,i_y,f_x,f_y,nx,ny)
                        implicit none
                        complex,dimension(:,:) :: ifg
                        integer :: i_x,i_y,nx,ny
                        real*8 :: f_x,f_y

                        real*8 :: dx,dy

                        dx = i_x + f_x - f_delay+1
                        dy = i_y + f_y - f_delay+1

                        intp_bilinear = bilinear_cx(dy,dx,ifg)

                    end function intp_bilinear

                    complex function intp_bicubic(ifg,i_x,i_y,f_x,f_y,nx,ny)
                        implicit none
                        complex,dimension(:,:) :: ifg
                        integer :: i_x,i_y,nx,ny
                        real*8 :: f_x,f_y

                        real*8 :: dx,dy

                        dx = i_x + f_x -f_delay+1
                        dy = i_y + f_y -f_delay+1
                        intp_bicubic = bicubic_cx(dy,dx,ifg)
                    end function intp_bicubic

                    complex function intp_nearest(ifg,i_x,i_y,f_x,f_y,nx,ny)
                        implicit none
                        complex,dimension(:,:) :: ifg
                        integer :: i_x,i_y,nx,ny
                        real*8 :: f_x,f_y
                        integer :: dx,dy

                        dx = nint(i_x+f_x-f_delay+1)
                        dy = nint(i_y+f_y-f_delay+1)

                        intp_nearest = ifg(dx,dy)
                    end function intp_nearest

            end module geocodeMethods
