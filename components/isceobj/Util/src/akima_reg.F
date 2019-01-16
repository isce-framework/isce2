!c Regular grid AKima resampling
!c Author : Piyush Agram
!c Date   : Dec 9, 2013
!c Adapted from SOSIE package : http://sourceforge.net/p/sosie/
!! Currently window sizes are fixed at 4 x 4
!! There are edge effects to using this library.
!! Rewritten to be  like other uniform_interp functions in ISCE.

        module AkimaLib

        implicit none
        integer, parameter :: aki_nsys = 16
        double precision, parameter :: aki_eps = epsilon(1.0d0)
        !Dimension of linear system to solve


        contains

            !!Equality operator to avoid underflow issues
            function aki_almostEqual(x,y)
                double precision, intent(in) :: x,y
                logical aki_almostEqual
                if (abs(x-y).le.aki_eps) then
                    aki_almostEqual = .true.
                else
                    aki_almostEqual = .false.
                endif
            end function aki_almostEqual

            subroutine printAkiNaN(nx,ny,ZZ,ix,iy,slpx,slpy,slpxy)
                !!Used only for debugging.
                integer, intent(in) :: nx, ny, ix, iy
                real*4, dimension(nx,ny), intent(in) :: ZZ
                double precision, intent(in):: slpx, slpy, slpxy
                logical flag
                integer i,j,ii,jj

                if (isnan(slpx).or.isnan(slpy).or.isnan(slpxy)) then 
                    print *, 'Slopes: ', slpx, slpy, slpxy
                    print *, 'Location ', iy, ix
                    print *, 'Data : '

                    do i=iy-2, iy+2
                       ii = min(max(i,3), ny-2)
                       do j=ix-2,ix+2
                          jj = min(max(j,3),nx-2)
                          print *, ZZ(jj,ii)
                        enddo
                    enddo
                    stop
                endif
            end subroutine printAkiNaN

            subroutine getParDer(nx,ny,ZZ,ix,iy,slpx,slpy,slpxy)
            !!Computer partial derivatives at (ix,iy)
                integer, intent(in) :: nx, ny, ix, iy
                integer :: xx, yy, ii, jj
                double precision, dimension(2,2) :: slpx, slpy, slpxy
                real*4, dimension(nx,ny), intent(in) :: ZZ

                double precision :: m1,m2,m3,m4
                double precision :: wx2, wx3, wy2, wy3
                double precision :: d22,e22,d23,e23
                double precision :: d42,e32,d43,e33


                do ii=1,2
                    yy = min(max(iy+ii,3),ny-2) 
                    do jj=1,2
                        xx = min(max(ix+jj,3),nx-2)

                        !!c Slope-X
                        m1 = (ZZ(xx-1,yy) - ZZ(xx-2,yy))
                        m2 = (ZZ(xx,yy) - ZZ(xx-1,yy))
                        m3 = (ZZ(xx+1,yy) - ZZ(xx,yy))
                        m4 = (ZZ(xx+2,yy) - ZZ(xx+1,yy))

                        !!
                        if (aki_almostEqual(m1,m2).and.aki_almostEqual(m3,m4)) then
                            slpx(jj,ii) = 0.5*(m2+m3)
                        else
                            wx2 = abs(m4 - m3)
                            wx3 = abs(m2 - m1)
                            slpx(jj,ii) = (wx2*m2 + wx3*m3)/(wx2+wx3)
                        endif

                        !!c Slope-Y
                        m1 = (ZZ(xx,yy-1) - ZZ(xx,yy-2))
                        m2 = (ZZ(xx,yy) - ZZ(xx,yy-1))
                        m3 = (ZZ(xx,yy+1) - ZZ(xx,yy))
                        m4 = (ZZ(xx,yy+2) - ZZ(xx,yy+1))

                        !!
                        if (aki_almostEqual(m1,m2).and.aki_almostEqual(m3,m4)) then
                            slpy(jj,ii) = 0.5*(m2+m3)
                        else
                            wy2 = abs(m4-m3)
                            wy3 = abs(m2-m1)
                            slpy(jj,ii) = (wy2*m2+wy3*m3)/(wy2+wy3)
                        endif

                        !!c Cross Derivative XY
                        d22 = ZZ(xx-1,yy) - ZZ(xx-1,yy-1)
                        d23 = ZZ(xx-1,yy+1) - ZZ(xx-1,yy)
                        d42 = ZZ(xx+1,yy) - ZZ(xx+1,yy-1)
                        d43 = ZZ(xx+1,yy+1) - ZZ(xx+1,yy)

                        e22 = m2 - d22
                        e23 = m3 - d23
                        e32 = d42 - m2
                        e33 = d43 - m3

                        !!
                        if (aki_almostEqual(wx2,0.0d0).and.aki_almostEqual(wx3,0.0d0) ) then
                            wx2 = 1.
                            wx3 = 1.
                        endif


                        if ( aki_almostEqual(wy2,0.0d0).and.aki_almostEqual(wy3,0.0d0) ) then
                            wy2 = 1.
                            wy3 = 1.
                        endif

                        slpxy(jj,ii) = (wx2*(wy2*e22+wy3*e23)+wx3*(wy2*e32+wy3*e33))/((wx2+wx3)*(wy2+wy3))
                       
!!                        if (isnan(slpxy(jj,ii))) then
!!                            print *, wx2, wx3, wy2, wy3
!!                            print *, e22, e23, e32, e33
!!                        endif
!!                        call printAkiNaN(nx,ny,ZZ,xx,yy,slpx(jj,ii),slpy(jj,ii),slpxy(jj,ii))
                    end do
                enddo

            end subroutine getParDer

            subroutine polyfitAkima(nx,ny,ZZ,ix,iy,poly)
                !!Compute the polynomial coefficients used for interpolation
                integer, intent(in) :: nx,ny
                integer :: ix,iy
                double precision, dimension(2,2) :: sx, sy, sxy
                double precision, dimension(aki_nsys) :: poly
                real*4, dimension(nx,ny), intent(in) :: ZZ


                double precision :: x, x2, x3, y, y2, y3, xy
                double precision :: b1, b2, b3, b4, b5, b6, b7, b8
                double precision :: b9, b10, b11, b12, b13,b14,b15,b16
        
                double precision :: c1, c2, c3, c4, c5, c6, c7, c8
                double precision :: c9,c10,c11,c12,c13,c14
                double precision :: c15,c16,c17,c18
                double precision :: d1, d2, d3, d4, d5, d6, d7, d8, d9
                double precision :: f1, f2, f3, f4, f5, f6


                !!First get partial derivatives
                call getParDer(nx,ny,ZZ,ix,iy,sx,sy,sxy)

                poly = 0.

                !!Local dx and dy
                x = 1.
                y = 1.
                !!
                x2 = x*x
                x3 = x2*x
                y2 = y*y
                y3 = y2*y
                xy = x*y
                !!
                !!Vector B at each point
                !!Values
                b1 = ZZ(ix,iy)
                b2 = ZZ(ix+1,iy)
                b3 = ZZ(ix+1,iy+1)
                b4 = ZZ(ix,iy+1)
                !!Slope x
                b5 = sx(1,1)
                b6 = sx(2,1)
                b7 = sx(2,2)
                b8 = sx(1,2)
                !!Slope y
                b9 = sy(1,1)
                b10 = sy(2,1)
                b11 = sy(2,2)
                b12 = sy(1,2)
                !!Cross derivative
                b13 = sxy(1,1)
                b14 = sxy(2,1)
                b15 = sxy(2,2)
                b16 = sxy(1,2)

                !!Bicubic polynomial
!
! System 16x16 :
! ==============          
!
!             (/ 0.    0.    0.   0.  0.    0.    0.   0. 0.   0.   0.  0. 0.  0. 0. 1. /)
!             (/ 0.    0.    0.   x^3  0.    0.    0.   x^2 0.   0.   0.  x  0.  0. 0. 1. /)
!             (/ x^3*y^3 x^3*y^2 x^3*y x^3  x^2*y^3 x^2*y^2 x^2*y x^2 x*y^3 x*y^2 x*y x  y^3  y^2 y  1. /)
!             (/ 0.    0.    0.   0.  0.    0.    0.   0. 0.   0.   0.  0. y^3  y^2 y  1. /)
!             (/ 0.      0.      0.     0.   0.     0.     0.    0.  0. 0. 0. 1. 0. 0. 0. 0. /)
!             (/ 0.      0.      0.     3*x^2 0.     0.     0.    2*x 0. 0. 0. 1. 0. 0. 0. 0. /)
!             (/ 3*x^2*y^3 3*x^2*y^2 3*x^2*y 3*x^2 2*x*y^3 2*x*y^2 2*x*y 2*x y^3 y^2 y  1. 0. 0. 0. 0. /)
!         A = (/ 0.      0.      0.     0.   0.     0.     0.    0.  y^3 y^2 y  1. 0. 0. 0. 0. /)
!             (/ 0.      0.     0.  0. 0.      0.     0. 0. 0.     0.    0. 0. 0.   0.  1. 0. /)
!             (/ 0.      0.     x^3  0. 0.      0.     x^2 0. 0.     0.    x  0. 0.   0.  1. 0. /)
!             (/ 3*x^3*y^2 2*x^3*y x^3  0. 3*x^2*y^2 2*x^2*y x^2 0. 3*x*y^2 2*x*y x  0. 3*y^2 2*y 1. 0. /)
!             (/ 0.      0.     0.  0. 0.      0.     0. 0. 0.     0.    0. 0. 3*y^2 2*y 1. 0. /)
!             (/ 0.      0.     0.   0. 0.     0.    0.  0. 0.   0.  1. 0. 0. 0. 0. 0. /)
!             (/ 0.      0.     3*x^2 0. 0.     0.    2*x 0. 0.   0.  1. 0. 0. 0. 0. 0. /)
!             (/ 9*x^2*y^2 6*x^2*y 3*x^2 0. 6*x*y^2 4*x*y 2*x 0. 3*y^2 2*y 1. 0. 0. 0. 0. 0. /)
!             (/ 0.      0.     0.   0. 0.     0.    0.  0. 3*y^2 2*y 1. 0. 0. 0. 0. 0. /)
!
!
! X = (/ a33, a32, a31, a30, a23, a22, a21, a20, a13, a12, a11, a10, a03, a02, a01, a00 /)
!
! B = (/ b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16  /)
!
!

                !!Polynomial coefficients forming the reference at [1,1[ of the
                !! 2 x 2 grid around the point of interest
                poly(11) = b13
                poly(12) = b5
                poly(15) = b9
                poly(16) = b1

                !!Scale the data values for local grid
                !!Not needed as we operate on integer grid for now
                !!Here in case, we want to apply different grid spacing
!!                b5 = x*b5
!!                b6 = x*b6
!!                b7 = x*b7
!!                b8 = x*b8

!!                b9 = y*b9
!!                b10 = y*b10
!!                b11 = y*b11
!!                b12 = y*b12

!!                b13 = xy*b13
!!                b14 = xy*b14
!!                b15 = xy*b15
!!                b16 = xy*b16

                !!Inversion of the 16x16 system
                c1=b1-b2      ; c2=b3-b4      ; c3=b5+b6
                c4=b7+b8      ; c5=b9-b10     ; c6=b11-b12
                c7=b13+b14    ; c8=b15+b16    ; c9=2*b5+b6
                c10=b7+2*b8   ; c11=2*b13+b14 ; c12=b15+2*b16
                c13=b5-b8     ; c14=b1-b4     ; c15=b13+b16
                c16= 2*b13 + b16 ; c17=b9+b12    ; c18=2*b9+b12
                !!
                d1=c1+c2   ; d2=c3-c4   ; d3=c5-c6
                d4=c7+c8   ; d5=c9-c10  ; d6=2*c5-c6
                d7=2*c7+c8 ; d8=c11+c12 ; d9=2*c11+c12
                !!
                f1=2*d1+d2 ; f2=2*d3+d4 ; f3=2*d6+d7
                f4=3*d1+d5 ; f5=3*d3+d8 ; f6=3*d6+d9
                !!
                poly(1)=2*f1+f2       
                poly(2)=-(3*f1+f3) 
                poly(3)=2*c5+c7
                poly(4)=2*c1+c3 
                poly(5)=-(2*f4+f5) 
                poly(6)=3*f4+f6
                poly(7)=-(3*c5+c11)
                poly(8)=-(3*c1+c9)
                poly(9)=2*c13+c15
                poly(10)=-(3*c13+c16)
                poly(13)=2*c14+c17
                poly(14)=-(3*c14+c18)


                !!Scale the polynomials with grid spacing
!!                vx(1)=vx(1)/(x3*y3) ; vx(2)=vx(2)/(x3*y2) ; vx(3)=vx(3)/(x3*y) ; vx(4)=vx(4)/x3
!!                vx(5)=vx(5)/(x2*y3) ; vx(6)=vx(6)/(x2*y2) ; vx(7)=vx(7)/(x2*y) ; vx(8)=vx(8)/x2
!!                vx(9)=vx(9)/(x*y3)  ; vx(10)=vx(10)/(x*y2); vx(13)=vx(13)/y3   ; vx(14)=vx(14)/y2

            end subroutine polyfitAkima

            function polyvalAkima(ix,iy,xx,yy,V)
                !!Evaluate the Akima polynomial at (x,y)
                !![x,y] should be between 0 and 1 each
                double precision :: polyvalAkima
                double precision, intent(in) :: xx,yy
                integer, intent(in) :: ix,iy
                double precision :: x,y
                double precision, dimension(aki_nsys), intent(in) :: V
                double precision :: p1,p2,p3,p4

                x = xx-ix
                y = yy-iy

                p1 = ( ( V(1)  * y + V(2)  ) * y + V(3)  ) * y + V(4)
                p2 = ( ( V(5)  * y + V(6)  ) * y + V(7)  ) * y + V(8)
                p3 = ( ( V(9)  * y + V(10) ) * y + V(11) ) * y + V(12)
                p4 = ( ( V(13) * y + V(14) ) * y + V(15) ) * y + V(16)
                polyvalAkima = ( ( p1    * x + p2    ) * x + p3    ) * x + p4

            end function polyvalAkima

            function akima_intp(nx,ny,z,x,y)
                double precision, intent(in) :: x,y
                integer, intent(in) :: nx,ny
                real*4, intent(in), dimension(:,:) :: z
                double precision :: akima_intp
                double precision, dimension(aki_nsys) :: poly
                integer :: xx,yy
                xx = int(x)
                yy = int(y)
                call polyfitAkima(nx,ny,z,xx,yy,poly)

                akima_intp = polyvalAkima(xx,yy,x,y,poly)
            end function akima_intp

        end module AkimaLib

