!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!# Author: Piyush Agram
!# Copyright 2014, by the California Institute of Technology. ALL RIGHTS RESERVED.
!# United States Government Sponsorship acknowledged.
!# Any commercial use must be negotiated with the Office of Technology Transfer at
!# the California Institute of Technology.
!# This software may be subject to U.S. export control laws.
!# By accepting this software, the user agrees to comply with all applicable U.S.
!# export laws and regulations. User has the responsibility to obtain export licenses,
!# or other export authority as may be required before exporting such information to
!# foreign countries or providing access to foreign persons.
!#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        module poly2dModule
            use, intrinsic:: iso_c_binding
            implicit none

            type, bind(C) :: poly2dType
                integer(C_INT) :: rangeOrder    !c Range poly2d order
                integer(C_INT) :: azimuthOrder  !c Azimuth poly2d order
                real(C_DOUBLE) :: meanRange  !c Mean range
                real(C_DOUBLE) :: meanAzimuth !c Mean azimuth
                real(C_DOUBLE) :: normRange   !c Norm range
                real(C_DOUBLE) :: normAzimuth !c Norm azimuth
                type(C_PTR) :: coeffs         !c Pointer to array of coeffs
            end type poly2dType

            interface

                function evalPoly2d_f(poly,azi,rng)BIND(C,NAME='evalPoly2d')
                    use, intrinsic :: iso_c_binding
                    import :: poly2dType
                    type(poly2dType) :: poly
                    real(C_DOUBLE), value :: azi,rng
                    real(C_DOUBLE)  :: evalPoly2d_f
                end function evalPoly2d_f

                subroutine modifyMean2d_f(src,targ,azioff,rngoff)BIND(C,NAME='modifyMean2d')
                    use, intrinsic :: iso_c_binding
                    import :: poly2dType
                    type(poly2dType) :: src, targ
                    real(C_DOUBLE), value :: azioff, rngoff
                end subroutine modifyMean2d_f

                subroutine modifyNorm2d_f(src,targ,azinorm,rngnorm)BIND(C,NAME='modifyNorm2d')
                    use, intrinsic :: iso_c_binding
                    import:: poly2dType
                    type(poly2dType) :: src, targ
                    real(C_DOUBLE), value :: azinorm, rngnorm
                end subroutine modifyNorm2d_f

                subroutine scalePoly2d_f(src,targ,naz,maz,nrg,mrg)BIND(C,NAME='scalePoly2d')
                    use, intrinsic :: iso_c_binding
                    import:: poly2dType
                    type(poly2dType) :: src, targ
                    real(C_DOUBLE), value:: naz, maz
                    real(C_DOUBLE), value:: nrg, mrg
                end subroutine scalePoly2d_f 

                subroutine setCoeff2d_f(src,i,j,val)BIND(C,NAME='setCoeff2d')
                    use, intrinsic :: iso_c_binding
                    import:: poly2dType
                    type(poly2dType) :: src
                    integer(C_INT), value :: i,j 
                    real(C_DOUBLE), value :: val
                end subroutine setCoeff2d_f

                function getCoeff2d_f(src,i,j)BIND(C,NAME='getCoeff2d')
                    use, intrinsic :: iso_c_binding
                    import :: poly2dType
                    type(poly2dType) :: src
                    integer(C_INT), value :: i,j
                    real(C_DOUBLE) :: getCoeff2d_f
                end function getCoeff2d_f

                !!Not to be used in fortran
                function createpoly2d_f(az,rg)BIND(C,NAME='createPoly2d')
                    use, intrinsic :: iso_c_binding
                    import :: poly2dType
                    type(poly2dType) :: createpoly2d_f
                    integer(C_INT), value :: az
                    integer(C_INT), value :: rg
                end function createpoly2d_f

                subroutine initpoly2d_f(poly,az,rg)BIND(C,NAME='initPoly2d')
                    use, intrinsic :: iso_c_binding
                    import :: poly2dType
                    type(poly2dType) :: poly
                    integer(C_INT), value :: az
                    integer(C_INT), value :: rg
                end subroutine initpoly2d_f

                subroutine cleanpoly2d_f(poly)BIND(C,NAME='cleanPoly2d')
                    use, intrinsic :: iso_c_binding
                    import :: poly2dType
                    type(poly2dType) :: poly
                end subroutine cleanpoly2d_f

                !!Not to be really used in Fortran
                subroutine deletepoly2d_f(poly)BIND(C,NAME='deletePoly2d')
                    use, intrinsic :: iso_c_binding
                    import :: poly2dType
                    type(poly2dType) :: poly
                end subroutine deletepoly2d_f

                subroutine printpoly2d_f(poly)BIND(C,NAME='printPoly2d')
                    use, intrinsic :: iso_c_binding
                    import :: poly2dType
                    type(poly2dType) :: poly
                end subroutine printpoly2d_f

            end interface

        end module poly2dModule

