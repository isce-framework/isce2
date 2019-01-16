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

        module poly1dModule
            use, intrinsic:: iso_c_binding
            implicit none

            type, bind(C) :: poly1dType
                integer(C_INT) :: order    !c poly1d order
                real(C_DOUBLE) :: mean     !c Mean
                real(C_DOUBLE) :: norm     !c Norm 
                type(C_PTR) :: coeffs         !c Pointer to array of coeffs
            end type poly1dType

            interface

                function evalPoly1d_f(poly,x)BIND(C,NAME='evalPoly1d')
                    use, intrinsic :: iso_c_binding
                    import :: poly1dType
                    type(poly1dType) :: poly
                    real(C_DOUBLE), value :: x
                    real(C_DOUBLE)  :: evalPoly1d_f
                end function evalPoly1d_f

                subroutine modifyMean1d_f(src,targ,off)BIND(C,NAME='modifyMean1d')
                    use, intrinsic :: iso_c_binding
                    import :: poly1dType
                    type(poly1dType) :: src, targ
                    real(C_DOUBLE), value :: off
                end subroutine modifyMean1d_f

                subroutine modifyNorm1d_f(src,targ,norm)BIND(C,NAME='modifyNorm1d')
                    use, intrinsic :: iso_c_binding
                    import:: poly1dType
                    type(poly1dType) :: src, targ
                    real(C_DOUBLE), value :: norm
                end subroutine modifyNorm1d_f

                subroutine scalepoly1d_f(src,targ,naz,maz)BIND(C,NAME='scalePoly1d')
                    use, intrinsic :: iso_c_binding
                    import:: poly1dType
                    type(poly1dType) :: src, targ
                    real(C_DOUBLE), value:: naz, maz
                end subroutine scalepoly1d_f 

                subroutine setCoeff1d_f(src,i,val)BIND(C,NAME='setCoeff1d')
                    use, intrinsic :: iso_c_binding
                    import:: poly1dType
                    type(poly1dType) :: src
                    integer(C_INT), value :: i 
                    real(C_DOUBLE), value :: val
                end subroutine setCoeff1d_f

                function getCoeff1d_f(src,i)BIND(C,NAME='getCoeff1d')
                    use, intrinsic :: iso_c_binding
                    import :: poly1dType
                    type(poly1dType) :: src
                    integer(C_INT), value :: i
                    real(C_DOUBLE) :: getCoeff1d_f
                end function getCoeff1d_f

                !!Not to be used in fortran
                function createpoly1d_f(az)BIND(C,NAME='createPoly1d')
                    use, intrinsic :: iso_c_binding
                    import :: poly1dType
                    type(poly1dType) :: createpoly1d_f
                    integer(C_INT), value :: az
                end function createpoly1d_f

                subroutine initpoly1d_f(poly,az)BIND(C,NAME='initPoly1d')
                    use, intrinsic :: iso_c_binding
                    import :: poly1dType
                    type(poly1dType) :: poly
                    integer(C_INT), value :: az
                end subroutine initpoly1d_f

                subroutine cleanpoly1d_f(poly)BIND(C,NAME='cleanPoly1d')
                    use, intrinsic :: iso_c_binding
                    import :: poly1dType
                    type(poly1dType) :: poly
                end subroutine cleanpoly1d_f

                !!Not to be really used in Fortran
                subroutine deletepoly1d_f(poly)BIND(C,NAME='deletePoly1d')
                    use, intrinsic :: iso_c_binding
                    import :: poly1dType
                    type(poly1dType) :: poly
                end subroutine deletepoly1d_f

                subroutine printpoly1d_f(poly)BIND(C,NAME='printPoly1d')
                    use, intrinsic :: iso_c_binding
                    import :: poly1dType
                    type(poly1dType) :: poly
                end subroutine printpoly1d_f

            end interface

        end module poly1dModule

