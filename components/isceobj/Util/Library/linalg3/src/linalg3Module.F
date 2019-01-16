        module linalg3Module
            use, intrinsic :: iso_c_binding
            implicit none

            interface 
                subroutine matmat_c(r_a,r_b,r_c)BIND(C,NAME='matmat_C')
                    use, intrinsic :: iso_c_binding
                    implicit none

                    real(C_DOUBLE), dimension(3,3) :: r_a
                    real(C_DOUBLE), dimension(3,3) :: r_b
                    real(C_DOUBLE), dimension(3,3) :: r_c
                end subroutine matmat_c


                subroutine matvec_c(r_t, r_v, r_w)BIND(C,NAME='matvec_C')
                    use, intrinsic :: iso_c_binding
                    implicit none

                    real(C_DOUBLE), dimension(3,3) :: r_t
                    real(C_DOUBLE), dimension(3) :: r_v
                    real(C_DOUBLE), dimension(3) :: r_w
                end subroutine matvec_c

                subroutine tranmat_c(r_a,r_b)BIND(C,NAME='tranmat_C')
                    use, intrinsic :: iso_c_binding
                    implicit none
                    
                    real(C_DOUBLE), dimension(3,3) :: r_a
                    real(C_DOUBLE), dimension(3,3) :: r_b
                end subroutine tranmat_c

            end interface

        contains

            subroutine cross(r_u,r_v,r_w)BIND(C, NAME='cross_C')

!c****************************************************************
!c**
!c**    FILE NAME: cross.f
!c**
!c**     DATE WRITTEN: 8/3/90
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: The subroutine takes two vectors and returns 
!c**     their cross product.
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************
                use, intrinsic :: iso_c_binding
                implicit none

            !c INPUT VARIABLES:
                real(C_DOUBLE), dimension(3) :: r_u
                real(C_DOUBLE), dimension(3) :: r_v

            !c OUTPUT VARIABLES
                real(C_DOUBLE), dimension(3) :: r_W

        
!c      PROCESSING STEPS:

!c       compute vector norm

                r_w(1) = r_u(2)*r_v(3) - r_u(3)*r_v(2)
                r_w(2) = r_u(3)*r_v(1) - r_u(1)*r_v(3)  
                r_w(3) = r_u(1)*r_v(2) - r_u(2)*r_v(1)  

            end subroutine cross

            function dot(r_v,r_w)BIND(C,NAME='dot_C')

!c****************************************************************
!c**
!c**    FILE NAME: dot.f
!c**
!c**     DATE WRITTEN:7/15/90 
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: This routine computes the dot product of
!c**     two 3 vectors as a function.
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************
                use, intrinsic :: iso_c_binding
                implicit none

!c      INPUT VARIABLES:
                real(C_DOUBLE), dimension(3) :: r_v
                real(C_DOUBLE), dimension(3) :: r_w
   
!c      OUTPUT VARIABLES: dot is the output
                real(C_DOUBLE):: dot

!c      PROCESSING STEPS:

!c   compute dot product of two 3-vectors

                dot = r_v(1)*r_w(1) + r_v(2)*r_w(2) + r_v(3)*r_w(3) 

            end function dot


            subroutine lincomb(r_k1,r_u,r_k2,r_v,r_w)BIND(C,NAME='lincomb_C')

!c****************************************************************
!c**
!c**    FILE NAME: lincomb.f
!c**
!c**     DATE WRITTEN: 8/3/90
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: The subroutine forms the linear combination
!c**    of two vectors.
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************
                use, intrinsic :: iso_c_binding 
                implicit none

!c      INPUT VARIABLES:
                real(C_DOUBLE), dimension(3) :: r_u     !3x1 vector
                real(C_DOUBLE), dimension(3) :: r_v     !3x1 vector
                real(C_DOUBLE), value :: r_k1           !scalar
                real(C_DOUBLE), value :: r_k2           !scalar
   
!c       OUTPUT VARIABLES:
                real(C_DOUBLE), dimension(3) :: r_w     !3x1 vector


!c       PROCESSING STEPS:

!c       compute linear combination

                r_w(1) = r_k1*r_u(1) + r_k2*r_v(1)
                r_w(2) = r_k1*r_u(2) + r_k2*r_v(2)
                r_w(3) = r_k1*r_u(3) + r_k2*r_v(3)
      
            end subroutine lincomb 


            subroutine matmat(r_a,r_b,r_c)BIND(C,NAME='matmat_F')

!c****************************************************************
!c**
!c**    FILE NAME: matmat.for
!c**
!c**     DATE WRITTEN: 8/3/90
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: The subroutine takes two 3x3 matrices
!c**     and multiplies them to return another 3x3 matrix.
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************
                use, intrinsic :: iso_c_binding
                implicit none

!c      INPUT VARIABLES:
                real(C_DOUBLE), dimension(3,3) :: r_a
                real(C_DOUBLE), dimension(3,3) :: r_b
   
!c      OUTPUT VARIABLES:
                real(C_DOUBLE), dimension(3,3) :: r_c           !3x3 matrix

!c      LOCAL VARIABLES:
                integer i         

!c      PROCESSING STEPS:

!c       compute matrix product

                do i=1,3
                    r_c(i,1) = r_a(i,1)*r_b(1,1) + r_a(i,2)*r_b(2,1) + 
     +                  r_a(i,3)*r_b(3,1)
                    r_c(i,2) = r_a(i,1)*r_b(1,2) + r_a(i,2)*r_b(2,2) + 
     +                  r_a(i,3)*r_b(3,2)
                    r_c(i,3) = r_a(i,1)*r_b(1,3) + r_a(i,2)*r_b(2,3) + 
     +                  r_a(i,3)*r_b(3,3)
                enddo 
          
            end subroutine matmat




            subroutine matvec(r_t,r_v,r_w)BIND(C,NAME='matvec_F')

!c****************************************************************
!c**
!c**    FILE NAME: matvec.f
!c**
!c**     DATE WRITTEN: 7/20/90
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: The subroutine takes a 3x3 matrix 
!c**     and a 3x1 vector a multiplies them to return another 3x1
!c**    vector.
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************
                use, intrinsic :: iso_c_binding
                implicit none

!c      INPUT VARIABLES:
                real(C_DOUBLE), dimension(3,3) :: r_t
                real(C_DOUBLE), dimension(3) :: r_v

   
!c      OUTPUT VARIABLES:
                real(C_DOUBLE), dimension(3) :: r_w         !3x1 vector


!c      PROCESSING STEPS:

!c       compute matrix product

                r_w(1) = r_t(1,1)*r_v(1) + r_t(1,2)*r_v(2) + r_t(1,3)*r_v(3)
                r_w(2) = r_t(2,1)*r_v(1) + r_t(2,2)*r_v(2) + r_t(2,3)*r_v(3)
                r_w(3) = r_t(3,1)*r_v(1) + r_t(3,2)*r_v(2) + r_t(3,3)*r_v(3)
          
            end subroutine matvec 


            function norm(r_v)BIND(C,NAME='norm_C')

!c****************************************************************
!c**
!c**    FILE NAME: norm.f
!c**
!c**     DATE WRITTEN: 8/3/90
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: The subroutine takes vector and returns 
!c**     its norm.
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************
                use, intrinsic :: iso_c_binding
                implicit none

!c      INPUT VARIABLES:
                real(C_DOUBLE), dimension(3) :: r_v   !3x1 vector
   

!c      LOCAL VARIABLES:
                real(C_DOUBLE) ::  norm

!c      PROCESSING STEPS:

!c       compute vector norm

                norm = sqrt(r_v(1)**2 + r_v(2)**2 + r_v(3)**2)
      
            end function norm

            subroutine tranmat(r_a,r_b)BIND(C,NAME='tranmat_F')

!c****************************************************************
!c**
!c**    FILE NAME: tranmat.f
!c**
!c**     DATE WRITTEN: 8/3/90
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: The subroutine takes a 3x3 matrix
!c**     and computes its transpose.
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************
                use, intrinsic :: iso_c_binding
                implicit none

!c      INPUT VARIABLES:
                real(C_DOUBLE), dimension(3,3) :: r_a       !3x3 matrix
   
!c      OUTPUT VARIABLES:
                real(C_DOUBLE), dimension(3,3) :: r_b       !3x3 matrix

!c      LOCAL VARIABLES:
                integer i,j         

!c      PROCESSING STEPS:


                do i=1,3
                    do j=1,3
                        r_b(i,j) = r_a(j,i)
                    enddo 
                enddo
          
            end subroutine tranmat


            subroutine unitvec(r_v,r_u)BIND(C,NAME='unitvec_C')

!c****************************************************************
!c**
!c**    FILE NAME: unitvec.f
!c**
!c**     DATE WRITTEN: 8/3/90
!c**
!c**     PROGRAMMER:Scott Hensley
!c**
!c**    FUNCTIONAL DESCRIPTION: The subroutine takes vector and returns 
!c**     a unit vector.
!c**
!c**     ROUTINES CALLED:none
!c**  
!c**     NOTES: none
!c**
!c**     UPDATE LOG:
!c**
!c*****************************************************************
                use, intrinsic :: iso_c_binding
                implicit none

!c      INPUT VARIABLES:
                real(C_DOUBLE), dimension(3) :: r_v     !3x1 vector
   
!c      OUTPUT VARIABLES:
                real(C_DOUBLE), dimension(3) :: r_u     !3x1 vector

!c      LOCAL VARIABLES:
                real*8 r_n

!c      PROCESSING STEPS:

!c       compute vector norm

                r_n = sqrt(r_v(1)**2 + r_v(2)**2 + r_v(3)**2)
      
                if(r_n .ne. 0)then  
                    r_u(1) = r_v(1)/r_n
                    r_u(2) = r_v(2)/r_n
                    r_u(3) = r_v(3)/r_n
                endif
 
            end subroutine unitvec  

        end module linalg3Module
