c****************************************************************

      subroutine svdvecfit9(i_mp,i_rd,i_fp,r_vecin,r_vobs,r_cov,
     +     i_np,r_a,r_at2,r_u,r_v,r_w,r_chisq,l_chisq)

c****************************************************************
c**   
c**   FILE NAME: svdvecfit.f
c**   
c**   DATE WRITTEN: 01/02/95 
c**   
c**   PROGRAMMER: Scott Hensley
c**   
c**   FUNCTIONAL DESCRIPTION: This routine does a least squares fit 
c**   to a vector valued observation least squares problem.
c**   
c**   ROUTINES CALLED: gaussj,svbksb,svdcmp,funcs
c**   
c**   NOTES: funcs is a user supplied function giving the jacobian
c**   of the observation parameters wrt to fit parameters. This routine
c**   is a generalization of Numerical Recipes svdfit. Note that this
c**   routine can also be used in a nonlinear least squares procedure
c**   by iterating properly.
c**
c**   Solves the least problem 
c**
c**             T   -1     -1     T   -1 
c**    A = (AMAT COV  AMAT)  (AMAT COV  )VOBS 
c**
c**    where AMAT is the jacobain of the observations vs parameters,
c**    COV is the covriance matrix of observations
c**    and VOBS is the vector of observations. 
c**
c**    r_a should be passed in with current best estimate of values
c**   
c**   UPDATE LOG: 
c**         
c**  4/17/95 - Reversed order of r_vecin, r_vobs, and r_cov    SJS
c**            revmoved r_vt, cleaned up parameter list
c**   
c*****************************************************************

      implicit none

c     PARAMETERS:
      integer I_NPE                  !number of parameters to estimate = i_np
      integer I_RDE                  !number of observations per point = i_rd 
      real*8 R_TOL,R_LAMBDA
      parameter(I_NPE=9)
      parameter(I_RDE=2)
      parameter(R_TOL=1.0d-20)
c      parameter(R_TOL=1.0d-14)
      parameter (R_LAMBDA=1.d0)

      integer i_basec,i_basecdot,i_baseh,i_basehdot,i_rngoff,i_azoff,i_azscale,i_basecddt,i_basehddt
c      parameter(i_basec=1,i_basecdot=3,i_baseh=2,i_basehdot=4,i_rngoff=7,i_azoff=8,i_azscale=9)
c      parameter(i_basecddt=5,i_basehddt=6)

c      parameter(i_basec=1,i_basecdot=4,i_baseh=2,i_basehdot=5,i_rngoff=3,i_azoff=7,i_azscale=6)
c      parameter(i_basecddt=8,i_basehddt=9)

c     INPUT VARIABLES:
      integer i_mp                   !number of input points
      integer i_rd                   !number of observations each point
      integer i_fp                   !number of input parameters to func
      integer i_np                   !number of parameters to solve for

      real*8  r_vecin(i_fp,i_mp) 	 !vector values for func 
      real*8  r_vobs(i_rd,i_mp) 	 !vector of observations
      real*8  r_cov(i_rd,i_rd,i_mp)  !covariance matrix of observation
      real*8  r_chisq(i_rd,0:i_mp) 	 !chisq for solution and fit vs observation 
      real*8  r_a(i_np)   			 !solution to least squares
                                     !for each point 
      logical l_chisq                !evaluate the chisq for this fit
      
c     OUTPUT VARIABLES:
      real*8 r_at2(i_np)             !delta to add to previous solution
      real*8 r_u(i_np,i_np)          !svd matrix, orthogonal matrix
      real*8 r_v(i_np,i_np)          !svd matrix, orthogonal matrix
      real*8 r_w(i_np)               !svd matrix, diagonal matrix

c     LOCAL VARIABLES:
      integer i,j,k,i_pts
      real*8  r_covtemp(I_RDE,I_RDE)
      real*8  r_am(I_NPE,I_RDE)
      real*8  r_amat(I_RDE,I_NPE)
      real*8  r_ptot(I_NPE)
      real*8  r_wmax,r_thres,r_b(I_RDE,1),r_chird(I_RDE)

c      real*8  r_ucp(i_np,i_np)
c      real*8  r_uck(i_np,i_np)
c      real*8  r_inv(i_np,i_np)
c      real*8  r_max, r_sum

      integer i_paramest(I_NPE),i_usedata(I_RDE)

      real*8 r_scale1, r_scale2, r_scale3, r_scale4
      parameter (r_scale1=1.0d4)
      parameter (r_scale2=1.0d7)
      parameter (r_scale3=1.0d0)
      parameter (r_scale4=1.0d3)

      common/funcom3/i_paramest,i_usedata
      common/estlist/i_basec,i_basecdot,i_basecddt,i_baseh,i_basehdot,i_basehddt,i_rngoff,i_azoff,i_azscale

c     DATA STATEMENTS:

C     FUNCTION STATEMENTS:

c     PROCESSING STEPS:

c     init some arrays

c      write(*,*)  ' '
c      write(*,*)  'Inside SVDVECFIT'
c      write(*,*)  ' '

      if (i_rd .ne. I_RDE) stop 'ERROR - i_rd not equal to I_RDE in SVDVECFIT'
      if (i_np .ne. I_NPE) stop 'ERROR - i_np not equal to I_NPE in SVDVECFIT'

      do i=1,i_np
         do j=1,i_np
            r_u(i,j) = 0.0
         enddo
         r_ptot(i) = 0.0
      enddo

c     loop over the input points

      do i_pts=1,i_mp

c     invert the covariance matrix of the observation

         do i=1,i_rd
            do j=1,i_rd
               r_covtemp(i,j) = r_cov(i,j,i_pts)
            enddo
         enddo

         call gaussj(r_covtemp,i_rd,i_rd,r_b,1,1)

c     get the required jacobian matrix

         call funcs(i_pts,i_rd,i_fp,r_vecin(1,i_pts),i_np,r_a,r_amat)

c         do i=1,i_rd
c            do j=1,i_np
c               write(*,*)  'i,j,r_amat = ',i,j,r_amat(i,j)
c            enddo
c         enddo

c     multiply amat transpose by the inverse cov matrix

         do i=1,i_np
            do j=1,i_rd
               r_am(i,j) = 0.0
               do k=1,i_rd
                  r_am(i,j) = r_am(i,j) + r_amat(k,i)*r_covtemp(k,j)
               enddo
            enddo
         enddo

c         do i=1,i_np
c            do j=1,i_rd
c               write(*,*)  'i,j,r_am = ',i,j,r_am(i,j)
c            enddo
c         enddo

c     multiply am by amat

         do i=1,i_np
            do j=1,i_np
               do k=1,i_rd
                  r_u(i,j) = r_u(i,j) + r_am(i,k)*r_amat(k,j)
               enddo
            enddo
         enddo

c     multilpy am by vobs


c         write(*,*)  'r_vobs,i_pts = ',i_pts,r_vobs(1,i_pts),r_vobs(2,i_pts)
         do i=1,i_np
            do k=1,i_rd
               r_ptot(i) = r_ptot(i) + r_am(i,k)*r_vobs(k,i_pts)
            enddo
         enddo

      enddo   !i_pts

c     print out vector r_ptot

c      do i=1,i_np
c         r_ptot(i) = r_ptot(i)
c         write(*,*)  'i,r_ptot = ', i,r_ptot(i)
c      enddo

c      write(6,'(a)') ''
c      write(6,'(a)') 'r_ptot ='
c      write(6,'(9(e12.6,x))') (r_ptot(i), i = 1, i_np)

c     find the SVD of the r_u matrix

c      write(6,'(a)') ''
c      write(6,'(a)') 'r_u before decomp. ='
c         do i=1,i_np
c            write(6,'(9(e12.6,x))') (r_u(i,j), j = 1, i_np)
c            do j=1,i_np
c               r_u(i,j) = r_u(i,j)/1.d7
c               r_ucp(i,j) = r_u(i,j)
c               r_u(i,j) = r_u(i,j)/i_mp
c               write(*,*)  'i,j,r_u = ',i,j,r_u(i,j)
c            enddo
c         enddo

      call svdcmp(r_u,i_np,i_np,i_np,i_np,r_w,r_v)

c      write(6,'(a)') ''
c      write(6,'(a)') 'r_u ='
c         do i=1,i_np
c            write(6,'(9(e12.6,x))') (r_u(i,j), j = 1, i_np)
c            do j=1,i_np
cc              write(*,*)  'i,j,r_u,r_v = ',i,j,r_u(i,j),r_v(i,j)
c            enddo
c         enddo

c      write(6,'(a)') ''
c      write(6,'(a)') 'r_v ='
c         do i=1,i_np
c            write(6,'(9(e12.6,x))') (r_v(i,j), j = 1, i_np)
c         enddo

c      write(6,'(a)') ''
c      write(6,'(a)') 'r_w ='
c      write(6,'(9(e12.6,x))') (r_w(i), i = 1, i_np)

c         do i=1,i_np
c            write(*,*)  'w = ',i,r_w(i)
c         enddo

c     kill off all the singular values

      r_wmax = 0.0
      do i=1,i_np
         if(r_w(i) .gt. r_wmax)then
            r_wmax = r_w(i)
         endif
      enddo
      r_thres = r_wmax*R_TOL
c      write(*,*)  'r_thres = ',r_thres

      do i=1,i_np
         if(r_w(i) .lt. r_thres)then
            r_w(i) = 0.0
         endif
      enddo

c      write(6,'(a)') ''
c      write(6,'(a)') 'r_w after killing singular ='
c      write(6,'(9(e12.6,x))') (r_w(i), i = 1, i_np)

c      do i=1,i_np
c         write(*,*)  'w = ',i,r_w(i)
c      enddo

c     verify the decomp is accurate

c      write(6,'(a)') ''
c      write(6,'(a)') 'verify decomp '
c      do i = 1, i_np
c         do j = 1, i_np
c            r_uck(i,j) = 0.d0
c            do k = 1, i_np
c               r_uck(i,j) = r_uck(i,j) + r_u(i,k)*r_v(j,k)*r_w(k)
c            enddo
c         enddo
cc         write(6,'(9(e12.6,x))') (r_uck(i,j), j = 1, i_np)
c      enddo

c     find max delta before original and matrix formed by decomp

c      r_max = -1.0d0
c      do i = 1, i_np
c         do j = 1, i_np
c            if (r_ucp(i,j) .gt. 1.0d-6) then
c               r_delta = abs((r_uck(i,j)-r_ucp(i,j))/r_ucp(i,j))
c               if (r_delta .gt. r_max) then
c                  r_max = r_delta
c                  i_row = i
c                  i_col = j
c               endif
c            endif
c         enddo
c      enddo
c      write(6,'(a)') ''
c      write(6,'(a,i2,i2,x,e12.6)') 'Max delta at (i,j) = ', i_row, i_col, r_max

c     get the product of VW'U

c      write(6,'(a)') ''
c      write(6,'(a)') 'r_inv ='
c      do i = 1, i_np
c         do j = 1, i_np
c            r_inv(i,j) = 0.d0
c            do k = 1, i_np
c               if (r_w(k) .gt. 1.d-10) then
c                  r_inv(i,j) = r_inv(i,j) + r_v(i,k)*r_u(j,k)/r_w(k)
c               endif
c            enddo
c         enddo
c         r_sum = 0.d0
c         do j = 1, i_np
c            r_sum = r_sum + r_inv(i,j)*r_ptot(j)
c         enddo
c         write(6,'(10(e12.6,x))') (r_inv(i,j), j = 1, i_np), r_sum
c      enddo
      
c     use the svbksb routine to solve for the desired parameters

      call svbksb(r_u,r_w,r_v,i_np,i_np,i_np,i_np,r_ptot,r_at2)

c     update the r_a vector

c      write(6,'(a)') 'r_at2 before scaling correction:'
c      write(6,'(9(e12.6,x))') (r_at2(i), i = 1, i_np)
      
      do i=1,i_np
         if (i.eq.i_basecdot .or. i.eq.i_basehdot) then
            r_at2(i) = -r_at2(i)*i_paramest(i)/r_scale1
         elseif (i.eq.i_basecddt .or. i.eq.i_basehddt) then
            r_at2(i) = -r_at2(i)*i_paramest(i)/r_scale2
         elseif (i.eq.i_azoff) then
            r_at2(i) = -r_at2(i)*i_paramest(i)/r_scale3
         elseif (i.eq.i_azscale) then
            r_at2(i) = -r_at2(i)*i_paramest(i)/r_scale4
         else 
            r_at2(i) = -r_at2(i)*i_paramest(i)
         endif
         r_a(i) = r_at2(i)/R_LAMBDA + r_a(i)
c         write(*,*)'a=',i,r_a(i),r_at2(i)
      enddo

c      write(6,'(a)') ''
c      write(6,'(a)') 'r_at2 ='
c      write(6,'(9(e12.6,x))') (r_at2(i), i = 1, i_np)

c      write(6,'(a)') ''
c      write(6,'(a)') 'r_a ='
c      write(6,'(9(e12.6,x))') (r_a(i), i = 1, i_np)

c     evaluate the chisq array (linearized version)

      if(l_chisq)then

c     loop over data points


         do i=1,i_rd
            r_chird(i) = 0.
         enddo
         r_chisq(1,0) = 0.0
         do i=1,i_mp

            call funcs(i,i_rd,i_fp,r_vecin(1,i),i_np,r_a,r_amat)

            do j=1,i_rd
               r_chisq(j,i) = 0.0
               do k=1,i_np
                  r_chisq(j,i) = r_chisq(j,i) + r_amat(j,k)*r_at2(k)
               enddo
c               write(*,*)  'r_chisq = ',i,j,r_chisq(j,i),r_vobs(j,i)
               r_chisq(j,i) = r_covtemp(j,j)*(r_chisq(j,i) - 
     +              r_vobs(j,i))**2
               r_chisq(1,0) = r_chisq(1,0) + r_chisq(j,i)
               r_chird(j) = r_chird(j) + r_chisq(j,i)
            enddo

         enddo                  !i_pts loop for chisq

         r_chisq(1,0) = sqrt(r_chisq(1,0)/(2.*i_mp))
         write(6,'(a,3(f15.7,x))') 'Chi Square Total/Range/Azimuth: ',r_chisq(1,0),sqrt(r_chird(1)/i_mp),sqrt(r_chird(2)/i_mp)

      endif
      
      end  

