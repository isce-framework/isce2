      subroutine fitoff

      use fortranUtils
      use fitoffState
!c Define variables for main program; n = columns, m = rows
      IMPLICIT NONE      
      integer n,i, mmax
      integer k, iter
      real*8 threshx,threshy
      real*8 numerator, denominator, per_soln_length
      real*8 per_soln_length_last, delta_length
      logical change

      !!Arrays needed for processing
      double precision, allocatable, dimension(:,:) :: a,a_old
      double precision, allocatable, dimension(:) :: resx,resy
      double precision, allocatable, dimension(:) :: b_old,b
      double precision, allocatable, dimension(:) :: data,c,e
      double precision, allocatable, dimension(:,:) :: u
      integer, allocatable, dimension(:) :: s


      integer m, np, mp
      real*8 v(nmax,nmax),w(nmax),x_prev(nmax)

      integer n2,m2
      real*8 toler

      integer p
      real*8 rmsx,rmsy,xsdev,ysdev,sdev

      real*8 d(2),f(2),r_rotang,r_rtod,pi
      real*8 r_u(2),r_u2,r_rot(2,2),r_aff(2,2),r_scale1,r_scale2,r_skew
      logical sing

      pi = getPI()
      r_rtod = 180.d0/pi
      change = .true.

      per_soln_length = 0.

      !!Save initial number of lines in mmax
      mmax = 2*(imax+2)

      do i=1,imax
        x2o(i) = x1o(i) + dx(i)
        y2o(i) = y1o(i) + dy(i)

!!        print *, x1o(i), dx(i), y1o(i), dy(i), snr(i), r_covac(i), r_covdn(i),r_covx(i)
      enddo

!!      print *,'Params: ', nsig, maxrms, minpoint, l1norm
!!      print *, 'Iters:', miniter, maxiter
      if(imax .lt. 2) then
         print *,'fitoff.F:  Need at least 2 points'
         goto 105
      endif


      !!!Allocate the arrays
      allocate(a(mmax,nmax))
      allocate(a_old(mmax, nmax))
      allocate(b(mmax))
      allocate(b_old(mmax))
      allocate(c(mmax))
      allocate(resx(mmax))
      allocate(resy(mmax))
      allocate(e(mmax))
      allocate(u(mmax,nmax))
      allocate(s(mmax))
      allocate(data(mmax))

!c now setup matrices to solve overdetermined system of equations:
!c  [x2]   [m1 m2]   [x1]                      [m5]
!c  [  ] = [     ] x [  ]       +              [  ]
!c  [y2]   [m3 m4]   [y1]                      [m6]
!c
!c   ^      ^         ^                         ^
!c   |      |         X = solution vector       |
!c   B     A = affine                        translation
!c vector  transformation matrix               vector

      do iter =1,maxiter+1
         do k = 1,(2*imax)
            if (k.le.imax) then

               !matrix B
               b(k) = x2o(k)
               !Matrix A              
               a(k,1)=x1o(k)
               a(k,2)=y1o(k)
               a(k,3)=0.0d0
               a(k,4)=0.0d0
               a(k,5)=1.0
               a(k,6)=0.0d0
            else
               !matrix B
               b(k) = y2o(k-imax)
               !matrix A
               a(k,1)=0.0d0 
               a(k,2)=0.0d0
               a(k,3)=x1o(k-imax)
               a(k,4)=y1o(k-imax)
               a(k,5)=0.0d0
               a(k,6)=1.0
            endif
         end do

         np = nmax
         mp = mmax

         if (.not.(l1norm)) then
!c use L2 Norm to compute M matrix, from Numerical Recipes (p. 57)
!c n = number of columns, m = number of rows

            n = 6
            m = 2*imax
            
!c save the A matrix before using svdcmp, because it will be destroyed
            do k = 1,np
               do i = 1, m
                  a_old(i,k) = a(i,k)
               end do
            end do
            
            do k = 1,m
               b_old(k) = b(k)
            end do
          
            call dsvdcmp(a,m,n,mp,np,w,v)       
            
            do k = 1,n
               do i = 1, m
                  u(i,k) = a(i,k)
               end do
            end do
            
            call dsvbksb(u,w,v,m,n,mp,np,b,x)
            
         endif

!c use L1 norm to compute M matrix

         if (l1norm) then

            n  = 6
            m  = 2*imax
            n2 = n + 2
            m2 = m + 2
            toler = 1.0d-20

!c save b and a arrays since they are destroyed in subroutines
            do k = 1,n
               do i = 1, m
                  a_old(i,k) = a(i,k)
               end do
            end do
            
            do k = 1,m
               b_old(k) = b(k)
            end do
           
            call L1(M,N,M2,N2,A,B,TOLER,X,E,S,nmax,mmax)
            
         endif

!c multiple A and X together and compute residues
         call mmul(M,N,A_OLD,X,C,nmax,mmax)

         do k = 1,imax
               resx(k) = c(k) - b_old(k)
               resy(k) = c(k+imax) - b_old(k+imax)
         end do 
         
         p    = imax
         rmsy = 0.0d0
         rmsx = 0.0d0

!c compute statistics for x coordinates: standard deviation, mean, & rms
         do k = 1,imax
            data(k)= resx(k)
            rmsx = rmsx + resx(k)**2.
         end do
        
         call dmoment(data,p,sdev)
         rmsx = sqrt(rmsx/imax)  
         xsdev = sdev
        
!         print *, 'Sdev1 :', sdev

!c compute statistics for y coordinates
                
         do k = 1,(imax)
            data(k)= resy(k)
            rmsy = rmsy + resy(k)**2.
         end do
         
         call dmoment(data,p,sdev)
         rmsy = sqrt(rmsy/imax)  
         ysdev = sdev
         
         if (rmsx.gt.maxrms) then
            threshx = nsig*xsdev
         else
            threshx = 99999
         endif

         if (rmsy.gt.maxrms) then
            threshy = nsig*ysdev
         else
            threshy = 99999
         endif

!         print *, 'Threshs: ', threshx, threshy, sdev
!c determine whether to remove points for next iteration
         if ((rmsx.gt.maxrms).or.(rmsy.gt.maxrms)) then
!c determine which points to save for next iteration
            i = 0
            do k = 1,imax
               if ((abs(resx(k)).lt.threshx)
     >              .and.(abs(resy(k)).lt.threshy)) then
                  i = i + 1
                  x2o(i) = x2o(k)
                  x1o(i) = x1o(k)
                  y2o(i) = y2o(k)
                  y1o(i) = y1o(k)
                  snr(i) = snr(k) 
                  r_covac(i) = r_covac(k)
                  r_covdn(i) = r_covdn(k)
                  r_covx(i)  = r_covx(k)
               endif
            end do
            imax = i
         endif

!c if fewer than minpoints, quit and output warning
         if (imax.le.minpoint) goto 97

!c if rms fit is good enough, then quit program
         if ((rmsx.lt.maxrms).and.(rmsy.lt.maxrms)) goto 99

         if (iter.gt.1) then
            numerator   = 0.0d0
            denominator = 0.0d0
!c if the soln. length does not change between iterations, and solution fit
!c doesn't match specified parameters, then quit

            do k = 1,6
               numerator   = numerator + (x(k) - x_prev(k))**2.
               denominator = (x_prev(k))**2. + denominator
            end do
            per_soln_length = sqrt(numerator/denominator)*100.
         end if
         
         if (iter.ge.miniter)  then 
            delta_length = (per_soln_length - 
     >           per_soln_length_last)
            
            if ((delta_length.eq.0).and.
     >           ((rmsx.gt.maxrms).or.(rmsy.gt.maxrms))) then
               change = .false.
               goto 96
            endif
            
         endif

         per_soln_length_last = per_soln_length
         
         do k = 1,6
            x_prev(k) = x(k)
         end do


      end do

!c exceeded maximum number of iterations, output garbage
      print *,'WARNING: Exceeded maximum number of iterations.'

!c solution length not changing and fit parameters not achieved
 96   if (.not.change) then
         print *,'WARNING: Solution length is not changing,'
         print *,'but does not meet fit criteria'
      endif

!c Fewer than minimum number of points, output garbage
 97   if (imax.le.minpoint) then
         print *, 'WARNING: Fewer than minimum points, there are only'
     >        ,imax
      endif

 99      print *,' '
      if (((iter.lt.maxiter).and.(imax.gt.minpoint)).and.
     >        (change)) then
         print *, '     << Fitoff Program >>    '

         print *, ' '
         print *, 'Number of points remaining =', imax
         print *, ' '
         print *, 'RMS in X = ', rmsx, '  RMS in Y = ', rmsy
         print *, ' '

!c     Decompose matrix and examine residuals

         print *, ' '
         print *, '      Matrix  Analysis       '
         print *, ' '
         print *, ' Affine Matrix '
         print *, ' '
         print *, x(1), x(2)
         print *, x(3), x(4)

 101     format(1x,f15.10,1x,f15.10)
         print *, ' '

         print *, 'Translation Vector'
         print *, ' '
         print *, x(5),x(6)

 102     format(1x,f11.3,1x,f11.3,1x)
         
!c     decompose affine matrix to find rotation matrix using QR decomposition
!c     R is an upper triangular matrix and Q is an orthogonal matrix such
!c     that A = QR.  For our 2 X 2 matrix we can consider
!c      T
!c     Q A = R, where Q is a Housholder matrix, which is also a rotation matrix
!c     Subroutine qrdcmp ( Numerical recipes, pg 92) returns the u vectors
!c     used to compute Q1 in r_aff(1,1). r_aff(1,2), d(1) and d(2) are
!c     the diagonal terms of the R matrix, while r_aff(1,2) is the other
!c     point in the R matrix and these can be used to find the scale and
!c     skew terms 
         
         r_aff(1,1) = x(1)
         r_aff(1,2) = x(2)
         r_aff(2,1) = x(3)
         r_aff(2,2) = x(4)
         
         call qrdcmp(r_aff,2,2,f,d,sing)
         
         r_u(1) = r_aff(1,1)
         r_u(2) = r_aff(2,1)
         
         r_u2 = .5d0*(r_u(1)**2  + r_u(2)**2)
         
         r_rot(1,1) = (1.d0 - (r_u(1)**2/r_u2))
         r_rot(1,2) = -(r_u(1)*r_u(2))/r_u2
         r_rot(2,1) = -(r_u(1)*r_u(2))/r_u2
         r_rot(2,2) = (1.d0 - (r_u(2)**2/r_u2))
         
         if(d(1) .lt. 0)then
            r_rot(1,1) = -r_rot(1,1)
            r_rot(2,1) = -r_rot(2,1)
            d(1) = -d(1)
            r_aff(1,2) = -r_aff(1,2)
         elseif(d(2) .lt. 0)then
            r_rot(1,2) = -r_rot(1,2)
            r_rot(2,2) = -r_rot(2,2)
            d(2) = -d(2)
         endif         
         
         r_scale1 = abs(d(1))
         r_scale2 = abs(d(2))
         
         r_skew = r_aff(1,2)/d(1)
         
         r_rotang = atan2(r_rot(2,1),r_rot(1,1))
         
         print *, ' '
         print *, ' Rotation Matrix '
         print *, ' '
         print *, r_rot(1,1),r_rot(1,2)
         print *, r_rot(2,1),r_rot(2,2)

         print *, ' '

         print *, 'Rotation Angle (deg) = ',r_rotang*r_rtod
         print *, ' '
         print *, ' Axis Scale Factors'

         print *, ' '
         print *, r_scale1,r_scale2
 103     format(1x,f11.7,1x,f11.7)
         print *,' '
         print *,' Skew Term'

         print *, ' '
         print *, r_skew

 104     format(1x,f11.7)

      endif

      !!Deallocate arrays
      deallocate(a,a_old)
      deallocate(b,b_old)
      deallocate(c,resx,resy)
      deallocate(e,u,s,data)

 105  end

!C     ALGORITHM 478 COLLECTED ALGORITHMS FROM ACM.
!C     ALGORITHM APPEARED IN COMM. ACM, VOL. 17, NO. 06,
!C     P. 319.
!C  NOTE:  this version is modified to allow double precision
      SUBROUTINE L1(M,N,M2,N2,A,B,TOLER,X,E,S,NMAX,MMAX)       
!C THIS SUBROUTINE USES A MODIFICATION OF THE SIMPLEX METHOD
!C OF LINEAR PROGRAMMING TO CALCULATE AN L1 SOLUTION TO AN
!C OVER-DETERMINED SYSTEM OF LINEAR EQUATIONS.
!C DESCRIPTION OF PARAMETERS.
!C M      NUMBER OF EQUATIONS.
!C N      NUMBER OF UNKNOWNS (M.GE.N).
!C M2     SET EQUAL TO M+2 FOR ADJUSTABLE DIMENSIONS.
!C N2     SET EQUAL TO N+2 FOR ADJUSTABLE DIMENSIONS.
!C A      TWO DIMENSIONAL REAL ARRAY OF SIZE (M2,N2).
!C        ON ENTRY, THE COEFFICIENTS OF THE MATRIX MUST BE
!C        STORED IN THE FIRST M ROWS AND N COLUMNS OF A.
!C        THESE VALUES ARE DESTROYED BY THE SUBROUTINE.
!C B      ONE DIMENSIONAL REAL ARRAY OF SIZE M. ON ENTRY, B
!C        MUST CONTAIN THE RIGHT HAND SIDE OF THE EQUATIONS.
!C        THESE VALUES ARE DESTROYED BY THE SUBROUTINE.
!C TOLER  A SMALL POSITIVE TOLERANCE. EMPIRICAL EVIDENCE
!C        SUGGESTS TOLER=10**(-D*2/3) WHERE D REPRESENTS
!C        THE NUMBER OF DECIMAL DIGITS OF ACCURACY AVALABLE
!C        (SEE DESCRIPTION).
!C X      ONE DIMENSIONAL REAL ARRAY OF SIZE N. ON EXIT, THIS
!C        ARRAY CONTAINS A SOLUTION TO THE L1 PROBLEM.
!C E      ONE DIMENSIONAL REAL ARRAY OF SIZE M. ON EXIT, THIS
!C        ARRAY CONTAINS THE RESIDUALS IN THE EQUATIONS.
!C S      INTEGER ARRAY OF SIZE M USED FOR WORKSPACE.
!C ON EXIT FROM THE SUBROUTINE, THE ARRAY A CONTAINS THE
!C FOLLOWING INFORMATION.
!C A(M+1,N+1)  THE MINIMUM SUM OF THE ABSOLUTE VALUES OF
!C             THE RESIDUALS.
!C A(M+1,N+2)  THE RANK OF THE MATRIX OF COEFFICIENTS.
!C A(M+2,N+1)  EXIT CODE WITH VALUES.
!C             0 - OPTIMAL SOLUTION WHICH IS PROBABLY NON-
!C                 UNIQUE (SEE DESCRIPTION).
!C             1 - UNIQUE OPTIMAL SOLUTION.
!C             2 - CALCULATIONS TERMINATED PREMATURELY DUE TO
!C                 ROUNDING ERRORS.
!C A(M+2,N+2)  NUMBER OF SIMPLEX ITERATIONS PERFORMED.
      Implicit None
      INTEGER m,m1,m2,n,n1,n2,NMAX,MMAX
      double precision SUM, MIN, MAX
      double precision :: A(Mmax,Nmax)
      double precision :: X(Nmax), E(Mmax), B(Mmax)
!!      REAL*8 MIN, MAX, A(Mmax,Nmax), X(Nmax), E(Mmax), B(Mmax)
      integer :: S(Mmax)
      INTEGER OUT
      LOGICAL STAGE, TEST
!c define variables in program whose type were assumed implicitly
      integer i,j,kr,k,kl,kount,in,l
      double precision d, pivot,toler,big
!C BIG MUST BE SET EQUAL TO ANY VERY LARGE REAL CONSTANT.
!C ITS VALUE HERE IS APPROPRIATE FOR THE IBM 370.
!c      DATA BIG/1.E75/
!C ITS VALUE HERE IS APPROPRIATE FOR SGI
      DATA BIG/1.E38/
!C INITIALIZATION.
      M1 = M + 1
      N1 = N + 1
      DO 10 J=1,N
        A(M2,J) = J
        X(J) = 0.0d0
   10 CONTINUE
      DO 40 I=1,M
        A(I,N2) = N + I
        A(I,N1) = B(I)
        IF (B(I).GE.0.0d0) GO TO 30
        DO 20 J=1,N2
          A(I,J) = -A(I,J)
   20   CONTINUE
   30   E(I) = 0.0d0
   40 CONTINUE
!C COMPUTE THE MARGINAL COSTS.
      DO 60 J=1,N1
        SUM = 0.0D0
        DO 50 I=1,M
          SUM = SUM + A(I,J)
   50   CONTINUE
        A(M1,J) = SUM
   60 CONTINUE
!C STAGE I.
!C DETERMINE THE VECTOR TO ENTER THE BASIS.
      STAGE = .TRUE.
      KOUNT = 0
      KR = 1
      KL = 1
   70 MAX = -1.
      DO 80 J=KR,N
        IF (ABS(A(M2,J)).GT.N) GO TO 80
        D = ABS(A(M1,J))
        IF (D.LE.MAX) GO TO 80
        MAX = D
        IN = J
   80 CONTINUE
      IF (A(M1,IN).GE.0.0d0) GO TO 100
      DO 90 I=1,M2
        A(I,IN) = -A(I,IN)
   90 CONTINUE
!C DETERMINE THE VECTOR TO LEAVE THE BASIS.
  100 K = 0
      DO 110 I=KL,M
        D = A(I,IN)
        IF (D.LE.TOLER) GO TO 110
        K = K + 1
        B(K) = A(I,N1)/D
        S(K) = I
        TEST = .TRUE.
  110 CONTINUE
  120 IF (K.GT.0) GO TO 130
      TEST = .FALSE.
      GO TO 150
  130 MIN = BIG
      DO 140 I=1,K
        IF (B(I).GE.MIN) GO TO 140
        J = I
        MIN = B(I)
        OUT = S(I)
  140 CONTINUE
      B(J) = B(K)
      S(J) = S(K)
      K = K - 1
!C CHECK FOR LINEAR DEPENDENCE IN STAGE I.
  150 IF (TEST .OR. .NOT.STAGE) GO TO 170
      DO 160 I=1,M2
        D = A(I,KR)
        A(I,KR) = A(I,IN)
        A(I,IN) = D
  160 CONTINUE
      KR = KR + 1
      GO TO 260
  170 IF (TEST) GO TO 180
      A(M2,N1) = 2.
      GO TO 350
  180 PIVOT = A(OUT,IN)
      IF (A(M1,IN)-PIVOT-PIVOT.LE.TOLER) GO TO 200
      DO 190 J=KR,N1
        D = A(OUT,J)
        A(M1,J) = A(M1,J) - D - D
        A(OUT,J) = -D
  190 CONTINUE
      A(OUT,N2) = -A(OUT,N2)
      GO TO 120
!C PIVOT ON A(OUT,IN).
  200 DO 210 J=KR,N1
        IF (J.EQ.IN) GO TO 210
        A(OUT,J) = A(OUT,J)/PIVOT
  210 CONTINUE
!c      DO 230 I=1,M1
!c        IF (I.EQ.OUT) GO TO 230
!c        D = A(I,IN)
!c        DO 220 J=KR,N1
!c          IF (J.EQ.IN) GO TO 220
!c          A(I,J) = A(I,J) - D*A(OUT,J)
!c  220   CONTINUE
!c  230 CONTINUE
!c impliment time saving change suggested in Barrodale and Roberts - collected 
!c algorithms from CACM
      DO 220 J = KR,N1
         IF (J.EQ.IN) GO TO 220
         CALL COL(A(1,J),A(1,IN),A(OUT,J),M1,OUT)
 220      CONTINUE
      DO 240 I=1,M1
        IF (I.EQ.OUT) GO TO 240
        A(I,IN) = -A(I,IN)/PIVOT
  240 CONTINUE
      A(OUT,IN) = 1./PIVOT
      D = A(OUT,N2)
      A(OUT,N2) = A(M2,IN)
      A(M2,IN) = D
      KOUNT = KOUNT + 1
      IF (.NOT.STAGE) GO TO 270
!C INTERCHANGE ROWS IN STAGE I.
      KL = KL + 1
      DO 250 J=KR,N2
        D = A(OUT,J)
        A(OUT,J) = A(KOUNT,J)
        A(KOUNT,J) = D
  250 CONTINUE
  260 IF (KOUNT+KR.NE.N1) GO TO 70
!C STAGE II.
      STAGE = .FALSE.
!C DETERMINE THE VECTOR TO ENTER THE BASIS.
  270 MAX = -BIG
      DO 290 J=KR,N
        D = A(M1,J)
        IF (D.GE.0.0d0) GO TO 280
        IF (D.GT.(-2.)) GO TO 290
        D = -D - 2.
  280   IF (D.LE.MAX) GO TO 290
        MAX = D
        IN = J
  290 CONTINUE
      IF (MAX.LE.TOLER) GO TO 310
      IF (A(M1,IN).GT.0.0d0) GO TO 100
      DO 300 I=1,M2
        A(I,IN) = -A(I,IN)
  300 CONTINUE
      A(M1,IN) = A(M1,IN) - 2.
      GO TO 100
!C PREPARE OUTPUT.
  310 L = KL - 1
      DO 330 I=1,L
        IF (A(I,N1).GE.0.0d0) GO TO 330
        DO 320 J=KR,N2
          A(I,J) = -A(I,J)
  320   CONTINUE
  330 CONTINUE
      A(M2,N1) = 0.0d0
      IF (KR.NE.1) GO TO 350
      DO 340 J=1,N
        D = ABS(A(M1,J))
        IF (D.LE.TOLER .OR. 2.-D.LE.TOLER) GO TO 350
  340 CONTINUE
      A(M2,N1) = 1.
  350 DO 380 I=1,M
        K = A(I,N2)
        D = A(I,N1)
        IF (K.GT.0) GO TO 360
        K = -K
        D = -D
  360   IF (I.GE.KL) GO TO 370
        X(K) = D
        GO TO 380
  370   K = K - N
        E(K) = D
  380 CONTINUE
      A(M2,N2) = KOUNT
      A(M1,N2) = N1 - KR
      SUM = 0.0D0
      DO 390 I=KL,M
        SUM = SUM + A(I,N1)
  390 CONTINUE
      A(M1,N1) = SUM
      RETURN
      END

      SUBROUTINE COL(V1,V2,MLT,M1,IOUT)
      IMPLICIT NONE
      INTEGER M1,I,IOUT
      REAL*8 V1(M1),V2(M1),MLT
      DO 1 I = 1,M1
         IF (I.EQ.IOUT) GO TO 1
         V1(I)=V1(I)-V2(I)*MLT
 1       CONTINUE
         RETURN
         END

!c The following three programs are used to find the L2 norm
      SUBROUTINE dsvbksb(u,w,v,m,n,mp,np,b,x)
      Implicit None
      INTEGER m,mp,n,np
!!      REAL*8 b(mmax),u(mmax,nmax),v(nmax,nmax),w(nmax),x(nmax)
      double precision :: b(mp),w(np),x(np)
      double precision :: u(mp,np),v(np,np)
      INTEGER i,j,jj
      DOUBLE PRECISION s,tmp(np)
      do 12 j=1,n
        s=0.0d0
        if(w(j).ne.0.0d0)then
          do 11 i=1,m
            s=s+u(i,j)*b(i)
11        continue
          s=s/w(j)
        endif
        tmp(j)=s
12    continue
      do 14 j=1,n
        s=0.0d0
        do 13 jj=1,n
          s=s+v(j,jj)*tmp(jj)
13      continue
        x(j)=s
14    continue
      return
      END

      SUBROUTINE dsvdcmp(a,m,n,mp,np,w,v)
      Implicit None
      INTEGER m,mp,n,np
!c      DOUBLE PRECISION a(mp,np),v(np,np),w(np)
!!      REAL*8 a(mmax,nmax),v(nmax,nmax),w(nmax)
      double precision :: w(np)
      double precision :: a(mp,np),v(np,np)
      double precision :: rv1(np)
!CU    USES dpythag
      INTEGER i,its,j,jj,k,l,nm
      DOUBLE PRECISION anorm,c,f,g,h,s,scale,x,y,z,dpythag

      g=0.0d0
      scale=0.0d0
      anorm=0.0d0
      do 25 i=1,n
        l=i+1
        rv1(i)=scale*g
        g=0.0d0
        s=0.0d0
        scale=0.0d0
        if(i.le.m)then
          do 11 k=i,m
            scale=scale+abs(a(k,i))
11        continue
          if(scale.ne.0.0d0)then
            do 12 k=i,m
              a(k,i)=a(k,i)/scale
              s=s+a(k,i)*a(k,i)
12          continue
            f=a(i,i)
            g=-sign(sqrt(s),f)
            h=f*g-s
            a(i,i)=f-g
            do 15 j=l,n
              s=0.0d0
              do 13 k=i,m
                s=s+a(k,i)*a(k,j)
13            continue
              f=s/h
              do 14 k=i,m
                a(k,j)=a(k,j)+f*a(k,i)
14            continue
15          continue
            do 16 k=i,m
              a(k,i)=scale*a(k,i)
16          continue
          endif
        endif
        w(i)=scale *g
        g=0.0d0
        s=0.0d0
        scale=0.0d0
        if((i.le.m).and.(i.ne.n))then
          do 17 k=l,n
            scale=scale+abs(a(i,k))
17        continue
          if(scale.ne.0.0d0)then
            do 18 k=l,n
              a(i,k)=a(i,k)/scale
              s=s+a(i,k)*a(i,k)
18          continue
            f=a(i,l)
            g=-sign(sqrt(s),f)
            h=f*g-s
            a(i,l)=f-g
            do 19 k=l,n
              rv1(k)=a(i,k)/h
19          continue
            do 23 j=l,m
              s=0.0d0
              do 21 k=l,n
                s=s+a(j,k)*a(i,k)
21            continue
              do 22 k=l,n
                a(j,k)=a(j,k)+s*rv1(k)
22            continue
23          continue
            do 24 k=l,n
              a(i,k)=scale*a(i,k)
24          continue
          endif
        endif
        anorm=max(anorm,(abs(w(i))+abs(rv1(i))))
25    continue
      do 32 i=n,1,-1
        if(i.lt.n)then
          if(g.ne.0.0d0)then
            do 26 j=l,n
              v(j,i)=(a(i,j)/a(i,l))/g
26          continue
            do 29 j=l,n
              s=0.0d0
              do 27 k=l,n
                s=s+a(i,k)*v(k,j)
27            continue
              do 28 k=l,n
                v(k,j)=v(k,j)+s*v(k,i)
28            continue
29          continue
          endif
          do 31 j=l,n
            v(i,j)=0.0d0
            v(j,i)=0.0d0
31        continue
        endif
        v(i,i)=1.0d0
        g=rv1(i)
        l=i
32    continue
      do 39 i=min(m,n),1,-1
        l=i+1
        g=w(i)
        do 33 j=l,n
          a(i,j)=0.0d0
33      continue
        if(g.ne.0.0d0)then
          g=1.0d0/g
          do 36 j=l,n
            s=0.0d0
            do 34 k=l,m
              s=s+a(k,i)*a(k,j)
34          continue
            f=(s/a(i,i))*g
            do 35 k=i,m
              a(k,j)=a(k,j)+f*a(k,i)
35          continue
36        continue
          do 37 j=i,m
            a(j,i)=a(j,i)*g
37        continue
        else
          do 38 j= i,m
            a(j,i)=0.0d0
38        continue
        endif
        a(i,i)=a(i,i)+1.0d0
39    continue
      do 49 k=n,1,-1
        do 48 its=1,30
          do 41 l=k,1,-1
            nm=l-1
            if((abs(rv1(l))+anorm).eq.anorm)  goto 2
            if((abs(w(nm))+anorm).eq.anorm)  goto 1
41        continue
1         c=0.0d0
          s=1.0d0
          do 43 i=l,k
            f=s*rv1(i)
            rv1(i)=c*rv1(i)
            if((abs(f)+anorm).eq.anorm) goto 2
            g=w(i)
            h=dpythag(f,g)
            w(i)=h
            h=1.0d0/h
            c= (g*h)
            s=-(f*h)
            do 42 j=1,m
              y=a(j,nm)
              z=a(j,i)
              a(j,nm)=(y*c)+(z*s)
              a(j,i)=-(y*s)+(z*c)
42          continue
43        continue
2         z=w(k)
          if(l.eq.k)then
            if(z.lt.0.0d0)then
              w(k)=-z
              do 44 j=1,n
                v(j,k)=-v(j,k)
44            continue
            endif
            goto 3
          endif
!          if(its.eq.30) pause 'no convergence in svdcmp'
          if(its.eq.30) then
             write (6,*) 'fitoff: no convergence in svdcmp, quitting'
             stop
          endif
          x=w(l)
          nm=k-1
          y=w(nm)
          g=rv1(nm)
          h=rv1(k)
          f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0d0*h*y)
          g=dpythag(f,1.0d0)
          f=((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x
          c=1.0d0
          s=1.0d0
          do 47 j=l,nm
            i=j+1
            g=rv1(i)
            y=w(i)
            h=s*g
            g=c*g
            z=dpythag(f,h)
            rv1(j)=z
            c=f/z
            s=h/z
            f= (x*c)+(g*s)
            g=-(x*s)+(g*c)
            h=y*s
            y=y*c
            do 45 jj=1,n
              x=v(jj,j)
              z=v(jj,i)
              v(jj,j)= (x*c)+(z*s)
              v(jj,i)=-(x*s)+(z*c)
45          continue
            z=dpythag(f,h)
            w(j)=z
            if(z.ne.0.0d0)then
              z=1.0d0/z
              c=f*z
              s=h*z
            endif
            f= (c*g)+(s*y)
            x=-(s*g)+(c*y)
            do 46 jj=1,m
              y=a(jj,j)
              z=a(jj,i)
              a(jj,j)= (y*c)+(z*s)
              a(jj,i)=-(y*s)+(z*c)
46          continue
47        continue
          rv1(l)=0.0d0
          rv1(k)=f
          w(k)=x
48      continue
3       continue
49    continue
      return
      END

      FUNCTION dpythag(a,b)
      Implicit None
!c      DOUBLE PRECISION a,b,dpythag
!c      DOUBLE PRECISION absa,absb
      Real*8 a,b,dpythag
      Real*8 absa,absb
      absa=abs(a)
      absb=abs(b)
      if(absa.gt.absb)then
        dpythag=absa*sqrt(1.0d0+(absb/absa)**2)
      else
        if(absb.eq.0.0d0)then
          dpythag=0.0d0
        else
          dpythag=absb*sqrt(1.0d0+(absa/absb)**2)
        endif
      endif
      return
      END

      SUBROUTINE MMUL (M,N,A_OLD,X,C,nmax,mmax)
      Implicit None

!C     *****PARAMETERS:
      Integer nmax, mmax, M, N
      double precision :: a_old(MMAX,NMAX)
      double precision :: x(NMAX),c(MMAX)
!!      REAL*8 a_old(mmax,nmax),x(nmax),c(mmax)

      INTEGER NA,NB,NC,L

!C     *****LOCAL VARIABLES:
      INTEGER I,K

      NA = M
      NB = nmax
      NC = M
      N  = nmax
      L  = 1

!C     *****SUBROUTINES CALLED:
!C     NONE
!C
!C     ------------------------------------------------------------------
!C
!C     *****PURPOSE:
!C     THIS SUBROUTINE COMPUTES THE MATRIX PRODUCT A*B AND STORES THE
!C        RESULT IN THE ARRAY C.  A IS M X N, B IS N X L, AND C IS
!C        M X L.  THE ARRAY C MUST BE DISTINCT FROM BOTH A AND B.
!C
!C     *****PARAMETER DESCRIPTION:
!C     ON INPUT:
!C        NA    ROW DIMENSION OF THE ARRAY CONTAINING A AS DECLARED
!C              IN THE CALLING PROGRAM DIMENSION STATEMENT;
!C
!C        NB    ROW DIMENSION OF THE ARRAY CONTAINING B AS DECLARED
!C              IN THE CALLING PROGRAM DIMENSION STATEMENT;
!C
!C        NC    ROW DIMENSION OF THE ARRAY CONTAINING C AS DECLARED
!C              IN THE CALLING PROGRAM DIMENSION STATEMENT;
!C
!C        L     NUMBER OF COLUMNS OF THE MATRICES B AND C;
!C
!C        M     NUMBER OF ROWS OF THE MATRICES A AND C;
!C
!C        N     NUMBER OF COLUMNS OF THE MATRIX A AND NUMBER OF ROWS
!C              OF THE MATRIX B;
!C
!C        A     AN M X N MATRIX;
!C
!C        B     AN N X L MATRIX.
!C
!C     ON OUTPUT:
!C
!C        C     AN M X L ARRAY CONTAINING A*B.
!C
!C     *****HISTORY:
!C     WRITTEN BY ALAN J. LAUB (ELEC. SYS. LAB., M.I.T., RM. 35-331,
!C     CAMBRIDGE, MA 02139,  PH.: (617)-253-2125), SEPTEMBER 1977.
!C     MOST RECENT VERSION: SEP. 21, 1977.
!C
!C     ------------------------------------------------------------------
!C
         DO 10 I=1,M
            C(I)=0.0d0
10       CONTINUE
         DO 30 K=1,N
            DO 20 I=1,M
               C(I)=C(I)+a_old(I,K)*x(K)
20          CONTINUE
30       CONTINUE
      RETURN

      END

!c Modify Numerical Recipes program moment.f to compute only 
!c standard deviation and allow double precision
      SUBROUTINE dmoment(data,p,sdev)
      Implicit None
      INTEGER p
      REAL*8 adev,ave,curt,sdev,skew,var,data(p)
      INTEGER j
      REAL*8 t,s,ep
!      if(p.le.1)pause 'p must be at least 2 in moment'
      if(p.le.1) then
         write (6,*) 'fitoff: p must be at least 2 in moment'
         write (6,*) '      culling points failed'
         stop
      endif
      s=0.0d0
      do 11 j=1,p
        s=s+data(j)
11    continue
      ave=s/p
      adev=0.0d0
      var=0.0d0
      skew=0.0d0
      curt=0.0d0
      ep=0.
      do 12 j=1,p
        s=data(j)-ave
        t=s*s
        var=var+t
12    continue
      adev=adev/p
      var=(var-ep**2/p)/(p-1)
      sdev=sqrt(var)
      return
      END

!c This program is used to find the rotation matrix from the affine matrix
      SUBROUTINE qrdcmp(a,n,np,c,d,sing)
      INTEGER n,np
      REAL*8 a(np,np),c(n),d(n)
      LOGICAL sing
      INTEGER i,j,k
      REAL*8 scale,sigma,sum,tau
      sing=.false.
      scale=0.
      do 17 k=1,n-1
        do 11 i=k,n
          scale=max(scale,abs(a(i,k)))
11      continue
        if(scale.eq.0.)then
          sing=.true.
          c(k)=0.
          d(k)=0.
        else
          do 12 i=k,n
            a(i,k)=a(i,k)/scale
12        continue
          sum=0.
          do 13 i=k,n
            sum=sum+a(i,k)**2
13        continue
          sigma=sign(sqrt(sum),a(k,k))
          a(k,k)=a(k,k)+sigma
          c(k)=sigma*a(k,k)
          d(k)=-scale*sigma
          do 16 j=k+1,n
            sum=0.
            do 14 i=k,n
              sum=sum+a(i,k)*a(i,j)
14          continue
            tau=sum/c(k)
            do 15 i=k,n
              a(i,j)=a(i,j)-tau*a(i,k)
15          continue
16        continue
        endif
17    continue
      d(n)=a(n,n)
      if(d(n).eq.0.)sing=.true.
      return
      END
!C  (C) Copr. 1986-92 Numerical Recipes Software $23#1yR.3Z9.
