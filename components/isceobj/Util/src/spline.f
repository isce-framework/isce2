!!!! Derived from interp_2p5min.f from http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/interp_2p5min.f
!!!! Modified for ISCE by Piyush Agram

      FUNCTION IFRAC(R)
        implicit none
        double precision :: R
        integer :: IFRAC

        IFRAC=R
        IF (R.GE.0) RETURN
        IF (R.EQ.IFRAC) RETURN
        IFRAC = IFRAC - 1
       END FUNCTION IFRAC

      SUBROUTINE INITSPLINE(Y, N, R, Q)

          implicit none
          integer :: N,K
          double precision, dimension(N) :: Y,R,Q
          double precision :: P
          Q(1) = 0.0
          R(1) = 0.0
          DO  K = 2, N-1
            P = Q(K-1)/2+2
            Q(K) = -0.5/P
            R(K) = (3*(Y(K+1)-2*Y(K)+Y(K-1)) - R(K-1)/2)/P
          ENDDO

          R(N) = 0.0
          DO K = N-1, 2, -1
            R(K) = Q(K)*R(K+1)+R(K)
          END DO
      END SUBROUTINE INITSPLINE
      
      
      FUNCTION SPLINE(X, Y, N, R)
      
          implicit none
          integer :: N,J
          integer :: IFRAC
          double precision, dimension(N) :: Y,R
          double precision :: X, XX
          double precision :: SPLINE
      
          IF (X.LT.1) THEN
            SPLINE = Y(1) + (X-1)*(Y(2)-Y(1)-R(2)/6)
          ELSEIF (X.GT.N) THEN
            SPLINE = Y(N) + (X-N)*(Y(N)-Y(N-1)+R(N-1)/6)
          ELSE
            J = IFRAC(X)
            XX = X - J
            SPLINE = Y(J) + XX * ((Y(J+1)-Y(J)-R(J)/3-R(J+1)/6) + XX * (R(J)/2 + XX * (R(J+1)-R(J))/6))
          ENDIF
      END FUNCTION SPLINE
      
      

      function interp2DSpline(order,nx,ny,z,x,y)

        implicit none
     
        integer :: order
        integer :: nx,ny
        real*4, dimension(ny,nx) :: z
        double precision :: x, y
        real*4:: interp2DSpline
        double precision :: SPLINE

        integer :: MINORDER, MAXORDER
        parameter(MINORDER=3, MAXORDER=20)
        double precision, dimension(MAXORDER) :: A, R, Q, HC
        integer :: I,J,I0, J0, II, JJ,INDI,INDJ
        double precision :: temp,inx,iny 

        LOGICAL LODD

        if ((order.lt.MINORDER).or.(order.gt.MAXORDER)) then
          print *, 'Spline order must be between ', MINORDER, ' and ', MAXORDER
          stop
        endif

        LODD=(order/2)*2.NE.order
        IF(LODD) THEN
            I0=y-0.5
            J0=x-0.5
        ELSE
            I0=y
            J0=x
        ENDIF

        I0=I0-order/2+1
        J0=J0-order/2+1
        II=I0+order-1
        JJ=J0+order-1

!!      print *, 'Y: ', y, I0, II, Z(100,100)
!!      print *, 'X: ', x, J0, JJ, Z(200,200)

      DO I=1,order
          INDI = min( max(I0+I,1), ny)
          DO J=1,order
            INDJ = min( max(J0+J,1), nx)
!!            print *, 'IND: ', INDI, INDJ, Z(1200,1200), Z(INDI,INDJ)
            A(J)=Z(INDI,INDJ)
          ENDDO

!!          print *, 'I: ', i, x-J0+1.
          CALL INITSPLINE(A,order,R,Q)
          HC(I) = SPLINE(x-J0,A,order,R)
      ENDDO
    
!!      print *, 'J: ', j, y-I0+1.
      CALL INITSPLINE(HC,order,R,Q)
      temp = SPLINE(y-I0,HC,order,R)
      interp2DSpline = sngl(temp)
      RETURN
      END FUNCTION interp2Dspline
      

