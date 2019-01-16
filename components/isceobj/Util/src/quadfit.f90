       subroutine quadfit(xin,yin,ndata,poly)

! Polynomial to be fitted:
! Y = a(0) + a(1).X + a(2).X^2 + ... + a(m).X^m

         USE lsq
         IMPLICIT NONE
         
         REAL*8    :: x(10000),y(10000),xrow(0:20), wt = 1.0, beta(0:20), &
                       var, covmat(231), sterr(0:20), totalSS, center
         real*4  poly(3)
         REAL*8    :: xin(10000), yin(10000)
         INTEGER   :: i, ier, iostatus, j, m, n, ndata
         LOGICAL   :: fit_const = .TRUE., lindep(0:20), xfirst

      do i=1,ndata
         y(i)=yin(i)
         x(i)=xin(i)
      end do

      n=ndata
      m=2

! Least-squares calculations

      CALL startup(m, fit_const)
      DO i = 1, n
         xrow(0) = 1.0
         DO j = 1, m
            xrow(j) = x(i) * xrow(j-1)
         END DO
         CALL includ(wt, xrow, y(i))
      END DO

      CALL sing(lindep, ier)
      IF (ier /= 0) THEN
         DO i = 0, m
            IF (lindep(i)) WRITE(*, '(a, i3)') ' Singularity detected for power: ', i
         END DO
      END IF

! Calculate progressive residual sums of squares
      CALL ss()
      var = rss(m+1) / (n - m - 1)

! Calculate least-squares regn. coeffs.
      CALL regcf(beta, m+1, ier)

! Calculate covariance matrix, and hence std. errors of coeffs.
      CALL cov(m+1, var, covmat, 231, sterr, ier)
      poly(1)=beta(0)
      poly(2)=beta(1)
      poly(3)=beta(2)

!!$      WRITE(*, *) 'Least-squares coefficients & std. errors'
!!$      WRITE(*, *) 'Power  Coefficient          Std.error      Resid.sum of sq.'
!!$      DO i = 0, m
!!$         WRITE(*, '(i4, g20.12, "   ", g14.6, "   ", g14.6)')  &
!!$              i, beta(i), sterr(i), rss(i+1)
!!$      END DO
!!$      
!!$      WRITE(*, *)
!!$      WRITE(*, '(a, g20.12)') ' Residual standard deviation = ', SQRT(var)
!!$      totalSS = rss(1)
!!$      WRITE(*, '(a, g20.12)') ' R^2 = ', (totalSS - rss(m+1))/totalSS
      return

      end subroutine quadfit

