MODULE lsq

!  Module for unconstrained linear least-squares calculations.
!  The algorithm is suitable for updating LS calculations as more
!  data are added.   This is sometimes called recursive estimation.
!  Only one dependent variable is allowed.
!  Based upon Applied Statistics algorithm AS 274.
!  Translation from Fortran 77 to Fortran 90 by Alan Miller.
!  A function, VARPRD, has been added for calculating the variances
!  of predicted values, and this uses a subroutine BKSUB2.

!  Version 1.14, 19 August 2002 - ELF90 compatible version
!  Author: Alan Miller
!  e-mail : amiller @ bigpond.net.au
!  WWW-pages: http://www.ozemail.com.au/~milleraj
!             http://users.bigpond.net.au/amiller/

!  Bug fixes:
!  1. In REGCF a call to TOLSET has been added in case the user had
!     not set tolerances.
!  2. In SING, each time a singularity is detected, unless it is in the
!     variables in the last position, INCLUD is called.   INCLUD assumes
!     that a new observation is being added and increments the number of
!     cases, NOBS.   The line:  nobs = nobs - 1 has been added.
!  3. row_ptr was left out of the DEALLOCATE statement in routine startup
!     in version 1.07.
!  4. In COV, now calls SS if rss_set = .FALSE.  29 August 1997
!  5. In TOLSET, correction to accomodate negative values of D.  19 August 2002

!  Other changes:
!  1. Array row_ptr added 18 July 1997.   This points to the first element
!     stored in each row thus saving a small amount of time needed to
!     calculate its position.
!  2. Optional parameter, EPS, added to routine TOLSET, so that the user
!     can specify the accuracy of the input data.
!  3. Cosmetic change of lsq_kind to dp (`Double precision')
!  4. Change to routine SING to use row_ptr rather than calculate the position
!     of first elements in each row.

!  The PUBLIC variables are:
!  dp       = a KIND parameter for the floating-point quantities calculated
!             in this module.   See the more detailed explanation below.
!             This KIND parameter should be used for all floating-point
!             arguments passed to routines in this module.

!  nobs    = the number of observations processed to date.
!  ncol    = the total number of variables, including one for the constant,
!            if a constant is being fitted.
!  r_dim   = the dimension of array r = ncol*(ncol-1)/2
!  vorder  = an integer vector storing the current order of the variables
!            in the QR-factorization.   The initial order is 0, 1, 2, ...
!            if a constant is being fitted, or 1, 2, ... otherwise.
!  initialized = a logical variable which indicates whether space has
!                been allocated for various arrays.
!  tol_set = a logical variable which is set when subroutine TOLSET has
!            been called to calculate tolerances for use in testing for
!            singularities.
!  rss_set = a logical variable indicating whether residual sums of squares
!            are available and usable.
!  d()     = array of row multipliers for the Cholesky factorization.
!            The factorization is X = Q.sqrt(D).R where Q is an ortho-
!            normal matrix which is NOT stored, D is a diagonal matrix
!            whose diagonal elements are stored in array d, and R is an
!            upper-triangular matrix with 1's as its diagonal elements.
!  rhs()   = vector of RHS projections (after scaling by sqrt(D)).
!            Thus Q'y = sqrt(D).rhs
!  r()     = the upper-triangular matrix R.   The upper triangle only,
!            excluding the implicit 1's on the diagonal, are stored by
!            rows.
!  tol()   = array of tolerances used in testing for singularities.
!  rss()   = array of residual sums of squares.   rss(i) is the residual
!            sum of squares with the first i variables in the model.
!            By changing the order of variables, the residual sums of
!            squares can be found for all possible subsets of the variables.
!            The residual sum of squares with NO variables in the model,
!            that is the total sum of squares of the y-values, can be
!            calculated as rss(1) + d(1)*rhs(1)^2.   If the first variable
!            is a constant, then rss(1) is the sum of squares of
!            (y - ybar) where ybar is the average value of y.
!  sserr   = residual sum of squares with all of the variables included.
!  row_ptr() = array of indices of first elements in each row of R.
!
!--------------------------------------------------------------------------

!     General declarations

IMPLICIT NONE

INTEGER, SAVE                :: nobs, ncol, r_dim
INTEGER, ALLOCATABLE, SAVE   :: vorder(:), row_ptr(:)
LOGICAL, SAVE                :: initialized = .false.,                  &
                                tol_set = .false., rss_set = .false.

! Note. dp is being set to give at least 12 decimal digit
!       representation of floating point numbers.   This should be adequate
!       for most problems except the fitting of polynomials.   dp is
!       being set so that the same code can be run on PCs and Unix systems,
!       which will usually represent floating-point numbers in `double
!       precision', and other systems with larger word lengths which will
!       give similar accuracy in `single precision'.

INTEGER, PARAMETER           :: dp = SELECTED_REAL_KIND(12,60)
REAL (dp), ALLOCATABLE, SAVE :: d(:), rhs(:), r(:), tol(:), rss(:)
REAL (dp), SAVE              :: zero = 0.0_dp, one = 1.0_dp, vsmall
REAL (dp), SAVE              :: sserr, toly

PUBLIC                       :: dp, nobs, ncol, r_dim, vorder, row_ptr, &
                                initialized, tol_set, rss_set,          &
                                d, rhs, r, tol, rss, sserr
PRIVATE                      :: zero, one, vsmall


CONTAINS

SUBROUTINE startup(nvar, fit_const)

!     Allocates dimensions for arrays and initializes to zero
!     The calling program must set nvar = the number of variables, and
!     fit_const = .true. if a constant is to be included in the model,
!     otherwise fit_const = .false.
!
!--------------------------------------------------------------------------

IMPLICIT NONE
INTEGER, INTENT(IN)  :: nvar
LOGICAL, INTENT(IN)  :: fit_const

!     Local variable
INTEGER   :: i

vsmall = 10. * TINY(zero)

nobs = 0
IF (fit_const) THEN
  ncol = nvar + 1
ELSE
  ncol = nvar
END IF

IF (initialized) DEALLOCATE(d, rhs, r, tol, rss, vorder, row_ptr)
r_dim = ncol * (ncol - 1)/2
ALLOCATE( d(ncol), rhs(ncol), r(r_dim), tol(ncol), rss(ncol), vorder(ncol),  &
          row_ptr(ncol) )

d = zero
rhs = zero
r = zero
sserr = zero

IF (fit_const) THEN
  DO i = 1, ncol
    vorder(i) = i-1
  END DO
ELSE
  DO i = 1, ncol
    vorder(i) = i
  END DO
END IF ! (fit_const)

! row_ptr(i) is the position of element R(i,i+1) in array r().

row_ptr(1) = 1
DO i = 2, ncol-1
  row_ptr(i) = row_ptr(i-1) + ncol - i + 1
END DO
row_ptr(ncol) = 0

initialized = .true.
tol_set = .false.
rss_set = .false.

RETURN
END SUBROUTINE startup




SUBROUTINE includ(weight, xrow, yelem)

!     ALGORITHM AS75.1  APPL. STATIST. (1974) VOL.23, NO. 3

!     Calling this routine updates D, R, RHS and SSERR by the
!     inclusion of xrow, yelem with the specified weight.

!     *** WARNING  Array XROW is overwritten.

!     N.B. As this routine will be called many times in most applications,
!          checks have been eliminated.
!
!--------------------------------------------------------------------------


IMPLICIT NONE
REAL (dp),INTENT(IN)                    :: weight, yelem
REAL (dp), DIMENSION(:), INTENT(IN OUT) :: xrow

!     Local variables

INTEGER     :: i, k, nextr
REAL (dp)   :: w, y, xi, di, wxi, dpi, cbar, sbar, xk

nobs = nobs + 1
w = weight
y = yelem
rss_set = .false.
nextr = 1
DO i = 1, ncol

!     Skip unnecessary transformations.   Test on exact zeroes must be
!     used or stability can be destroyed.

  IF (ABS(w) < vsmall) RETURN
  xi = xrow(i)
  IF (ABS(xi) < vsmall) THEN
    nextr = nextr + ncol - i
  ELSE
    di = d(i)
    wxi = w * xi
    dpi = di + wxi*xi
    cbar = di / dpi
    sbar = wxi / dpi
    w = cbar * w
    d(i) = dpi
    DO k = i+1, ncol
      xk = xrow(k)
      xrow(k) = xk - xi * r(nextr)
      r(nextr) = cbar * r(nextr) + sbar * xk
      nextr = nextr + 1
    END DO
    xk = y
    y = xk - xi * rhs(i)
    rhs(i) = cbar * rhs(i) + sbar * xk
  END IF
END DO ! i = 1, ncol

!     Y * SQRT(W) is now equal to the Brown, Durbin & Evans recursive
!     residual.

sserr = sserr + w * y * y

RETURN
END SUBROUTINE includ



SUBROUTINE regcf(beta, nreq, ifault)

!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

!     Modified version of AS75.4 to calculate regression coefficients
!     for the first NREQ variables, given an orthogonal reduction from
!     AS75.1.
!
!--------------------------------------------------------------------------

IMPLICIT NONE
INTEGER, INTENT(IN)                  :: nreq
INTEGER, INTENT(OUT)                 :: ifault
REAL (dp), DIMENSION(:), INTENT(OUT) :: beta

!     Local variables

INTEGER   :: i, j, nextr

!     Some checks.

ifault = 0
IF (nreq < 1 .OR. nreq > ncol) ifault = ifault + 4
IF (ifault /= 0) RETURN

IF (.NOT. tol_set) CALL tolset()

DO i = nreq, 1, -1
  IF (SQRT(d(i)) < tol(i)) THEN
    beta(i) = zero
    d(i) = zero
    ifault = -i
  ELSE
    beta(i) = rhs(i)
    nextr = row_ptr(i)
    DO j = i+1, nreq
      beta(i) = beta(i) - r(nextr) * beta(j)
      nextr = nextr + 1
    END DO ! j = i+1, nreq
  END IF
END DO ! i = nreq, 1, -1

RETURN
END SUBROUTINE regcf



SUBROUTINE tolset(eps)

!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

!     Sets up array TOL for testing for zeroes in an orthogonal
!     reduction formed using AS75.1.

REAL (dp), INTENT(IN), OPTIONAL :: eps

!     Unless the argument eps is set, it is assumed that the input data are
!     recorded to full machine accuracy.   This is often not the case.
!     If, for instance, the data are recorded to `single precision' of about
!     6-7 significant decimal digits, then singularities will not be detected.
!     It is suggested that in this case eps should be set equal to
!     10.0 * EPSILON(1.0)
!     If the data are recorded to say 4 significant decimals, then eps should
!     be set to 1.0E-03
!     The above comments apply to the predictor variables, not to the
!     dependent variable.

!     Correction - 19 August 2002
!     When negative weights are used, it is possible for an alement of D
!     to be negative.

!     Local variables.
!
!--------------------------------------------------------------------------

!     Local variables

INTEGER    :: col, row, pos
REAL (dp)  :: eps1, ten = 10.0, total, work(ncol)

!     EPS is a machine-dependent constant.

IF (PRESENT(eps)) THEN
  eps1 = MAX(ABS(eps), ten * EPSILON(ten))
ELSE
  eps1 = ten * EPSILON(ten)
END IF

!     Set tol(i) = sum of absolute values in column I of R after
!     scaling each element by the square root of its row multiplier,
!     multiplied by EPS1.

work = SQRT(ABS(d))
DO col = 1, ncol
  pos = col - 1
  total = work(col)
  DO row = 1, col-1
    total = total + ABS(r(pos)) * work(row)
    pos = pos + ncol - row - 1
  END DO
  tol(col) = eps1 * total
END DO

tol_set = .TRUE.
RETURN
END SUBROUTINE tolset




SUBROUTINE sing(lindep, ifault)

!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

!     Checks for singularities, reports, and adjusts orthogonal
!     reductions produced by AS75.1.

!     Correction - 19 August 2002
!     When negative weights are used, it is possible for an alement of D
!     to be negative.

!     Auxiliary routines called: INCLUD, TOLSET
!
!--------------------------------------------------------------------------

INTEGER, INTENT(OUT)                :: ifault
LOGICAL, DIMENSION(:), INTENT(OUT)  :: lindep

!     Local variables

REAL (dp)  :: temp, x(ncol), work(ncol), y, weight
INTEGER    :: pos, row, pos2

ifault = 0

work = SQRT(ABS(d))
IF (.NOT. tol_set) CALL tolset()

DO row = 1, ncol
  temp = tol(row)
  pos = row_ptr(row)         ! pos = location of first element in row

!     If diagonal element is near zero, set it to zero, set appropriate
!     element of LINDEP, and use INCLUD to augment the projections in
!     the lower rows of the orthogonalization.

  lindep(row) = .FALSE.
  IF (work(row) <= temp) THEN
    lindep(row) = .TRUE.
    ifault = ifault - 1
    IF (row < ncol) THEN
      pos2 = pos + ncol - row - 1
      x = zero
      x(row+1:ncol) = r(pos:pos2)
      y = rhs(row)
      weight = d(row)
      r(pos:pos2) = zero
      d(row) = zero
      rhs(row) = zero
      CALL includ(weight, x, y)
                             ! INCLUD automatically increases the number
                             ! of cases each time it is called.
      nobs = nobs - 1
    ELSE
      sserr = sserr + d(row) * rhs(row)**2
    END IF ! (row < ncol)
  END IF ! (work(row) <= temp)
END DO ! row = 1, ncol

RETURN
END SUBROUTINE sing



SUBROUTINE ss()

!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

!     Calculates partial residual sums of squares from an orthogonal
!     reduction from AS75.1.
!
!--------------------------------------------------------------------------

!     Local variables

INTEGER    :: i
REAL (dp)  :: total

total = sserr
rss(ncol) = sserr
DO i = ncol, 2, -1
  total = total + d(i) * rhs(i)**2
  rss(i-1) = total
END DO

rss_set = .TRUE.
RETURN
END SUBROUTINE ss



SUBROUTINE cov(nreq, var, covmat, dimcov, sterr, ifault)

!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

!     Calculate covariance matrix for regression coefficients for the
!     first nreq variables, from an orthogonal reduction produced from
!     AS75.1.

!     Auxiliary routine called: INV
!
!--------------------------------------------------------------------------

INTEGER, INTENT(IN)                   :: nreq, dimcov
INTEGER, INTENT(OUT)                  :: ifault
REAL (dp), INTENT(OUT)                :: var
REAL (dp), DIMENSION(:), INTENT(OUT)  :: covmat, sterr

!     Local variables.

INTEGER                :: dim_rinv, pos, row, start, pos2, col, pos1, k
REAL (dp)              :: total
REAL (dp), ALLOCATABLE :: rinv(:)

!     Check that dimension of array covmat is adequate.

IF (dimcov < nreq*(nreq+1)/2) THEN
  ifault = 1
  RETURN
END IF

!     Check for small or zero multipliers on the diagonal.

ifault = 0
DO row = 1, nreq
  IF (ABS(d(row)) < vsmall) ifault = -row
END DO
IF (ifault /= 0) RETURN

!     Calculate estimate of the residual variance.

IF (nobs > nreq) THEN
  IF (.NOT. rss_set) CALL ss()
  var = rss(nreq) / (nobs - nreq)
ELSE
  ifault = 2
  RETURN
END IF

dim_rinv = nreq*(nreq-1)/2
ALLOCATE ( rinv(dim_rinv) )

CALL INV(nreq, rinv)
pos = 1
start = 1
DO row = 1, nreq
  pos2 = start
  DO col = row, nreq
    pos1 = start + col - row
    IF (row == col) THEN
      total = one / d(col)
    ELSE
      total = rinv(pos1-1) / d(col)
    END IF
    DO K = col+1, nreq
      total = total + rinv(pos1) * rinv(pos2) / d(k)
      pos1 = pos1 + 1
      pos2 = pos2 + 1
    END DO ! K = col+1, nreq
    covmat(pos) = total * var
    IF (row == col) sterr(row) = SQRT(covmat(pos))
    pos = pos + 1
  END DO ! col = row, nreq
  start = start + nreq - row
END DO ! row = 1, nreq

DEALLOCATE(rinv)
RETURN
END SUBROUTINE cov



SUBROUTINE inv(nreq, rinv)

!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

!     Invert first nreq rows and columns of Cholesky factorization
!     produced by AS 75.1.
!
!--------------------------------------------------------------------------

INTEGER, INTENT(IN)                  :: nreq
REAL (dp), DIMENSION(:), INTENT(OUT) :: rinv

!     Local variables.

INTEGER    :: pos, row, col, start, k, pos1, pos2
REAL (dp)  :: total

!     Invert R ignoring row multipliers, from the bottom up.

pos = nreq * (nreq-1)/2
DO row = nreq-1, 1, -1
  start = row_ptr(row)
  DO col = nreq, row+1, -1
    pos1 = start
    pos2 = pos
    total = zero
    DO k = row+1, col-1
      pos2 = pos2 + nreq - k
      total = total - r(pos1) * rinv(pos2)
      pos1 = pos1 + 1
    END DO ! k = row+1, col-1
    rinv(pos) = total - r(pos1)
    pos = pos - 1
  END DO ! col = nreq, row+1, -1
END DO ! row = nreq-1, 1, -1

RETURN
END SUBROUTINE inv



SUBROUTINE partial_corr(in, cormat, dimc, ycorr, ifault)

!     Replaces subroutines PCORR and COR of:
!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

!     Calculate partial correlations after the variables in rows
!     1, 2, ..., IN have been forced into the regression.
!     If IN = 1, and the first row of R represents a constant in the
!     model, then the usual simple correlations are returned.

!     If IN = 0, the value returned in array CORMAT for the correlation
!     of variables Xi & Xj is:
!       sum ( Xi.Xj ) / Sqrt ( sum (Xi^2) . sum (Xj^2) )

!     On return, array CORMAT contains the upper triangle of the matrix of
!     partial correlations stored by rows, excluding the 1's on the diagonal.
!     e.g. if IN = 2, the consecutive elements returned are:
!     (3,4) (3,5) ... (3,ncol), (4,5) (4,6) ... (4,ncol), etc.
!     Array YCORR stores the partial correlations with the Y-variable
!     starting with YCORR(IN+1) = partial correlation with the variable in
!     position (IN+1).
!
!--------------------------------------------------------------------------

INTEGER, INTENT(IN)                  :: in, dimc
INTEGER, INTENT(OUT)                 :: ifault
REAL (dp), DIMENSION(:), INTENT(OUT) :: cormat, ycorr

!     Local variables.

INTEGER    :: base_pos, pos, row, col, col1, col2, pos1, pos2
REAL (dp)  :: rms(in+1:ncol), sumxx, sumxy, sumyy, work(in+1:ncol)

!     Some checks.

ifault = 0
IF (in < 0 .OR. in > ncol-1) ifault = ifault + 4
IF (dimc < (ncol-in)*(ncol-in-1)/2) ifault = ifault + 8
IF (ifault /= 0) RETURN

!     Base position for calculating positions of elements in row (IN+1) of R.

base_pos = in*ncol - (in+1)*(in+2)/2

!     Calculate 1/RMS of elements in columns from IN to (ncol-1).

IF (d(in+1) > zero) rms(in+1) = one / SQRT(d(in+1))
DO col = in+2, ncol
  pos = base_pos + col
  sumxx = d(col)
  DO row = in+1, col-1
    sumxx = sumxx + d(row) * r(pos)**2
    pos = pos + ncol - row - 1
  END DO ! row = in+1, col-1
  IF (sumxx > zero) THEN
    rms(col) = one / SQRT(sumxx)
  ELSE
    rms(col) = zero
    ifault = -col
  END IF ! (sumxx > zero)
END DO ! col = in+1, ncol-1

!     Calculate 1/RMS for the Y-variable

sumyy = sserr
DO row = in+1, ncol
  sumyy = sumyy + d(row) * rhs(row)**2
END DO ! row = in+1, ncol
IF (sumyy > zero) sumyy = one / SQRT(sumyy)

!     Calculate sums of cross-products.
!     These are obtained by taking dot products of pairs of columns of R,
!     but with the product for each row multiplied by the row multiplier
!     in array D.

pos = 1
DO col1 = in+1, ncol
  sumxy = zero
  work(col1+1:ncol) = zero
  pos1 = base_pos + col1
  DO row = in+1, col1-1
    pos2 = pos1 + 1
    DO col2 = col1+1, ncol
      work(col2) = work(col2) + d(row) * r(pos1) * r(pos2)
      pos2 = pos2 + 1
    END DO ! col2 = col1+1, ncol
    sumxy = sumxy + d(row) * r(pos1) * rhs(row)
    pos1 = pos1 + ncol - row - 1
  END DO ! row = in+1, col1-1

!     Row COL1 has an implicit 1 as its first element (in column COL1)

  pos2 = pos1 + 1
  DO col2 = col1+1, ncol
    work(col2) = work(col2) + d(col1) * r(pos2)
    pos2 = pos2 + 1
    cormat(pos) = work(col2) * rms(col1) * rms(col2)
    pos = pos + 1
  END DO ! col2 = col1+1, ncol
  sumxy = sumxy + d(col1) * rhs(col1)
  ycorr(col1) = sumxy * rms(col1) * sumyy
END DO ! col1 = in+1, ncol-1

ycorr(1:in) = zero

RETURN
END SUBROUTINE partial_corr




SUBROUTINE vmove(from, to, ifault)

!     ALGORITHM AS274 APPL. STATIST. (1992) VOL.41, NO. 2

!     Move variable from position FROM to position TO in an
!     orthogonal reduction produced by AS75.1.
!
!--------------------------------------------------------------------------

INTEGER, INTENT(IN)    :: from, to
INTEGER, INTENT(OUT)   :: ifault

!     Local variables

REAL (dp)  :: d1, d2, x, d1new, d2new, cbar, sbar, y
INTEGER    :: m, first, last, inc, m1, m2, mp1, col, pos, row

!     Check input parameters

ifault = 0
IF (from < 1 .OR. from > ncol) ifault = ifault + 4
IF (to < 1 .OR. to > ncol) ifault = ifault + 8
IF (ifault /= 0) RETURN

IF (from == to) RETURN

IF (.NOT. rss_set) CALL ss()

IF (from < to) THEN
  first = from
  last = to - 1
  inc = 1
ELSE
  first = from - 1
  last = to
  inc = -1
END IF

DO m = first, last, inc

!     Find addresses of first elements of R in rows M and (M+1).

  m1 = row_ptr(m)
  m2 = row_ptr(m+1)
  mp1 = m + 1
  d1 = d(m)
  d2 = d(mp1)

!     Special cases.

  IF (d1 < vsmall .AND. d2 < vsmall) GO TO 40
  x = r(m1)
  IF (ABS(x) * SQRT(d1) < tol(mp1)) THEN
    x = zero
  END IF
  IF (d1 < vsmall .OR. ABS(x) < vsmall) THEN
    d(m) = d2
    d(mp1) = d1
    r(m1) = zero
    DO col = m+2, ncol
      m1 = m1 + 1
      x = r(m1)
      r(m1) = r(m2)
      r(m2) = x
      m2 = m2 + 1
    END DO ! col = m+2, ncol
    x = rhs(m)
    rhs(m) = rhs(mp1)
    rhs(mp1) = x
    GO TO 40
  ELSE IF (d2 < vsmall) THEN
    d(m) = d1 * x**2
    r(m1) = one / x
    r(m1+1:m1+ncol-m-1) = r(m1+1:m1+ncol-m-1) / x
    rhs(m) = rhs(m) / x
    GO TO 40
  END IF

!     Planar rotation in regular case.

  d1new = d2 + d1*x**2
  cbar = d2 / d1new
  sbar = x * d1 / d1new
  d2new = d1 * cbar
  d(m) = d1new
  d(mp1) = d2new
  r(m1) = sbar
  DO col = m+2, ncol
    m1 = m1 + 1
    y = r(m1)
    r(m1) = cbar*r(m2) + sbar*y
    r(m2) = y - x*r(m2)
    m2 = m2 + 1
  END DO ! col = m+2, ncol
  y = rhs(m)
  rhs(m) = cbar*rhs(mp1) + sbar*y
  rhs(mp1) = y - x*rhs(mp1)

!     Swap columns M and (M+1) down to row (M-1).

  40 pos = m
  DO row = 1, m-1
    x = r(pos)
    r(pos) = r(pos-1)
    r(pos-1) = x
    pos = pos + ncol - row - 1
  END DO ! row = 1, m-1

!     Adjust variable order (VORDER), the tolerances (TOL) and
!     the vector of residual sums of squares (RSS).

  m1 = vorder(m)
  vorder(m) = vorder(mp1)
  vorder(mp1) = m1
  x = tol(m)
  tol(m) = tol(mp1)
  tol(mp1) = x
  rss(m) = rss(mp1) + d(mp1) * rhs(mp1)**2
END DO

RETURN
END SUBROUTINE vmove



SUBROUTINE reordr(list, n, pos1, ifault)

!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

!     Re-order the variables in an orthogonal reduction produced by
!     AS75.1 so that the N variables in LIST start at position POS1,
!     though will not necessarily be in the same order as in LIST.
!     Any variables in VORDER before position POS1 are not moved.

!     Auxiliary routine called: VMOVE
!
!--------------------------------------------------------------------------

INTEGER, INTENT(IN)               :: n, pos1
INTEGER, DIMENSION(:), INTENT(IN) :: list
INTEGER, INTENT(OUT)              :: ifault

!     Local variables.

INTEGER    :: next, i, l, j

!     Check N.

ifault = 0
IF (n < 1 .OR. n > ncol+1-pos1) ifault = ifault + 4
IF (ifault /= 0) RETURN

!     Work through VORDER finding variables which are in LIST.

next = pos1
i = pos1
10 l = vorder(i)
DO j = 1, n
  IF (l == list(j)) GO TO 40
END DO
30 i = i + 1
IF (i <= ncol) GO TO 10

!     If this point is reached, one or more variables in LIST has not
!     been found.

ifault = 8
RETURN

!     Variable L is in LIST; move it up to position NEXT if it is not
!     already there.

40 IF (i > next) CALL vmove(i, next, ifault)
next = next + 1
IF (next < n+pos1) GO TO 30

RETURN
END SUBROUTINE reordr



SUBROUTINE hdiag(xrow, nreq, hii, ifault)

!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2
!
!                         -1           -1
! The hat matrix H = x(X'X) x' = x(R'DR) x' = z'Dz

!              -1
! where z = x'R

! Here we only calculate the diagonal element hii corresponding to one
! row (xrow).   The variance of the i-th least-squares residual is (1 - hii).
!--------------------------------------------------------------------------

INTEGER, INTENT(IN)                  :: nreq
INTEGER, INTENT(OUT)                 :: ifault
REAL (dp), DIMENSION(:), INTENT(IN)  :: xrow
REAL (dp), INTENT(OUT)               :: hii

!     Local variables

INTEGER    :: col, row, pos
REAL (dp)  :: total, wk(ncol)

!     Some checks

ifault = 0
IF (nreq > ncol) ifault = ifault + 4
IF (ifault /= 0) RETURN

!     The elements of xrow.inv(R).sqrt(D) are calculated and stored in WK.

hii = zero
DO col = 1, nreq
  IF (SQRT(d(col)) <= tol(col)) THEN
    wk(col) = zero
  ELSE
    pos = col - 1
    total = xrow(col)
    DO row = 1, col-1
      total = total - wk(row)*r(pos)
      pos = pos + ncol - row - 1
    END DO ! row = 1, col-1
    wk(col) = total
    hii = hii + total**2 / d(col)
  END IF
END DO ! col = 1, nreq

RETURN
END SUBROUTINE hdiag



FUNCTION varprd(x, nreq) RESULT(fn_val)

!     Calculate the variance of x'b where b consists of the first nreq
!     least-squares regression coefficients.
!
!--------------------------------------------------------------------------

INTEGER, INTENT(IN)                  :: nreq
REAL (dp), DIMENSION(:), INTENT(IN)  :: x
REAL (dp)                            :: fn_val

!     Local variables

INTEGER    :: ifault, row
REAL (dp)  :: var, wk(nreq)

!     Check input parameter values

fn_val = zero
ifault = 0
IF (nreq < 1 .OR. nreq > ncol) ifault = ifault + 4
IF (nobs <= nreq) ifault = ifault + 8
IF (ifault /= 0) THEN
  WRITE(*, '(1x, a, i4)') 'Error in function VARPRD: ifault =', ifault
  RETURN
END IF

!     Calculate the residual variance estimate.

var = sserr / (nobs - nreq)

!     Variance of x'b = var.x'(inv R)(inv D)(inv R')x
!     First call BKSUB2 to calculate (inv R')x by back-substitution.

CALL BKSUB2(x, wk, nreq)
DO row = 1, nreq
  IF(d(row) > tol(row)) fn_val = fn_val + wk(row)**2 / d(row)
END DO

fn_val = fn_val * var

RETURN
END FUNCTION varprd



SUBROUTINE bksub2(x, b, nreq)

!     Solve x = R'b for b given x, using only the first nreq rows and
!     columns of R, and only the first nreq elements of R.
!
!--------------------------------------------------------------------------

INTEGER, INTENT(IN)                  :: nreq
REAL (dp), DIMENSION(:), INTENT(IN)  :: x
REAL (dp), DIMENSION(:), INTENT(OUT) :: b

!     Local variables

INTEGER    :: pos, row, col
REAL (dp)  :: temp

!     Solve by back-substitution, starting from the top.

DO row = 1, nreq
  pos = row - 1
  temp = x(row)
  DO col = 1, row-1
    temp = temp - r(pos)*b(col)
    pos = pos + ncol - col - 1
  END DO
  b(row) = temp
END DO

RETURN
END SUBROUTINE bksub2


END MODULE lsq
