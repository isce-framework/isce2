module uniform_interp

contains
 subroutine crop_indices(low,high, idx)
    ! crops indices to the borders of the image
    integer, intent(in) :: low, high
    real*8, intent(inout) :: idx
    if( idx < low) idx = dble(low)
    if( idx > high) idx = dble(high)
  end subroutine crop_indices


  real*8 function bilinear(x,y,z)
    ! Bilinear interpolation, input values x,y are
    ! expected in unitless decimal index value
    real*8, intent(in) :: x,y
    real*4, intent(in), dimension(:,:) :: z
    real*8 :: x1, x2, y1, y2
    real*8 :: q11, q12, q21, q22


    x1 = floor(x)
    x2 = ceiling(x)
    y1 = ceiling(y)
    y2 = floor(y)

    q11 = z(int(y1),int(x1))
    q12 = z(int(y2),int(x1))
    q21 = z(int(y1),int(x2))
    q22 = z(int(y2),int(x2))

    if(y1.eq.y2.and.x1.eq.x2) then
       bilinear = q11
    elseif(y1.eq.y2) then
       bilinear = (x2 - x)/(x2 - x1)*q11 + (x - x1)/(x2 - x1)*q21
    elseif (x1.eq.x2) then
       bilinear = (y2 - y)/(y2 - y1)*q11 + (y - y1)/(y2 - y1)*q12
    else
       bilinear = q11*(x2 - x)*(y2 - y)/((x2 - x1)*(y2 - y1)) + &
            q21*(x - x1)*(y2 - y)/((x2 - x1)*(y2 - y1)) + &
            q12*(x2 - x)*(y - y1)/((x2 - x1)*(y2 - y1)) + &
            q22*(x - x1)*(y - y1)/((x2 - x1)*(y2 - y1))
    end if
  end function bilinear

 complex function bilinear_cx(x,y,z)
      ! Bilinear interpolation, input values x,y are
      ! expected in unitless decimal index value
      real*8, intent(in) :: x,y
      complex, intent(in), dimension(:,:) :: z
      real*8 :: x1, x2, y1, y2
      complex :: q11, q12, q21, q22


      x1 = floor(x)
      x2 = ceiling(x)
      y1 = ceiling(y)
      y2 = floor(y)

      q11 = z(int(y1),int(x1))
      q12 = z(int(y2),int(x1))
      q21 = z(int(y1),int(x2))
      q22 = z(int(y2),int(x2))

      if(y1.eq.y2.and.x1.eq.x2) then
         bilinear_cx = q11
      elseif(y1.eq.y2) then
         bilinear_cx = (x2 - x)/(x2 - x1)*q11 + (x - x1)/(x2 - x1)*q21
      elseif (x1.eq.x2) then
         bilinear_cx = (y2 - y)/(y2 - y1)*q11 + (y - y1)/(y2 - y1)*q12
      else
         bilinear_cx = q11*(x2 - x)*(y2 - y)/((x2 - x1)*(y2 - y1)) + &
              q21*(x - x1)*(y2 - y)/((x2 - x1)*(y2 - y1)) + &
              q12*(x2 - x)*(y - y1)/((x2 - x1)*(y2 - y1)) + &
              q22*(x - x1)*(y - y1)/((x2 - x1)*(y2 - y1))
      end if
    end function bilinear_cx

     real*4 function bilinear_f(x,y,z)
      ! Bilinear interpolation, input values x,y are
      ! expected in unitless decimal index value
      real*8, intent(in) :: x,y
      real*4, intent(in), dimension(:,:) :: z
      real*8 :: x1, x2, y1, y2
      real*4 :: q11, q12, q21, q22


      x1 = floor(x)
      x2 = ceiling(x)
      y1 = ceiling(y)
      y2 = floor(y)

      q11 = z(int(y1),int(x1))
      q12 = z(int(y2),int(x1))
      q21 = z(int(y1),int(x2))
      q22 = z(int(y2),int(x2))

      if(y1.eq.y2.and.x1.eq.x2) then
         bilinear_f = q11
      elseif(y1.eq.y2) then
         bilinear_f = (x2 - x)/(x2 - x1)*q11 + (x - x1)/(x2 - x1)*q21
      elseif (x1.eq.x2) then
         bilinear_f = (y2 - y)/(y2 - y1)*q11 + (y - y1)/(y2 - y1)*q12
      else
         bilinear_f = q11*(x2 - x)*(y2 - y)/((x2 - x1)*(y2 - y1)) + &
              q21*(x - x1)*(y2 - y)/((x2 - x1)*(y2 - y1)) + &
              q12*(x2 - x)*(y - y1)/((x2 - x1)*(y2 - y1)) + &
              q22*(x - x1)*(y - y1)/((x2 - x1)*(y2 - y1))
      end if
    end function bilinear_f

  real*8 function bicubic(x,y,z)
    ! Bicubic interpolation, input values x,y are
    ! expected in unitless decimal index value
    ! (based on Numerical Recipes Algorithm)
    real*8, intent(in) :: x,y
    real*4, intent(in), dimension(:,:) :: z
    integer :: x1, x2, y1, y2, i, j, k, l
    real*8, dimension(4) :: dzdx,dzdy,dzdxy,zz
    real*8, dimension(4,4) :: c  ! coefftable
    real*8 :: q(16),qq,wt(16,16),cl(16),t,u
    save wt
    DATA wt/1,0,-3,2,4*0,-3,0,9,-6,2,0,-6,4,8*0,3,0,-9,6,-2,0,6,-4,&
      10*0,9,-6,2*0,-6,4,2*0,3,-2,6*0,-9,6,2*0,6,-4,&
      4*0,1,0,-3,2,-2,0,6,-4,1,0,-3,2,8*0,-1,0,3,-2,1,0,-3,2,&
      10*0,-3,2,2*0,3,-2,6*0,3,-2,2*0,-6,4,2*0,3,-2,&
      0,1,-2,1,5*0,-3,6,-3,0,2,-4,2,9*0,3,-6,3,0,-2,4,-2,&
      10*0,-3,3,2*0,2,-2,2*0,-1,1,6*0,3,-3,2*0,-2,2,&
      5*0,1,-2,1,0,-2,4,-2,0,1,-2,1,9*0,-1,2,-1,0,1,-2,1,&
      10*0,1,-1,2*0,-1,1,6*0,-1,1,2*0,2,-2,2*0,-1,1/


    x1 = floor(x)
    x2 = ceiling(x)
!!$    y1 = ceiling(y)
!!$    y2 = floor(y)
    y1 = floor(y)
    y2 = ceiling(y)


    zz(1) = z(y1,x1)
    zz(4) = z(y2,x1)
    zz(2) = z(y1,x2)
    zz(3) = z(y2,x2)

    ! compute first order derivatives
    dzdx(1) = (z(y1,x1+1)-z(y1,x1-1))/2.d0
    dzdx(2) = (z(y1,x2+1)-z(y1,x2-1))/2.d0
    dzdx(3) = (z(y2,x2+1)-z(y2,x2-1))/2.d0
    dzdx(4) = (z(y2,x1+1)-z(y2,x1-1))/2.d0
    dzdy(1) = (z(y1+1,x1)-z(y1-1,x1))/2.d0
    dzdy(2) = (z(y1+1,x2+1)-z(y1-1,x2))/2.d0
    dzdy(3) = (z(y2+1,x2+1)-z(y2-1,x2))/2.d0
    dzdy(4) = (z(y2+1,x1+1)-z(y2-1,x1))/2.d0

    ! compute cross derivatives
    dzdxy(1) = 0.25d0*(z(y1+1,x1+1)-z(y1-1,x1+1)-&
         z(y1+1,x1-1)+z(y1-1,x1-1))
    dzdxy(4) = 0.25d0*(z(y2+1,x1+1)-z(y2-1,x1+1)-&
         z(y2+1,x1-1)+z(y2-1,x1-1))
    dzdxy(2) = 0.25d0*(z(y1+1,x2+1)-z(y1-1,x2+1)-&
         z(y1+1,x2-1)+z(y1-1,x2-1))
    dzdxy(3) = 0.25d0*(z(y2+1,x2+1)-z(y2-1,x2+1)-&
         z(y2+1,x2-1)+z(y2-1,x2-1))

    ! compute polynomial coeff
    ! pack values into temp array qq
    do i = 1,4
       q(i) = zz(i)
       q(i+4) = dzdx(i)
       q(i+8) = dzdy(i)
       q(i+12) = dzdxy(i)
    enddo
    ! matrix multiply by the stored table
    do i = 1,16
       qq = 0.d0
       do k = 1,16
          qq = qq+wt(i,k)*q(k)
       enddo
       cl(i)=qq
    enddo

    ! unpack results into the coeff table
    l = 0
    do i = 1,4
       do j = 1,4
          l = l + 1
          c(i,j) = cl(l)
       enddo
    enddo

    ! normalize desired points from 0to1
    t = (x - x1)
    u = (y - y1)
    bicubic = 0.d0
    do i=4,1,-1
       bicubic = t*bicubic+((c(i,4)*u+c(i,3))*u+c(i,2))*u+c(i,1)
    enddo

  end function bicubic


  complex function bicubic_cx(x,y,z)
    ! Bicubic interpolation, input values x,y are
    ! expected in unitless decimal index value
    ! (based on Numerical Recipes Algorithm)
    real*8, intent(in) :: x,y
    complex, intent(in), dimension(:,:) :: z
    integer :: x1, x2, y1, y2, i, j, k, l
    complex, dimension(4) :: dzdx,dzdy,dzdxy,zz
    complex, dimension(4,4) :: c  ! coefftable
    complex :: q(16),qq,cl(16)
    real*8 :: wt(16,16),t,u
    save wt
    DATA wt/1,0,-3,2,4*0,-3,0,9,-6,2,0,-6,4,8*0,3,0,-9,6,-2,0,6,-4,&
      10*0,9,-6,2*0,-6,4,2*0,3,-2,6*0,-9,6,2*0,6,-4,&
      4*0,1,0,-3,2,-2,0,6,-4,1,0,-3,2,8*0,-1,0,3,-2,1,0,-3,2,&
      10*0,-3,2,2*0,3,-2,6*0,3,-2,2*0,-6,4,2*0,3,-2,&
      0,1,-2,1,5*0,-3,6,-3,0,2,-4,2,9*0,3,-6,3,0,-2,4,-2,&
      10*0,-3,3,2*0,2,-2,2*0,-1,1,6*0,3,-3,2*0,-2,2,&
      5*0,1,-2,1,0,-2,4,-2,0,1,-2,1,9*0,-1,2,-1,0,1,-2,1,&
      10*0,1,-1,2*0,-1,1,6*0,-1,1,2*0,2,-2,2*0,-1,1/


    x1 = floor(x)
    x2 = ceiling(x)
!!$    y1 = ceiling(y)
!!$    y2 = floor(y)
    y1 = floor(y)
    y2 = ceiling(y)


    zz(1) = z(y1,x1)
    zz(4) = z(y2,x1)
    zz(2) = z(y1,x2)
    zz(3) = z(y2,x2)

    ! compute first order derivatives
    dzdx(1) = (z(y1,x1+1)-z(y1,x1-1))/2.d0
    dzdx(2) = (z(y1,x2+1)-z(y1,x2-1))/2.d0
    dzdx(3) = (z(y2,x2+1)-z(y2,x2-1))/2.d0
    dzdx(4) = (z(y2,x1+1)-z(y2,x1-1))/2.d0
    dzdy(1) = (z(y1+1,x1)-z(y1-1,x1))/2.d0
    dzdy(2) = (z(y1+1,x2+1)-z(y1-1,x2))/2.d0
    dzdy(3) = (z(y2+1,x2+1)-z(y2-1,x2))/2.d0
    dzdy(4) = (z(y2+1,x1+1)-z(y2-1,x1))/2.d0

    ! compute cross derivatives
    dzdxy(1) = 0.25d0*(z(y1+1,x1+1)-z(y1-1,x1+1)-&
         z(y1+1,x1-1)+z(y1-1,x1-1))
    dzdxy(4) = 0.25d0*(z(y2+1,x1+1)-z(y2-1,x1+1)-&
         z(y2+1,x1-1)+z(y2-1,x1-1))
    dzdxy(2) = 0.25d0*(z(y1+1,x2+1)-z(y1-1,x2+1)-&
         z(y1+1,x2-1)+z(y1-1,x2-1))
    dzdxy(3) = 0.25d0*(z(y2+1,x2+1)-z(y2-1,x2+1)-&
         z(y2+1,x2-1)+z(y2-1,x2-1))

    ! compute polynomial coeff
    ! pack values into temp array qq
    do i = 1,4
       q(i) = zz(i)
       q(i+4) = dzdx(i)
       q(i+8) = dzdy(i)
       q(i+12) = dzdxy(i)
    enddo
    ! matrix multiply by the stored table
    do i = 1,16
       qq = 0.d0
       do k = 1,16
          qq = qq+wt(i,k)*q(k)
       enddo
       cl(i)=qq
    enddo

    ! unpack results into the coeff table
    l = 0
    do i = 1,4
       do j = 1,4
          l = l + 1
          c(i,j) = cl(l)
       enddo
    enddo

    ! normalize desired points from 0to1
    t = (x - x1)
    u = (y - y1)
    bicubic_cx = 0.d0
    do i=4,1,-1
       bicubic_cx = t*bicubic_cx+((c(i,4)*u+c(i,3))*u+c(i,2))*u+c(i,1)
    enddo

  end function bicubic_cx

!!$c****************************************************************

      subroutine sinc_coef(r_beta,r_relfiltlen,i_decfactor,r_pedestal,&
          i_weight,i_intplength,i_filtercoef,r_filter)

!!$c****************************************************************
!!$c**
!!$c**   FILE NAME: sinc_coef.f
!!$c**
!!$c**   DATE WRITTEN: 10/15/97
!!$c**
!!$c**   PROGRAMMER: Scott Hensley
!!$c**
!!$c**   FUNCTIONAL DESCRIPTION: The number of data values in the array
!!$c**   will always be the interpolation length * the decimation factor,
!!$c**   so this is not returned separately by the function.
!!$c**
!!$c**   ROUTINES CALLED:
!!$c**
!!$c**   NOTES:
!!$c**
!!$c**   UPDATE LOG:
!!$c**
!!$c**   Date Changed        Reason Changed                  CR # and Version #
!!$c**   ------------       ----------------                 -----------------
!!$c**   06/18/21           adjust r_soff to make sure coef at the center is 1.
!!$c**                      note that the routine doesn't work well for odd sequences
!!$c*****************************************************************

      use fortranUtils

      implicit none

!c     INPUT VARIABLES:

      real*8 r_beta             !the "beta" for the filter
      real*8 r_relfiltlen       !relative filter length
      integer i_decfactor       !the decimation factor
      real*8 r_pedestal         !pedestal height
      integer i_weight          !0 = no weight , 1=weight

!c     OUTPUT VARIABLES:

      integer i_intplength      !the interpolation length
      integer i_filtercoef      !number of coefficients
      real*8 r_filter(*)        !an array of data values

!c     LOCAL VARIABLES:

      real*8 r_wgt,r_s,r_fct,r_wgthgt,r_soff,r_wa
      integer i
      real*8 pi,j

!c     COMMON BLOCKS:

!c     EQUIVALENCE STATEMENTS:

!c     DATA STATEMENTS:

!C     FUNCTION STATEMENTS:

!c     PROCESSING STEPS:

!c     number of coefficients

      pi = getPi()

      i_intplength = nint(r_relfiltlen/r_beta)
      i_filtercoef = i_intplength*i_decfactor
      r_wgthgt = (1.d0 - r_pedestal)/2.d0
      r_soff = i_filtercoef/2.d0

      do i=0,i_filtercoef-1
         r_wa = i - r_soff
         r_s = r_wa*r_beta/(1.0d0 * i_decfactor)
         if(r_s .ne. 0.0)then
            r_fct = sin(pi*r_s)/(pi*r_s)
         else
            r_fct = 1.0d0
         endif
         if(i_weight .eq. 1)then
            r_wgt = (1.d0 - r_wgthgt) + r_wgthgt*cos((pi*r_wa)/r_soff)
            r_filter(i+1) = r_fct*r_wgt
         else
            r_filter(i+1) = r_fct
         endif

!!         print *, i, r_wa, r_wgt,j,r_s,r_fct
      enddo

    end subroutine sinc_coef

!cc-------------------------------------------

      complex*8 function sinc_eval(arrin,nsamp,intarr,idec,ilen,intp,frp)

      integer ilen,idec,intp, nsamp
      real*8 frp
      complex arrin(0:nsamp-1)
      real*4 intarr(0:idec*ilen-1)

! note: arrin is a zero based coming in, so intp must be a zero-based index.

      sinc_eval = cmplx(0.,0.)
      if(intp .ge. ilen-1 .and. intp .lt. nsamp ) then
         ifrac= min(max(0,int(frp*idec)),idec-1)
         do k=0,ilen-1
            sinc_eval = sinc_eval + arrin(intp-k)*intarr(k + ifrac*ilen)
         enddo
      end if

      end function sinc_eval

    real*4 function sinc_eval_2d_f(arrin,intarr,idec,ilen,intpx,intpy,frpx,frpy,xlen,ylen)

      integer ilen,idec,intpx,intpy,xlen,ylen,k,m,ifracx,ifracy
      real*8 frpx,frpy
      real*4 arrin(0:xlen-1,0:ylen-1)
      real*4 intarr(0:idec*ilen-1)

! note: arrin is a zero based coming in, so intp must be a zero-based index.

      sinc_eval_2d_f = 0.
      if((intpx.ge.ilen-1.and.intpx.lt.xlen) .and. (intpy.ge.ilen-1.and.intpy.lt.ylen)) then

        ifracx= min(max(0,int(frpx*idec)),idec-1)
        ifracy= min(max(0,int(frpy*idec)),idec-1)

        do k=0,ilen-1
           do m=0,ilen-1
              sinc_eval_2d_f = sinc_eval_2d_f + arrin(intpx-k,intpy-m)*&
                   intarr(k + ifracx*ilen)*intarr(m + ifracy*ilen)
           enddo
        enddo

      end if
    end function sinc_eval_2d_f

    real*4 function sinc_eval_2d_d(arrin,intarr,idec,ilen,intpx,intpy,frpx,frpy,xlen,ylen)

      integer ilen,idec,intpx,intpy,xlen,ylen,k,m,ifracx,ifracy
      real*8 frpx,frpy
      real*8 arrin(0:xlen-1,0:ylen-1)
      real*8 intarr(0:idec*ilen-1)

! note: arrin is a zero based coming in, so intp must be a zero-based index.

      sinc_eval_2d_d = 0.d0
      if((intpx.ge.ilen-1.and.intpx.lt.xlen) .and. (intpy.ge.ilen-1.and.intpy.lt.ylen)) then

        ifracx= min(max(0,int(frpx*idec)),idec-1)
        ifracy= min(max(0,int(frpy*idec)),idec-1)

        do k=0,ilen-1
           do m=0,ilen-1
              sinc_eval_2d_d = sinc_eval_2d_d + arrin(intpx-k,intpy-m)*&
                   intarr(k + ifracx*ilen)*intarr(m + ifracy*ilen)
           enddo
        enddo
      end if
    end function sinc_eval_2d_d

    complex function sinc_eval_2d_cx(arrin,intarr,idec,ilen,intpx,intpy,frpx,frpy,xlen,ylen)

      integer ilen,idec,intpx,intpy,xlen,ylen,k,m,ifracx,ifracy
      real*8 frpx,frpy, fweight, fweightsum
      complex arrin(0:xlen-1,0:ylen-1)
      real*4 intarr(0:idec*ilen-1)

! note: arrin is a zero based coming in, so intp must be a zero-based index.

      sinc_eval_2d_cx = cmplx(0.,0.)
      if((intpx.ge.ilen-1.and.intpx.lt.xlen) .and. (intpy.ge.ilen-1.and.intpy.lt.ylen)) then

        ifracx= min(max(0,int(frpx*idec)),idec-1)
        ifracy= min(max(0,int(frpy*idec)),idec-1)

        ! to normalize the sinc interpolator
        fweightsum = 0.

        do k=0,ilen-1
           do m=0,ilen-1
              fweight = intarr(k + ifracx*ilen)*intarr(m + ifracy*ilen)
              sinc_eval_2d_cx = sinc_eval_2d_cx + arrin(intpx-k,intpy-m) * fweight
              fweightsum = fweightsum + fweight
           enddo
        enddo
        sinc_eval_2d_cx = sinc_eval_2d_cx/fweightsum
      end if

    end function sinc_eval_2d_cx




end module uniform_interp

