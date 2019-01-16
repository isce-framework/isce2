      Double Precision function BesselDiffs(num,mcent,data,p,tol)

c*    Interpolates a curve by the Bessel interpolation formula. (Acton, 1970)
c*    This version uses throwback at fourth order. (see Abramowitz and Stegun)
c*    given data(x), interpolate to  data(val).
c*    
c*    num   is the number of array values passed.
c*    mcent is the array index of the location just less than val
c*    data  is the array of ordinate centers
c*    tol   is a dummy variable in this case
c*    p     is the fractional distance from x(mcent) to desired x
c*    i.e. p = (x-x0)/(x1-x0)
      
      
      implicit real*8 (a-h,o-z)
      
      save
      real*8 data(*),b(0:20),difarr(20)
      Integer Order
      logical*1 skip
      data prevp/-1.d0/         ! impossible value
      
      
      skip = .false.
      if(p .eq. prevp) skip = .true.
      prevp = p
      if(mcent .lt. 2) stop 'BesselDiffs: Not enough data '
      if(num-mcent .lt. 3) stop 'BesselDiffs: Not enough data '
      if(num .gt. 21) stop 'BesselDiffs: Arrays are too small '
      
      do  i = 1 , num-1
         difarr(i) = data(i+1) - data(i) ! first differences
      end do
      difs = difarr(mcent)
      
        
      b(0) = .5d0
      b(1) = p - .5d0

      BesselDiffs = Data(mcent) + p * difs
      do order = 2 , 3
         n = order/2
         if(.not. skip) then    ! don't recompute unless necessary
            fl_n=float(n)
            fl_o=float(order)
            fl_o1=float(order-1)
            a = (p+fl_n-1.d0) * (p-fl_n) / fl_o
            b(order) = b(order-2) * a / fl_o1
         end if
         do k = 1 , num - order ! next order differences
            difarr(k) = difarr(k+1)-difarr(k)
         end do
         if(mod(order,2) .eq. 0) then
            difs = difarr(mcent-n)+difarr(mcent-n+1)
         else
            difs = difarr(mcent-n)
         end if
         BesselDiffs = BesselDiffs + b(order) * difs
      end do
      n = 2
      do k = 1 , num - 4        ! fourth order differences
         difarr(k) = difarr(k+1)-difarr(k)
      end do
      difs = difarr(mcent-n)+difarr(mcent-n+1)
      BesselDiffs = BesselDiffs - .184d0*B(2)*difs ! throw back fourth difference
      return
      end

