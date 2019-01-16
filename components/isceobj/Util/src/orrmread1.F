      subroutine orrmread1(inunit,dtime,dos,dvos, reset)
c  
c  subroutine to read orrm orbit data files; returns vectors which are bessel
c  interpolations (at requested time) of position and velocity vectors
c  stored in the orrm file.  all arguments double precision.
c  all vectors are body fixed.
c  
c  input is dtime  --  utc in seconds since start of day
c           file name -- 
c  
c  output vectors (all at dtime) --
c  
c  dos         s/c position centered on body (km)
c  dvos        s/c velocity (km/s)
c  
c  original version of  14 oct 81  by  howard zebker
c  generalized   12 feb 82  by  ras
c  made efficient for the vax may 13, 1983 by par
c  made accurate enough to use august 17 1984 by par
c  modified for use on mv10000 nov 1985 by dlg
c  modified for new ras crs files jan 1986 by dlg
c  modified to be useful for all ras crsfiles regardless of size and with
c  3 or 4 bodies in them by par jun 1987
c converted for ERS work by par mar 1993

      implicit real*8 (d)
      save
      parameter    (icrssz = 15000)   !1440 org sh change 9/14/96
      dimension    dos(3),dvos(3)
      dimension    data1(6,icrssz),data3(6)
      dimension    dt(icrssz)
      character*80 cbuf
      integer reset, size,j,i
      real*8       work(31),besseldiffs
      Logical    firsttime,found
      
      external besseldiffs

      Data         firsttime/.true./,found/.false./
      
c     
c  explanation of parameters
c     dincr       is the spacing of entries (in secs)
c     initializations
c     
      if (firsttime .or. reset .eq. 1) then
         loc = 1
         i = 1
         read(inunit,'(a)') cbuf
         read(inunit,'(a)') cbuf
         do while(i .le. icrssz) 
            read(inunit,*,end=999) i1, dt(i),i3,(data1(k,i),k=1,6)
            dt(i) = dt(i) * 1.d-3
            i = i + 1
         end do
 999     size = i - 1
         Write(*,*) 'done reading crs file'
         write(*,*) 'read in ',size,' elements.'
         dincr = dt(2) - dt(1)
         write(*,*) 'sample spacing is ',dincr
         firsttime = .false.
      end if
      if(reset .eq. 1) return
      
      if(dtime .gt. dt(size) .or. dtime .lt. dt(1)) then
         write(*,*)  'time and limits ', dtime, dt(1), dt(size)
         write(*,*)  'orrmread: time out of bounds'
c         stop 'orrmread: may need to enlarge vector arrays'
         if(dtime .ge. dtsize)then
            j = size - 1
         else
            j = 5
         endif
         do  i = 1 , 3
            dos(i)  = data1(i,j)
            dvos(i) = data1(i+3,j)
         end do
         return
      end if
      
c     search for time bracket.
      
      loc = (dtime-dt(1))/dincr   + 1
      found = .false.
      
      do while(.not. found) 
         if(dtime .ge. dt(loc) .and. dtime .lt. dt(loc+1)) then
            found = .true.
         elseif(dtime .gt. dt(loc)) then
            loc = loc + 1
         elseif(dtime .lt. dt(loc)) then
            loc = loc - 1
         end if
      end do
c     
c  if correct time bracket found, interpolate.
c  


      lim = 3
      if(loc .lt. 3) lim = loc
      if(size-loc .lt. 3) lim = size-loc
      delta=(dtime-dt(loc))/dincr
      do i = 1 , 6
         do j = -lim , lim
            work(j+lim+1)= data1(i,loc+j)
         enddo
         data3(i) = besseldiffs(2*lim+1,lim+1,work,delta,1.d-15)
      enddo
c     
c  transfer to argument arrays (compiler will not allow equivalence)
c  

      do  i = 1 , 3
         dos(i)  = data3(i)
         dvos(i) = data3(i+3)
      end do
      
      return
      end

