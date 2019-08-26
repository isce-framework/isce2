c  get alos position and times
      subroutine readOrbitPulse(ledAccessor,rawAccessor,auxAccessor)
      use readOrbitPulseState
      implicit none 
      integer*8 ledAccessor
      integer*8 rawAccessor
      integer*8 auxAccessor
      integer i,idoy,ilocation,n,iyear,isec,numpoints,ims
      double precision ddate(2)
      character*60 ledfile, datafile, str
      character*1 descriptor(720)
      character*1 summary(4096)
      character*1 position(4680)
      character*4 int4
      double precision val,gete22_15
      double precision x(3),v(3),xold(3),vest(3)
      double precision xx(3,28),vv(3,28),t(28),time
      double precision timefirst,timedelta,timeorbit(28),timeline(100000),timeslope
      double precision avetime,refline
      double precision sumx,sumy,sumsqx,sumsqy,sumxy,ssxx,ssyy,ssxy
      integer*1 indata(32768)
      integer statb(13),stat
      integer numdata,rowPos,colPos,eof
      integer*4 unpackBytes
c  read the leader file descriptor record
      
      !!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!
      ! here the ledAccessos stream is linearized i.e. is created in python such that it has one column and nlines lines. 
      !!!!!!!!!!!!!!
      !!!!!!!!!!!!!!

      !get the first 720 bytes
      numData = 720
      rowPos = 1

      call getStreamAtPos(ledAccessor,descriptor,rowPos,numData)
c  read the leader file summary record
      !!!!!jng ierr=ioread(ichan,summary,4096)

      !get 4096 bytes from the previous stream position
      rowPos = numData + rowPos  
      numData = 4096
      call getStreamAtPos(ledAccessor,summary,rowPos,numData)

c  read the leader file position record
      !!!! jng ierr=ioread(ichan,position,4680)

      !get 4096 bytes from the previous stream position
      rowPos = numData + rowPos  
      numData = 4680
      call getStreamAtPos(ledAccessor,position,rowPos,numData)
c  print out some parameters first

c  time of first data point
      int4=position(145)//position(146)//position(147)//position(148)
      read(int4,*)iyear

      int4=position(157)//position(158)//position(159)//position(160)
      read(int4,*)idoy

      timefirst=gete22_15(position(161))
      timedelta=gete22_15(position(183))

      print *,'First data point ',iyear,idoy,timefirst
      print *,'Orbit record spacing ',timedelta

      int4=position(141)//position(142)//position(143)//position(144)
      print *,int4
      read(int4,*)numpoints
      print *,'Number of orbit points ',numpoints

      do n=1,numpoints
         timeorbit(n)=timefirst+(n-1)*timedelta

         x(1)=gete22_15(position(387+(n-1)*66*2))
         x(2)=gete22_15(position(387+(n-1)*66*2+22))
         x(3)=gete22_15(position(387+(n-1)*66*2+44))

         v(1)=gete22_15(position(387+(n-1)*66*2+66))
         v(2)=gete22_15(position(387+(n-1)*66*2+22+66))
         v(3)=gete22_15(position(387+(n-1)*66*2+44+66))

c         print *,x,v,timeorbit(n)
c         print *,n,timeorbit(n),x!,v

c         if(n.ge.6.and.n.le.9)then
            xx(1,n)=x(1)
            xx(2,n)=x(2)
            xx(3,n)=x(3)
            vv(1,n)=v(1)
            vv(2,n)=v(2)
            vv(3,n)=v(3)
c         end if

      end do

      ! jng ichandata=initdk(22,datafile)

c  read in the raw data file line by line
      ! jng ierr=stat(datafile,statb)
      
      !nlines=statb(8)/len ! now is set from python

      print *,'Lines in data file ',nlines,len
      !call initSequentialAccessor(rawAccessor,1)
      do i=1,nlines
         ! jng ierr=ioread(ichandata,indata,len)
         call getLineSequential(rawAccessor,indata,eof)
         iyear = unpackBytes(indata(40), indata(39), indata(38), indata(37))
         idoy  = unpackBytes(indata(44), indata(43), indata(42), indata(41))
         ims   = unpackBytes(indata(48), indata(47), indata(46), indata(45))
         ddate(2) = ims*1000.0 !we save days in the year and microsec in the day
         ddate(1) = 1.*idoy
         call setLineSequential(auxAccessor,ddate)  
       end do

!      print *, sumx,sumsqx,ssxx,sumy,sumsqy,sumxy,nlines,(ssxy/ssxx)/timeslope
!      print *,timeslope,avetime,1.d0/timeslope


!      open(31,file='position.out')

      end

      double precision function gete22_15(str)

      character*1 str(*)
      character*22 e22_15
      double precision val

      do i=1,22
         e22_15(i:i)=str(i)
      end do
c      print *,e22_15

      read(e22_15,*)val
c      print *,val

      gete22_15=val

      return

      end

      integer*4 function unpackBytes(i1, i2, i3, i4)
      integer*4                      i1, i2, i3, i4
      unpackBytes = iand(i1, 255)*256*256*256 + iand(i2, 255)*256*256 +
     $              iand(i3, 255)*256         + iand(i4, 255)
      end function
