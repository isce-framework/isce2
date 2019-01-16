!c  simamplitude - convert a shaded relief i*2 into a simulated amplitude image
      subroutine simamplitude(topoAccessor,simampAccessor)

      use simamplitudeState
      implicit none
      real, allocatable :: hgt(:) !Should just be real file
      real, allocatable :: shade(:)
      integer*8 topoAccessor,simampAccessor
      integer line,i,j

      allocate(hgt(len))
      allocate(shade(len))

      line = 1
      do i=1,lines
         call getLineSequential(topoAccessor,hgt,line)
!c  shade this line
         do j=1,len-1
            shade(j)=(hgt(j+1)-hgt(j))*scale+100
!c            if(shade(j).lt.0.)shade(j)=0.
!c            if(shade(j).gt.200.)shade(j)=200.
         end do
         shade(len)=0

         call setLineSequential(simampAccessor,shade,line)
      end do
      deallocate(hgt)
      deallocate(shade)

      end
