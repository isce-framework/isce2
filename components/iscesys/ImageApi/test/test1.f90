!c***************************************************************

      subroutine test1(accessor1,accessor2,width1,width2,test)


      implicit none

!c    PARAMETER STATEMENTS:

      integer*8 accessor1,accessor2
      integer width1,width2,i,j,k,test,eofFlag

      complex*8, allocatable :: data1(:)
      real*4, allocatable :: data2(:,:)
      allocate(data1(width1))
      allocate(data2(2,width2))
      eofFlag = 0
      if(test .eq. 1) then
        
        do
            call getLineSequential(accessor1,data1,eofFlag)
            if(eofFlag .lt. 0)then 
                write(6,*) 'eof'
                exit
            endif
            do  i = 1,width1
                data2(1,i) = real(data1(i))
                data2(2,i) = aimag(data1(i))
            enddo
            call setLineSequential(accessor2,data2)
        enddo
      endif
      if(test .eq. 2) then
        
        do
            call getLineSequential(accessor2,data2,eofFlag)
            if(eofFlag .lt. 0) exit
            do  i = 1,width2
                data1(i) = cmplx(data2(1,i),data2(2,i))
            enddo
            call setLineSequential(accessor1,data1)
        enddo
              
      endif




      deallocate(data1)
      deallocate(data2)
      end
