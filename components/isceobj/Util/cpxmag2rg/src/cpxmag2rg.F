!c** cpxmag2rg - convert two cpx magnitude files to a single rg file
    
      subroutine cpxmag2rg(imageIn1,imageIn2,imageOut)
      use cpxmag2rgState 
      implicit none 
      !integer*8 n
      integer*8 imageIn1,imageIn2,imageOut
      complex, allocatable ::  a1(:),a2(:),outLine(:)
      integer i,k,igrn,lineRead

      allocate(outLine(len))
      allocate(a1(len))
      allocate(a2(len))
      !jng doubt the file could be so big
      !ierr = stat(f1,statb)
      !n=statb(8)
      !if(n.le.0)n=n+2**31+2**31
      !lines=n/len/8

      !write(*,*)'File length, lines: ',lines


      do i=1,lines
         !use this if the lines are accessed sequentially
         call getLineSequential(imageIn1,a1,lineRead)
         !read(21,rec=i)(a1(k),k=1,len)
         igrn=i+idnoff
         if(igrn.lt.1)igrn=1
         if(igrn.gt.lines)igrn=lines
         !use this to access a specific line
         call getLine(imageIn2,a2,igrn)
         !read(22,rec=igrn)(a2(k),k=1,len)
         do k=1,len
            outLine(k) = (0,0);
            !if(k+iacoff.gt.0.and.k+iacoff.le.len)p(k)=cabs(a2(k+iacoff))
            if(k+iacoff.gt.0.and.k+iacoff.le.len) outLine(k) = cmplx(cabs(a1(k)),cabs(a2(k+iacoff)))
        
         end do
         call setLineSequential(imageOut,outLine,lineRead)
      end do
      
      deallocate(outLine)
      deallocate(a1)
      deallocate(a2)

      end
