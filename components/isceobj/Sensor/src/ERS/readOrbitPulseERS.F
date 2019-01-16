
        subroutine readOrbitPulseERS(rawAccessor)
          use readOrbitPulseERSState
          integer*8 rawAccessor
          integer ::  k,x(lines),i,tmpICU,ierr
          real*8 :: ICU0,t0
          character, allocatable :: line(:)
          logical :: swap,ltlend
          character :: char4(4)
          integer*4 :: int4,eof
          integer*8 :: int8,ICUold
          integer*8,allocatable :: ICU(:)
          equivalence (int4,char4)

          allocate(line(width))
          allocate(ICU(width))
          ! swap bytes for little endian machine ?
          swap = ltlend()
          
          ! read the ICU counter for each line
          i = 0
          ICUold = 0

          !open(62,file='ICU.out')
          eof = 1
          do k = 1,lines
             !read(21,rec=k) line
             call getLineSequential(rawAccessor,line,eof)
             char4 = line(ICUoffset:(ICUoffset+3))
             if(swap) call byteswapi4(int4) 

             ! convert signed int to unsigned
             int8 = int4
             if(int8.lt.0) then
                int8 = int8 + 4294967295
                int8 = int8 + 1
             endif

             if(int8.eq.(ICUold+1)) then
                ! build vectors of (line#-2,ICU count)
                i = i + 1
                x(i) = k - 2
                ICU(i) = int8
                !write(62,*) x(i),ICU(i)
             endif
             ICUold = int8   
          enddo
          
          !close(62)

          ! least squares estimate of first line ICU counter, 
          !  constrain slope to be PRI or PRI/dICU in ICU units
          !  ICU0 actually corresponds to  line 0

          ! Hardcoded along-track bias of 1.25ms added per
          !"Geometric Calibration of ERS Satellite SAR Images"
          !   Johan Mohr and Soren Madsen April 2001
          ICU0 =  sum(dble(ICU(1:i))-dble(x(1:i))/prf/dICU)/dble(i)
          startingTime = (ICU0-ICUb)*dICU + sat_utc + .00125d0

          deallocate(line)
          deallocate(ICU)
        end subroutine

       logical function ltlend()
!c     checks if this is a little endian machine
!c     returns litend=.true. if it is, litend=.false. if not

       integer*1 j(2)
       integer*2 i
       equivalence (i,j)
  
       i = 1
       if (j(1).eq.1) then
            ltlend = .true.
       else
            ltlend = .false.
       end if
  
      end function ltlend

        subroutine byteswapi4(k)

        !c     does a byteswap on integer4 number

          integer*1 ii(4), jj(4)
          integer*4 i, j, k
          equivalence (i,ii)
          equivalence (j,jj)
          
          i = k
          
          jj(1) = ii(4)
          jj(2) = ii(3)
          jj(3) = ii(2)
          jj(4) = ii(1)
          
          k = j
          
        end subroutine byteswapi4

