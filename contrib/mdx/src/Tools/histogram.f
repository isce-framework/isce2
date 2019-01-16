c
c Copyright 2004, by the California Institute of Technology. 
c ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. 
c Any commercial use must be negotiated with the Office of Technology 
c Transfer at the California Institute of Technology.
c 
c This software may be subject to U.S. export control laws and regulations.  
c By accepting this document, the user agrees to comply with all applicable 
c U.S. export laws and regulations.  User has the responsibility to obtain 
c export licenses, or other export authority as may be required before 
c exporting such information to foreign countries or providing access to 
c foreign persons.
c
c
        program historgram

          implicit none

          real*4 r_data(32768)
          integer i_cdata(1000)
          integer i_rdata(1000)
          integer i_minrc(100000)
          integer i_maxrc(100000)
          integer i_mincr(100000)
          integer i_maxcr(100000)
          integer i_minr
          integer i_maxr
          integer i_minc
          integer i_maxc
          integer i_pts
          integer i_num

          real*4 r_hist(0:256)
          real*4 r_nullv
          real*4 r_dist
          real*4 r_minv
          real*4 r_maxv
          real*4 r_min
          real*4 r_max

          real*8 r_peg(3)


          character*255 a_datafile
          character*255 a_pntsfile
	  character*255 a_value
          character*512 a_cmnd
	  
          integer i
          integer j
          integer k
          integer i_c
          integer i_r
          integer i_h

	  integer i_arg
          integer i_inarg
          integer i_samps
          integer i_lines
          integer i_send
          integer i_sfmt
          integer i_shdr
          integer i_rhdr
          integer i_rtlr
          integer i_chdr
          integer i_ctlr

          integer i_flg

	  
	  byte b_buff(4)
	  integer i_buff
	  equivalence(b_buff,i_buff)
	  integer i_endian

  
          integer iargc
          external iargc

          integer length
          external length
  
          i_sfmt = 4
          
c          write(6,*) 'initializing min/max rc'
          do i=1,100000
            i_minrc(i)=-1
            i_maxrc(i)=-1
            i_mincr(i)=-1
            i_maxcr(i)=-1
          end do

          do i=0,256
            r_hist(i)=0.0
          end do

c
c     Determine endian ness of machine
c
          b_buff(1) = 0
	  b_buff(2) = 0
	  b_buff(3) = 0
	  b_buff(4) = 1
	  if (i_buff .eq. 1) then
	     i_endian = 1
	  else
	     i_endian = -1
	  end if 
	 


c          write(6,*) 'getting command line arguments'
          i_inarg = iargc()
          if (i_inarg .lt. 4) then
            call write_greeting()
            stop 'done'
          else
            call getarg(1,a_pntsfile)
            call getarg(2,a_datafile)
            call getarg(3,a_value)
            read(a_value,*) i_samps
            write(6,*) 'i_setcols = ',i_samps
            call getarg(4,a_value)
            read(a_value,*) i_lines
            write(6,*) 'i_setrows = ',i_lines
            call getarg(5,a_value)
            read(a_value,*) i_send
            write(6,*) 'i_setvend = ',i_send
            call getarg(6,a_value)
            read(a_value,*) i_sfmt
            write(6,*) 'i_setvfmt = ',i_sfmt
            if (i_inarg .ge. 7) then
              call getarg(7,a_value)
              read(a_value,*) i_shdr
              write(6,*) 'i_setshdr = ',i_shdr
            else
              i_shdr = 0
            end if
            if (i_inarg .ge. 8) then
              call getarg(8,a_value)
              read(a_value,*) i_rhdr
              write(6,*) 'i_setrhdr = ',i_rhdr
            else
              i_rhdr = 0
            end if
            if (i_inarg .ge. 9) then
              call getarg(9,a_value)
              read(a_value,*) i_rtlr
              write(6,*) 'i_setrtlr = ',i_rtlr
            else
              i_rtlr = 0
            end if
            if (i_inarg .ge. 10) then
              call getarg(10,a_value)
              read(a_value,*) i_chdr
              write(6,*) 'i_setchdr = ',i_chdr
            else
              i_chdr = 0
            end if
            if (i_inarg .ge. 11) then
              call getarg(11,a_value)
              read(a_value,*) i_ctlr
              write(6,*) 'i_setctlr = ',i_ctlr
            else
              i_ctlr = 0
            end if
            if (i_inarg .ge. 12) then
              call getarg(12,a_value)
              read(a_value,*) r_minv
              write(6,*) 'r_setminv = ',r_minv
            else
              r_maxv=-1.e27
            end if
            if (i_inarg .ge. 13) then
              call getarg(13,a_value)
              read(a_value,*) r_maxv
              write(6,*) 'r_setmaxv = ',r_maxv
            else
              r_maxv=+1.e27
            end if
          end if

          if (i_send*i_endian .lt. 0) stop '*** byte swapping not supported ***'
          if (i_sfmt .ne. 4) stop '*** Sample format not supported ***'
          if (mod(i_shdr,i_samps*4+i_rhdr+i_rtlr) .ne. 0) stop '*** set header not a multiple of record length ***'
          if (i_chdr .ne. 0 .or. i_ctlr .ne. 0) stop '*** Non-zero column headers not supported ***'

          write(6,*) 'record length in bytes = ',i_samps*4+i_rhdr
          open(unit=20,file=a_datafile,status='old',form='unformatted',access='direct',recl=i_samps*4+i_rhdr+i_rtlr)
          open(unit=30,file=a_pntsfile,status='old',form='formatted')
          open(unit=40,file='histogram.dat',status='unknown',form='formatted')

          i_pts=0

c          write(6,*) 'reading in points'          
          i_minr=1e10
          i_maxr=-1e10
          i_minc=1e10
          i_maxc=-1e10
          do i=1,1000
            read(30,*,err=900,end=900) i_cdata(i),i_rdata(i)
            i_pts=i
            i_minr=min(i_minr,i_rdata(i))
            i_maxr=max(i_maxr,i_rdata(i))
            i_minc=min(i_minc,i_cdata(i))
            i_maxc=max(i_maxc,i_cdata(i))
        
          end do
          write(6,*) 'Too many points - only using first 1000'
900       continue

          write(6,*) 'min/max row = ',i_minr,i_maxr
          write(6,*) 'min/max col = ',i_minc,i_maxc

          if (i_maxr-i_minr .ge. 1000000) stop 'Region too big. Must be less than 1000000 rows'

          do i=1,i_pts
c            write(6,*) 'at point: ',i
            j=i+1
            if (j .gt. i_pts) j=1
            r_dist = sqrt(float(i_cdata(j)-i_cdata(i))**2.+float(i_rdata(j)-i_rdata(i))**2.)
c            write(6,*) ' ',i_rdata(i),i_cdata(i),r_dist
            do k=0,r_dist*4
              i_c = i_cdata(i)+(i_cdata(j)-i_cdata(i))*k/(4*r_dist)
              i_r = i_rdata(i)+(i_rdata(j)-i_rdata(i))*k/(4*r_dist)
c              write(6,*) '   ',i_r,i_c
              if (i_minrc(i_r-i_minr+1) .eq. -1) i_minrc(i_r-i_minr+1)=i_c
              if (i_maxrc(i_r-i_minr+1) .eq. -1) i_maxrc(i_r-i_minr+1)=i_c
              i_minrc(i_r-i_minr+1) = min(i_minrc(i_r-i_minr+1),i_c)
              i_maxrc(i_r-i_minr+1) = max(i_maxrc(i_r-i_minr+1),i_c)
              if (i_mincr(i_c-i_minc+1) .eq. -1) i_mincr(i_c-i_minc+1)=i_r
              if (i_maxcr(i_c-i_minc+1) .eq. -1) i_maxcr(i_c-i_minc+1)=i_r
              i_mincr(i_c-i_minc+1) = min(i_mincr(i_c-i_minc+1),i_r)
              i_maxcr(i_c-i_minc+1) = max(i_maxcr(i_c-i_minc+1),i_r)
            end do
          end do

          r_min=+1e27
          r_max=-1e27
          do i_r=i_minr,i_maxr
c            write(6,*) 'Reading line: ',i_r,'   Samps: ',i_minrc(i_r+1),i_maxrc(i_r+1)
            read(20,rec=i_r+1+i_shdr/(i_samps*4+i_rhdr+i_rtlr)) (r_data(i_c),i_c=1,i_rhdr/4+min(i_samps,i_maxrc(i_r-i_minr+1)+1))
            do i_c=i_minrc(i_r-i_minr+1),i_maxrc(i_r-i_minr+1)
              if (r_data(i_c+i_rhdr/4+1) .ge. r_minv .and. r_data(i_c+i_rhdr/4+1) .le. r_maxv) then
                if (r_data(i_c+i_rhdr/4+1) .le. r_min .or. r_data(i_c+i_rhdr/4+1) .ge. r_max) then
c                  call checkinside(i_pts,i_rdata,i_cdata,i_r,i_c,i_flg)
c                  if (i_flg .eq. 1) then
                  if (i_r .ge. i_mincr(i_c-i_minc+1) .and. i_r .le. i_maxcr(i_c-i_minc+1)) then
                    r_min=min(r_min,r_data(i_c+i_rhdr/4+1))
                    r_max=max(r_max,r_data(i_c+i_rhdr/4+1))
                  end if
                end if
              end if
            end do
          end do

          write(6,*) 'min/max data values: ',r_min,r_max

          i_num = 0
          do i_r=i_minr,i_maxr
            read(20,rec=i_r+1+i_shdr/(i_samps*4+i_rhdr)) (r_data(i_c),i_c=1,i_rhdr/4+min(i_samps,i_maxrc(i_r-i_minr+1)+1))
            do i_c=i_minrc(i_r-i_minr+1),i_maxrc(i_r-i_minr+1)
              if (i_num .lt. 999999) then
c                call checkinside(i_pts,i_rdata,i_cdata,i_r,i_c,i_flg)
c                if (i_flg .eq. 1) then ! inside polygon
                if (i_r .ge. i_mincr(i_c-i_minc+1) .and. i_r .le. i_maxcr(i_c-i_minc+1)) then
                  r_hist(nint(256*(r_data(i_c+i_rhdr/4+1)-r_min)/(r_max-r_min))) = r_hist(nint(256*(r_data(i_c+i_rhdr/4+1)-r_min)/(r_max-r_min))) + 1
                  i_num = i_num +1
                end if
              end if
            end do
          end do

          if (i_num .gt. 0) then
            do i_h = 0,256
              r_hist(i_h) = r_hist(i_h)/i_num
              write(40,*) i_h*(r_max-r_min)/256.+r_min,r_hist(i_h)
            end do
c            write(6,*) 'Forming xmgrace command'
            write(a_cmnd,'(5a,(a,i6,a),a)') 'xmgrace -geometry 800x600 -noask -free ',
     &       '-pexec "s0 line type 3" -pexec "s0 dropline on" ',
     &       '-pexec ''yaxis label "Fraction of Total"'' ',
     &       '-pexec ''xaxis label "Value"'' ',
     &       '-pexec ''title "'//a_datafile(1:length(a_datafile))//'"'' ',
     &       '-pexec ''subtitle "Total number of points:',i_num,'" '' ',
     &       ' histogram.dat &'
            write(6,*) a_cmnd
            call system(a_cmnd)
          else
            write(6,*) 'No points in region'
          end if
          write(6,*) 'Histogram v11 Done'
        end


        subroutine checkinside(i_pts,i_rdata,i_cdata,i_r,i_c,i_flg)
          implicit none

          integer i_pts
          integer i_rdata(i_pts)
          integer i_cdata(i_pts)
          integer i_r
          integer i_c
          integer i_flg

          integer i
          integer j
          real*4 r_ang
          real*4 r_angle
          real*4 r_mag1
          real*4 r_mag2
          real*4 r_dot
          real*4 r_cross

            r_ang=0.0
c            write(6,*) 'row/col = ',i_r,i_c
            do i=1,i_pts-1
              j=i+1
              if (j .gt. i_pts) j=j-i_pts
c              write(6,*) 'pts1 = ',(i_rdata(i)-i_r),(i_cdata(i)-i_c)
c              write(6,*) 'pts2 = ',(i_rdata(j)-i_r),(i_cdata(j)-i_c)
              r_mag1 = sqrt((i_rdata(i)-i_r)**2.+(i_cdata(i)-i_c)**2.)
              r_mag2 = sqrt((i_rdata(j)-i_r)**2.+(i_cdata(j)-i_c)**2.) 
              r_dot  = (i_rdata(i)-i_r)*(i_rdata(j)-i_r)+(i_cdata(i)-i_c)*(i_cdata(j)-i_c)
              r_cross= (i_rdata(i)-i_r)*(i_cdata(j)-i_c)-(i_cdata(i)-i_c)*(i_rdata(j)-i_r)
              r_angle= asin(min(max(r_cross/(r_mag1*r_mag2),-0.999999),0.999999))
              if (r_angle .ne. r_angle) write(6,*) 'NAN at ',i,(i_rdata(i)-i_r),(i_cdata(i)-i_c),(i_rdata(j)-i_r),(i_cdata(j)-i_c),i_r,i_c
              if (r_dot .lt. 0.) then
                if (r_angle .gt. 0) then 
                  r_angle = 3.14159265-r_angle
                else
                  r_angle =-3.14159265-r_angle
                end if
              end if                
              r_ang=r_ang+r_angle
c              write(6,*) i,r_angle,r_ang,r_cross,r_mag1,r_mag2,r_cross/(r_mag1*r_mag2)
            end do
            if (abs(r_ang) .lt. 3.1415) then
              i_flg = 0
c              write(6,*) 'OUTSIDE'
            else
              i_flg = 1
c              write(6,*) 'INSIDE',r_ang
            end if
            
          return
        end


c****************************************************************

      integer*4 function length(a_string)

c****************************************************************
c**   
c**   FILE NAME: rdf_reader.f
c**   
c**   DATE WRITTEN: 15-Sept-1997
c**   
c**   PROGRAMMER: Scott Shaffer
c**   
c**   FUNCTIONAL DESCRIPTION: This function returns the position 
c**   of the last none blank character in the string. 
c**   
c**   ROUTINES CALLED:
c**   
c**   NOTES: 
c**   
c**   UPDATE LOG:
c**   
c**   Date Changed        Reason Changed                  CR # and Version #
c**   ------------       ----------------                 -----------------
c**   
c*****************************************************************

      implicit none

c     INPUT VARIABLES:

      character*(*) a_string
      
c     OUTPUT VARIABLES:

c     LOCAL VARIABLES:

      integer i_len

c     COMMON BLOCKS:

c     EQUIVALENCE STATEMENTS:

c     DATA STATEMENTS:

c     FUNCTION_STATEMENTS:

c     PROCESSING STEPS:

c      write(6,*) 'here =',a_string(1:60)
      i_len=len(a_string)
      do while(i_len .gt. 0 .and. (a_string(i_len:i_len) .eq. ' ' .or. 
     &     ichar(a_string(i_len:i_len)) .eq. 0))
         i_len=i_len-1
c         write(6,*) i_len,' ',ichar(a_string(i_len:i_len)),' ',a_string(i_len:i_len)
      enddo

      length=i_len
      return

      end


        subroutine write_greeting()
          implicit none

            write(6,*) 'This is a very unfriendly program.  Figure it out for youself'

          return
        end
