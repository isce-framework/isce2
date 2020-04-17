c!@##$%^&012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012
c!       1         2         3         4         5         6         7         8         9        10        11        12        13
c
c Copyright 2001, by the California Institute of Technology.
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
c***************************************************************

        program mdx

c****************************************************************
c**
c**   FILE NAME: mdx.f
c**
c**   PROGRAM NAME: mdx
c**
c**   DATE STARTED: 12/7/2001
c**
c**   PROGRAMMER:   Scott Shaffer
c**   Jet Propulsion Lab
c**
c**   DESCRIPTION:  This program displays images
c**   in a variaty of image formats including byte,
c**   integer*2, integer*4, real*4, and complex.
c**
c**   ROUTINES CALLED:
c**     init_gx
c**     get_wininfo
c**     getevent
c**     setcolor
c**     display_label
c**     display_rmg
c**     getarg
c**     plus others
c**
c**   NOTES:        Tons of Fun
c**
c**
c**   UPDATE LOG:
cc*
cc*   v178 2009-03-13  Fixed error created in v177 that caused an extra squaring of magnitude for c8 images
cc*                    added flag for GFORTRAN compiling to handle declaration of iargc
cc*
cc*   v186 2010-01-11  Fixed initialization on entry windows for unsed fields
cc*
cc*   v187 2010-01-12  Increased PPM conversion array sizes to handle same limit as screen display
cc*
cc*   v190 2010-03-07  SJS - added flag (shift key) to show position in meters when clicking on image
cc*
cc*   v191 2012-02-22  SJS - Minor changes to report buffer overrun info and allow addr/mult to be entered in .mdx file
cc*
cc*   v192 2012-03-13  SJS - Added flag for reading SRTM 30mx30m tiles
cc*
cc*   v193 2012-03-21  SJS - Increased maximum number of channels to 8, added -r4mag data type
cc*
cc*   v194 2013-03-29  SJS - Incorporated initialization that Ron suggested to fix printing with -P, also increased to 10 channels
cc*
c*****************************************************************

          implicit none

          character*10000 a_cmd
          character*1000 a_value

          integer i_arg
          integer i_inarg

          integer rdflen
          external rdflen

#ifdef IO64
          integer*8 mdxfunc_sample
          external mdxfunc_sample

          integer*8 i_eight
          external i_eight
#else
          integer*4 mdxfunc_sample
          external mdxfunc_sample

          integer*4 i_eight
          external i_eight
#endif


          character*18 version_mdx
          external version_mdx

          integer version_gx
          external version_gx

          integer i,j
          byte   b_data(4000000)
          real*4 r_data(1000000)
          equivalence(b_data,r_data)

          a_cmd = '-V'
          i_inarg = command_argument_count()
          if (i_inarg .eq. 0) then
            write(6,*) '  '
            write(6,'(1x,a,a18,a)'     ) '   <<  mdx    Version  ',version_mdx(),        '  >>   '
            write(6,'(1x,a,f5.1,13x,a)') '   <<  graphx Version  ',float(version_gx()),'  >>   '
            write(6,*) '  '
            call write_greeting()
            stop 'done'
          else
            do i_arg =1, i_inarg
              call getarg(i_arg,a_value)
c             write(6,*) i_arg,':',a_value(1:20)
              do i=1,rdflen(a_value)+1
                if (ichar(a_value(i:i)) .eq. 0) a_value(i:i)=' '
              end do
              if (a_cmd .eq. ' ') then
                a_cmd = a_value
              else
                a_cmd = a_cmd(:max(rdflen(a_cmd),1))//' '//a_value
              end if
            end do
          end if
c         write(6,*) a_cmd


          do i=1,500
            do j=1,500
              r_data(i+(j-1)*500) = i+j/500.
            end do
          end do
          i=mdxfunc_sample(2,1,i_eight(0),4*500*500,b_data)
c         write(6,*) 'Calling mdxsub'
          call mdxsub(a_cmd,i_eight(4000000),mdxfunc_sample)
        end


#ifdef IO64
        integer*8 function mdxfunc_sample(i_flag,i_chn,i_start,i_num,b_data)

          implicit none

c
c  Input Variables
c
          integer*4 i_flag          ! Controls weather the function is returning data or the size of the buffer.  Can also do other functions
          integer*4 i_chn           ! Provides subroutine with the channel number 
          integer*8 i_start         ! Start byte of data to be displayed - Is also an output 
          integer*4 i_num           ! Number of bytes to be displayed
          byte b_data(*)            ! Data buffer

c
c  Local Variables
c
          integer*4 i               ! Counter
          integer*8 i_back          ! Returned value - number of bytes read or total bytes in file
          integer*8 i_bmax          ! Max number of bytes in the internal file
          byte b_buff(4000000)      ! Internal buffer of image data

          data i_bmax /0/

          save i_bmax
          save b_buff
#else
        integer*4 function mdxfunc_sample(i_flag,i_chn,i_start,i_num,b_data)

          implicit none

c
c  Input Variables
c
          integer*4 i_flag
          integer*4 i_chn
          integer*4 i_start
          integer*4 i_num
          byte b_data(1)

c
c  Local Variables
c
          integer*4 i               ! Counter
          integer*4 i_back          ! Returned value -
          integer*4 i_bmax
          byte b_buff(4000000)

          data i_bmax /0/

          save i_bmax
          save b_buff

#endif

            if (i_flag .eq. 0) then        !  return image data in byte array
              i_back = 0
              do i=1,i_num
                if (i_start+i .ge. 1 .and. i_start+i .le. i_bmax) then
                  b_data(i)=b_buff(i_start+i)
                  i_back=i_back+1
                end if
              end do
            else if (i_flag .eq. 1) then   ! return number of bytes in image array
              i_back = i_bmax
            else if (i_flag .eq. 2) then   ! load data into image array (not called within mdx)
              i_back=0
              do i=1,i_num
                if (i_start+i .ge. 1 .and. i_start+i .le. 4000000) then
                  b_buff(i_start+i) = b_data(i)
                  i_back = i_back+1
                  if (i_start+i .gt. i_bmax) i_bmax=i_start+i
                end if
              end do
            else if (i_flag .eq. 3) then   ! clears image array buffer (not called inside mdx)
              i_bmax = 0
              i_back = 0
            end if
            mdxfunc_sample = i_back
          return
        end

        character*(*) function version_mdx()

            version_mdx = '194.0 29-Mar-2013'
          return
        end
