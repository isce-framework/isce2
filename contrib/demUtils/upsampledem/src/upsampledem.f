!c****************************************************************

!c     Program upsampledem 

!c****************************************************************
!c**     
!c**   FILE NAME: upsampledem.f
!c**     
!c**   DATE WRITTEN: 12/09/2013
!c**     
!c**   PROGRAMMER: Piyush Agram
!c**     
!c**   FUNCTIONAL DESCRIPTION: This program will take a dem and will
!c**   upsample it in both dimensions by integer factor using Akima
!c**   interpolation.
!c**     
!c**   ROUTINES CALLED: Functions from AkimaLib 
!c**     
!c**   NOTES: 
!c**
!c**   1. This program does not fill voids. Voids in input dem must be 
!c**   filled first. Else expect artifacts around voids.
!c**
!c*****************************************************************

      subroutine upsampledem(inAccessor,outAccessor,method) 
      
      use upsampledemState
      use AkimaLib
      use fortranUtils
      implicit none


      !character*120 a_infile,a_outfile,a_string,a_geoidfile
      integer*8 inAccessor,outAccessor
      character*20000 MESSAGE
      integer i,j,ii,jj,ip,xx,yy
      integer method
      real*8 ix,iy
      real*4, allocatable, dimension(:,:) :: r_indata
      real*4, allocatable, dimension(:,:) :: r_outdata
      real*4, allocatable, dimension(:)   :: r_lastline
      real*8, dimension(aki_nsys) :: poly

      integer i_outnumlines,i_outsamples,npatch,i_line
      integer i_completed

      real*4 interp2DSpline
      !!Overlap between patches for Akima-resampling
      integer , parameter :: i_overlap = 8

!c     PROCESSING STEPS:
      write(MESSAGE,*) ' '
      call write_out(stdWriter,MESSAGE)
      write(MESSAGE,'(a)') '    <<Upsample DEM>> '
      call write_out(stdWriter,MESSAGE)

      write(MESSAGE,*) ' '
      call write_out(stdWriter,MESSAGE)

      write(MESSAGE,'(a)') 'Jet Propulsion Laboratory - Radar Science and Engineering '
      call write_out(stdWriter,MESSAGE)

      write(MESSAGE,*) ' '
      call write_out(stdWriter,MESSAGE)

      write(MESSAGE,*) 'Input dimensions: ', i_numlines, i_samples
      call write_out(stdWriter, MESSAGE)
      !! Compute output dimensions
      i_outsamples = (i_samples-1)*i_xfactor + 1
      i_outnumlines = (i_numlines-1)*i_yfactor + 1
      npatch = ceiling(real(i_numlines)/(i_patch-i_overlap))
      write(MESSAGE,*) 'Number of patches: ', npatch
      call write_out(stdWriter,MESSAGE)

      !! Allocate statements
      allocate(r_indata(i_samples, i_patch))
      allocate(r_outdata(i_outsamples, i_yfactor))
      allocate(r_lastline(i_outsamples))

      write(MESSAGE,*) 'Scale Factors : ', i_yfactor, i_xfactor
      call write_out(stdWriter, MESSAGE)
      write(MESSAGE,*) 'Output Dimensions: ', i_outnumlines, i_outsamples
      call write_out(stdWriter, MESSAGE)

      !! Start patch wise processing
      i_completed = 0
      do ip=1,npatch
      
        r_indata = 0.
        r_outdata = 0.

        !!Read a patch of the input DEM
        do i=1,i_patch
            i_line = (ip-1)*(i_patch-i_overlap) + i
            if(i_line.le.i_numlines) then
                call getLine(inAccessor,r_indata(:,i), i_line)
            endif
        enddo

        if (method.eq.AKIMA_METHOD) then
            !!Start interpolating the patch
            do i=1,i_patch-i_overlap/2
                i_line = (ip-1)*(i_patch-i_overlap) + i    !!Get input line number
                if (i_line.gt.i_completed) then    !!Skip lines already completed
                    if (mod(i_line,100).eq.0) then
                        write(MESSAGE,*)'Completed line: ', i_line
                        call write_out(stdWriter, MESSAGE)
                    endif

                    do j=1,i_samples-1
                        !!Create the Akima polynomial
                        call polyfitAkima(i_samples, i_patch, r_indata,j,i,poly)
                        do ii=1,i_yfactor
                            iy = i + (ii-1)/(1.0*i_yfactor)
                            yy = (i_line-1)*i_yfactor + ii
                            do jj=1,i_xfactor
                                ix =j + (jj-1)/(1.0*i_xfactor)
                                xx = (j-1)*i_xfactor + jj

                                !!Evaluate the Akima polynomial
                                r_outdata(xx,ii) = polyvalAkima(j,i,ix,iy,poly)
                            enddo
                        enddo
                    enddo

                    !!Write lines to output
                    do ii=1,i_yfactor
                        yy = (i_line-1)*i_yfactor + ii
                        !!Fill out last data point
                        r_outdata(i_outsamples,ii) = r_outdata(i_outsamples-1,ii)
                        if(yy.lt.i_outnumlines) then
                            call setLineSequential(outAccessor, r_outdata(:,ii))
                            r_lastline = r_outdata(:,ii)
                        endif
                    enddo

                    i_completed = i_completed+1

                    !!Fill out last line if needed
                    if(i_completed.eq.(i_numlines-1)) then
                        call setLineSequential(outAccessor, r_lastline)
                        i_completed = i_completed + 1
                    endif
                endif 
            enddo !!End of patch interpolation

        else if (method.eq.BIQUINTIC_METHOD) then

            !!Start interpolating the path
            do i=1,i_patch-i_overlap/2
                i_line = (ip-1)*(i_patch-i_overlap) + i !!Get input line number
                if (i_line .gt. i_completed) then   !!Skip lines already completed
                    if (mod(i_line,100).eq.0) then
                        write(MESSAGE,*) 'Completed line: ', i_line
                        call write_out(stdWriter, MESSAGE)
                    endif

                    do j=1, i_samples-1
                        
                        do ii=1,i_yfactor
                            iy = i + (ii-1)/(1.0*i_yfactor)
                            yy = (i_line-1)*i_yfactor + ii
                            do jj=1, i_xfactor
                                ix = j + (jj-1)/(1.0*i_xfactor)
                                xx = (j-1)*i_xfactor + jj

                                r_outdata(xx,ii) = interp2DSpline(6,i_patch,i_samples,r_indata,iy,ix)
                            end do
                        end do
                    end do

                    !!Write lines to output
                    do ii=1,i_yfactor
                        yy = (i_line-1)*i_yfactor + ii
                        !! Fill out last data point 
                        r_outdata(i_outsamples,ii) = r_outdata(i_outsamples-1,ii)
                        if (yy.lt.i_outnumlines) then
                            call setLineSequential(outAccessor, r_outdata(:,ii))
                            r_lastline = r_outdata(:,ii)
                        endif
                    enddo

                    i_completed = i_completed + 1

                    !!Fill out last line if needed
                    if (i_completed.eq.(i_numlines-1)) then
                        call setLineSequential(outAccessor, r_lastline)
                        i_completed = i_completed + 1
                    endif
                endif
            enddo
        else
            print *, 'Unknown interpolation method: ', method
            stop
        endif
      enddo
                            
      deallocate(r_indata)
      deallocate(r_outdata)
      deallocate(r_lastline)
      
      end subroutine upsampledem


