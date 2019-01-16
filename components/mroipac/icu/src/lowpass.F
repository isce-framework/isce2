!c***************************************************************************

      subroutine lowpass(c_in, c_out, sline, eline, ssamp, esamp, r_ranwin, r_azwin)

!c***************************************************************************
!c**     
!c**   FILE NAME: lowpass.f
!c**     
!c**   DATE WRITTEN: 6-Mar-97
!c**     
!c**   PROGRAMMER: Scott Hensley and Charles Werner
!c**     
!c**   FUNCTIONAL DESCRIPTION: This routine is a simple box car filter
!c**   for complex data smoothing. Note fractional window sizes less than 3
!c**   are allowed in which case a triangular weighting is used.
!c**     
!c**   ROUTINES CALLED:
!c**     
!c**   NOTES: 
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed        Reason Changed      CR # and Version #
!c**   ------------       ----------------     -----------------
!c**     
!c*****************************************************************
      use icuState
      implicit none


!c     INPUT VARIABLES:
      integer*4 sline,eline		!starting and ending line with valid data
      integer*4 ssamp,esamp		!starting and ending sample with valid data
      real*4 r_ranwin,r_azwin
      complex*8 c_in(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)

      complex*8, dimension(:,:), allocatable :: ctmp
	
!c     OUTPUT VARIABLES:

      complex*8 c_out(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)

!c     LOCAL VARIABLES:

      real*4 a
      integer*4 ir,ia,mra,maz,i, i_first/1/


!c     PROCESSING STEPS:

      if (i_first .eq. 1) then

         ALLOCATE ( ctmp(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1) )
         i_first = 0
      end if

      if (r_ranwin .ge. 3.0) then

         mra = int(r_ranwin/2)*2 + 1		!forces window size to be odd

         do ia = sline, eline

            do ir = ssamp, ssamp+mra/2-1
               ctmp(ir,ia) = cmplx(0.,0.)
               do i = max(ssamp, ir-mra/2-1),min(esamp,ir+mra/2-1)
                  ctmp(ir,ia) = ctmp(ir,ia) + c_in(i,ia)
               end do
            end do

            do ir = ssamp+mra/2, esamp-mra/2-1
               ctmp(ir,ia) = c_in(ir-mra/2,ia)
            end do
            
            do i = 1,mra-1
               do ir = ssamp+mra/2, esamp-mra/2-1
                  ctmp(ir,ia) = ctmp(ir,ia) + c_in(ir-mra/2-1+i,ia)
               end do
            end do
            
            do ir = esamp-mra/2, esamp
               ctmp(ir,ia) = cmplx(0.,0.)
               do i = max(ssamp, ir-mra/2-1), min(esamp,ir+mra/2-1)
                  ctmp(ir,ia) = ctmp(ir,ia) + c_in(i,ia)
               end do
            end do

         end do
         
      else if (r_ranwin .le. 1.1) then

         do ia = sline, eline
            do ir = ssamp,esamp
               ctmp(ir,ia) = c_in(ir,ia)
            end do
         end do

      else

         a = 0.5*(r_ranwin - 1)
         do ia = sline, eline
            do ir = ssamp+1, esamp-1
               ctmp(ir,ia) = c_in(ir,ia) + a*(c_in(ir-1,ia) + c_in(ir+1,ia))
            end do
            ctmp(ssamp,ia) = c_in(ssamp,ia)
            ctmp(esamp,ia) = c_in(esamp,ia)
         end do

      endif
      
!c	azimuth filtering

      if (r_azwin .ge. 3.0) then
         maz = int(r_azwin/2)*2 + 1

         do ia = sline, eline
            i = max(sline, ia-maz/2)
            do ir = ssamp, esamp
               c_out(ir,ia) = ctmp(ir,i)
            end do
            do i = max(sline, ia-maz/2), min(eline, ia+maz/2)
               do ir = ssamp, esamp
                  c_out(ir,ia) = c_out(ir,ia) + ctmp(ir,i)
               end do
            end do
         end do

      else if (r_azwin .le. 1.1) then

         do ia = sline, eline
            do ir = ssamp, esamp
               c_out(ir,ia) = ctmp(ir,ia)
            end do
         end do

      else

         a = 0.5*(r_azwin - 1)
         do ia = sline+1, eline-1
            do ir = ssamp, esamp
               c_out(ir,ia) = ctmp(ir,ia) + a*(ctmp(ir,ia-1) + ctmp(ir,ia+1))
            end do
         end do

         do ir = ssamp, esamp
            c_out(ir,0) = ctmp(ir,0)
            c_out(ir,eline) = ctmp(ir,eline)
         end do

      endif

      end



