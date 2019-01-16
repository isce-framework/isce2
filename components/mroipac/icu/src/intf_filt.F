!c****************************************************************

      subroutine intf_filt(c_intb, c_ampb, c_intb_filt)

!c****************************************************************
!c**     
!c**   FILE NAME: intf_filt.f
!c**     
!c**   DATE WRITTEN: 4-Mar-98
!c**     
!c**   PROGRAMMER: Charles Werner
!c**     
!c**   FUNCTIONAL DESCRIPTION: This routine performs general filtering
!c**   of the interferogram
!c**     
!c**   ROUTINES CALLED: psfilt, lowpass
!c**
!c**   NOTES: i_filttype: 0=lowpass, 1=PS filtering    
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed   Reason Changed    
!c**   ------------   ----------------
!c**    4-Mar-98      Created
!C**
!c*****************************************************************
      
      use icuState
      implicit none

!c     INPUT VARIABLES:

      complex*8 c_intb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!input interferogram
      complex*8 c_ampb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!amplitude data in (SLC-1,SLC-2) pair format

!c     OUTPUT VARIABLES:

      complex*8 c_intb_filt(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)!output filtered interf.

c     LOCAL VARIABLES:

      integer*4 k,l,j,i				!loop indices
       
      do i=0, infp%i_azbufsize - 1		!initialize output data
        do j=0, infp%i_rsamps - 1
          c_intb_filt(j,i)=cmplx(0.0,0.0)
        end do
      end do

      if(infp%i_filtopt .eq. 0) then
!c        write(6,'(1x,a)') 'INTF_FILT: no interferogram filtering applied'
        do l = infp%i_sline, infp%i_eline 
          do k=infp%i_ssamp, infp%i_esamp
            c_intb_filt(k,l) = c_intb(k,l)
          end do
        end do
        return
      end if
     
      if(infp%i_filttype .eq. 0) then 		!lowpass filter

        call lowpass(c_intb, c_intb_filt, infp%i_sline, infp%i_eline, 
     $   infp%i_ssamp, infp%i_esamp, infp%r_lpwinrng, infp%r_lpwinaz)

      elseif (infp%i_filttype .eq. 1) then 		!adaptive power spectrum filter

        if(infp%i_useamp .eq. 1) then
          call psfilt(c_intb, c_ampb, c_intb_filt, infp%i_sline, infp%i_eline, 
     $     infp%i_ssamp, infp%i_esamp, infp%r_ps_alpha)
        else
          call psfilt(c_intb, c_intb, c_intb_filt, infp%i_sline, infp%i_eline, 
     $     infp%i_ssamp, infp%i_esamp, infp%r_ps_alpha)
        end if

      end if

      return
      end
