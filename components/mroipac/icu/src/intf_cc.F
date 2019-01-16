!c****************************************************************

      subroutine intf_cc(c_intb, c_intb_filt, c_ampb, c_pslope,
     $ r_std_cc, r_norm_cc, r_sigma_cc, r_sigma )

!c****************************************************************
!c**     
!c**   FILE NAME: intf_cc.f
!c**     
!c**   DATE WRITTEN: 12-Feb-98
!c**     
!c**   PROGRAMMER: Charles Werner
!c**     
!c**   FUNCTIONAL DESCRIPTION: This routine performs calculates the 
!c**   correlation  and/or phase standard deviation,
!c**     
!c**   ROUTINES CALLED: std_cor, norm_cor, ph_sigma
!c**
!C**   NOTES: none
!c**     
!c**   UPDATE LOG:
!c**
!c**   Date Changed   Reason Changed    
!c**   ------------   ----------------
!c**    12-Feb-98      Created
!C**
!c*****************************************************************
      use icuState
      implicit none


!c     INPUT VARIABLES:

      complex*8 c_intb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!input interferogram
      complex*8 c_intb_filt(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)!filtered interferogram
      complex*8 c_ampb(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!amp. data in (SLC-1,SLC-2) format
      complex*8 c_pslope(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!phase gradient


!c     OUTPUT VARIABLES:

      real*4 r_std_cc(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!standard correlation
      real*4 r_norm_cc(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!normalized correlation
      real*4 r_sigma_cc(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!corr. derived from phase std. dev.
      real*4 r_sigma(0:infp%i_rsamps-1, 0:infp%i_azbufsize-1)	!phase standard deviation (radians)

!c     LOCAL VARIABLES:

      integer*4 i,j				!loop indices

!c     PROCESSING STEPS:

      do i=0, infp%i_azbufsize - 1
        do j=0, infp%i_rsamps - 1
          r_std_cc(j,i) = 0.0
          r_norm_cc(j,i) = 0.0
          r_sigma_cc(j,i) = 0.0
          r_sigma(j,i) = 0.0
        end do
      end do

      if(infp%i_cc_std .eq. 1) then
        call std_cor( c_intb, c_ampb, infp%i_sline, infp%i_eline, 
     $   infp%i_ssamp, infp%i_esamp, infp%i_cc_winsz, r_std_cc)
      end if

      if(infp%i_cc_norm .eq. 1) then
        call norm_cor( c_intb, c_ampb, c_pslope, infp%i_sline, infp%i_eline,
     $  infp%i_ssamp, infp%i_esamp, infp%i_cc_winsz, r_norm_cc)
      end if
 
      if(infp%i_cc_sigma .eq. 1) then
        call ph_sigma( c_intb_filt, c_pslope,infp%i_sline, infp%i_eline, 
     $  infp%i_ssamp, infp%i_esamp, infp%i_phs_winsz, r_sigma, r_sigma_cc)
      end if

      return
      end

