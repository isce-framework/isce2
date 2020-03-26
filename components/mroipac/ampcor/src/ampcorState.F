      module ampcorState
        character*10 a_datatype(2)  !Data Type for Reference Image Real or Complex
                                    !Data Type for Search Image Real or Complex
                                    !WARNING ampcor uses rdflower()
        integer i_samples(2)        !Number of Samples in Reference/Search Images
        integer i_lines(2)          !Number of lines in Reference/Search Images
        integer i_strtline, i_endline, i_skipline
                                    !Start, End and Skip Lines in Reference Image
        integer i_strtsamp,i_endsamp,i_skipsamp
                                    !Start, End and Skip Samples
                                    !in Reference Image   
        integer i_wsxi,i_wsyi       !Reference Window Size Samples/Lines
        integer i_srchx,i_srchy     !Search Pixels Samples/Lines
        integer i_avgx,i_avgy       !Pixel Averaging Samples/Lines
        integer i_covs,i_cw         !Covariance Surface OversampleFactor 
                                    !and Window Size
        integer i_grossx,i_grossy   !Mean Offset Between Reference
                                    !and Search Images Samples/Lines
        real*4 r_snrth,r_covth      !SNR and Covariance Thresholds
        logical l_debug,l_display   !Debug and Display Flags T/F
        real*4 r_scalex, r_scaley   !Scale factors in X and Y
       
!c assumed size arrays corresponding to the different columns of the ampcor output      
        integer, allocatable, dimension(:) :: i_centerxiArr
        integer, allocatable, dimension(:) :: i_centeryiArr
        real*4, allocatable, dimension(:) :: r_shftxoscArr
        real*4, allocatable, dimension(:) :: r_shftyoscArr
        real*4, allocatable, dimension(:) :: r_snrArr   
        real*4, allocatable, dimension(:) :: r_cov1Arr
        real*4, allocatable, dimension(:) :: r_cov2Arr
        real*4, allocatable, dimension(:) :: r_cov3Arr  

!c we only know the max number of rows. at the end of the ampcor we can get it        
        integer numRowTable
        
        integer winsize_filt
        integer i_covs_filt

      end module ampcorState
