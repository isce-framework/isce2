!c File Parameters

       subroutine setImageDatatype1(a_datatype1_val)
          use ampcorState
          implicit none
          character*(*) a_datatype1_val
          a_datatype(1) = a_datatype1_val
       end

       subroutine setLineLength1(i_samples1_val)
          use ampcorState
          implicit none
          integer i_samples1_val
          i_samples(1) = i_samples1_val
       end

       subroutine setImageDatatype2(a_datatype2_val)
          use ampcorState
          implicit none
          character*(*) a_datatype2_val
          a_datatype(2) = a_datatype2_val
       end

       subroutine setLineLength2(i_samples2_val)
          use ampcorState
          implicit none
          integer i_samples2_val
          i_samples(2) = i_samples2_val
       end

       subroutine setImageLength1(ival)
           use ampcorState
           implicit none
           integer ival
           i_lines(1) = ival
       end 

       subroutine setImageLength2(ival)
           use ampcorState
           implicit none
           integer ival
           i_lines(2) = ival
       end

!c Processing parameters

       subroutine setFirstSampleDown(i_strtline_val)
          use ampcorState
          implicit none
          integer i_strtline_val
          i_strtline = i_strtline_val
       end

       subroutine setLastSampleDown(i_endline_val)
          use ampcorState
          implicit none
          integer i_endline_val
          i_endline = i_endline_val
       end

       subroutine setSkipSampleDown(i_skipline_val)
          use ampcorState
          implicit none
          integer i_skipline_val
          i_skipline = i_skipline_val
       end

       subroutine setFirstSampleAcross(i_strtsamp_val)
          use ampcorState
          implicit none
          integer i_strtsamp_val
          i_strtsamp = i_strtsamp_val
       end

       subroutine setLastSampleAcross(i_endsamp_val)
          use ampcorState
          implicit none
          integer i_endsamp_val
          i_endsamp = i_endsamp_val
       end

       subroutine setSkipSampleAcross(i_skipsamp_val)
          use ampcorState
          implicit none
          integer i_skipsamp_val
          i_skipsamp = i_skipsamp_val
       end

       subroutine setWindowSizeWidth(i_wsxi_val)
          use ampcorState
          implicit none
          integer i_wsxi_val
          i_wsxi = i_wsxi_val
       end

       subroutine setWindowSizeHeight(i_wsyi_val)
          use ampcorState
          implicit none
          integer i_wsyi_val
          i_wsyi = i_wsyi_val
       end

       subroutine setSearchWindowSizeWidth(i_srchx_val)
          use ampcorState
          implicit none
          integer i_srchx_val
          i_srchx = i_srchx_val
       end

       subroutine setSearchWindowSizeHeight(i_srchy_val)
          use ampcorState
          implicit none
          integer i_srchy_val
          i_srchy = i_srchy_val
       end

       subroutine setAcrossLooks(i_avgx_val)
          use ampcorState
          implicit none
          integer i_avgx_val
          i_avgx = i_avgx_val
       end

       subroutine setDownLooks(i_avgy_val)
          use ampcorState
          implicit none
          integer i_avgy_val
          i_avgy = i_avgy_val
       end

       subroutine setOverSamplingFactor(i_covs_val)
          use ampcorState
          implicit none
          integer i_covs_val
          i_covs = i_covs_val
       end

       subroutine setZoomWindowSize(i_cw_val)
          use ampcorState
          implicit none
          integer i_cw_val
          i_cw = i_cw_val
       end

       subroutine setAcrossGrossOffset(i_grossx_val)
          use ampcorState
          implicit none
          integer i_grossx_val
          i_grossx = i_grossx_val
       end

       subroutine setDownGrossOffset(i_grossy_val)
          use ampcorState
          implicit none
          integer i_grossy_val
          i_grossy = i_grossy_val
       end

       subroutine setThresholdSNR(r_snrth_val)
          use ampcorState
          implicit none
          real*8 r_snrth_val
          r_snrth = r_snrth_val
       end

       subroutine setThresholdCov(r_covth_val)
          use ampcorState
          implicit none
          real*8 r_covth_val
          r_covth = r_covth_val
       end

       subroutine setDebugFlag(l_debug_val)
          use ampcorState
          implicit none
          logical l_debug_val
          l_debug = l_debug_val
       end

       subroutine setDisplayFlag(l_display_val)
          use ampcorState
          implicit none
          logical l_display_val
          l_display = l_display_val
       end

       subroutine setScaleFactorX(var)
          use ampcorState
          implicit none
          real*4 var
          r_scalex = var
       end

       subroutine setScaleFactorY(var)
          use ampcorState
          implicit none
          real*4 var
          r_scaley = var
       end
      
       subroutine setWinsizeFilt(winsize_filt_val)
          use ampcorState
          implicit none
          integer winsize_filt_val
          winsize_filt = winsize_filt_val
       end

       subroutine setOversamplingFactorFilt(i_covs_filt_val)
          use ampcorState
          implicit none
          integer i_covs_filt_val
          i_covs_filt = i_covs_filt_val
       end
