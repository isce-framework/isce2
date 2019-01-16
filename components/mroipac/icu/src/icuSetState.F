
        subroutine setWidth(width)
            use icuState
            implicit none
            integer width

            infp%i_rsamps = width
        end subroutine

        subroutine setStartSample(start)
            use icuState
            implicit none
            integer start

            infp%i_ssamp = start+1   !c Python to Fortran count
        end subroutine

        subroutine setEndSample(last)
            use icuState
            implicit none
            integer last

            infp%i_esamp = last      !c Python count
        end subroutine

        subroutine setStartingLine(start)
            use icuState
            implicit none
            integer start

            i_strtline = start+1
        end subroutine

        subroutine setLength(len)
            use icuState
            implicit none
            integer len

            i_numlines = len
        end subroutine

        subroutine setAzimuthBufferSize(len)
            use icuState
            implicit none
            integer len

            i_azbuf = len
        end subroutine

        subroutine setOverlap(len)
            use icuState
            implicit none
            integer len

            i_azcomlin = len
        end subroutine

        subroutine setFilteringFlag(flag)
            use icuState
            implicit none
            integer flag
            infp%i_filtopt = flag
        end subroutine

        subroutine setUnwrappingFlag(flag)
            use icuState
            implicit none
            integer flag
            i_unwrap_flag = flag
        end subroutine

        subroutine setFilterType(flag)
            use icuState
            implicit none
            integer flag
            infp%i_filttype = flag
        end subroutine

        subroutine setLPRangeWinSize(len)
            use icuState
            implicit none
            real*4 len
            infp%r_lpwinrng = len
        end subroutine

        subroutine setLPAzimuthWinSize(len)
            use icuState
            implicit none
            real*4 len
            infp%r_lpwinaz = len
        end subroutine

        subroutine setFilterExponent(pow)
            use icuState
            implicit none
            real*4 pow
            infp%r_ps_alpha = pow
        end subroutine

        subroutine setUseAmplitudeFlag(flag)
            use icuState
            implicit none
            integer flag
            infp%i_useamp = flag
        end subroutine


        subroutine setCorrelationType(flag)
            use icuState
            implicit none
            integer flag

            !c Default values
            infp%i_slope = 0
            infp%i_cc_norm = 0
            infp%i_cc_std = 1
            infp%i_cc_sigma = 0
            infp%i_cc_layer = 1
      
            if(flag .eq. 0)then     !c None 
                infp%i_slope = 0
                infp%i_cc_norm = 0
                infp%i_cc_std = 0
                infp%i_cc_sigma = 0
                infp%i_cc_layer = 0

            elseif(flag .eq. 1)then !c Noslope
                infp%i_slope = 0
                infp%i_cc_norm = 0
                infp%i_cc_std = 1
                infp%i_cc_sigma = 0
                infp%i_cc_layer = 1

            elseif(flag  .eq. 2)then !c Slope 
                infp%i_slope = 1
                infp%i_cc_norm = 1
                infp%i_cc_std = 0
                infp%i_cc_sigma = 0
                infp%i_cc_layer = 2

            elseif(flag .eq. 3)then !c Phase Sigma
                infp%i_slope = 1
                infp%i_cc_norm = 0
                infp%i_cc_std = 0
                infp%i_cc_sigma = 1            !use sigma corr for unwrap mask
                infp%i_cc_layer = 3

            elseif(flag .eq. 4) then !All
                infp%i_slope = 1
                infp%i_cc_norm = 1
                infp%i_cc_std = 1
                infp%i_cc_sigma = 1
                infp%i_cc_layer = 3
            endif
        end subroutine

        subroutine setCorrelationBoxSize(len)
            use icuState
            implicit none
            integer len
            infp%i_cc_winsz = len
        end subroutine

        subroutine setPhaseSigmaBoxSize(len)
            use icuState
            implicit none
            integer len
            infp%i_phs_winsz = len
        end subroutine

        subroutine setPhaseVarThreshold(var)
            use icuState
            implicit none
            real*4 var
            unwp%r_phvarmax = var
        end subroutine

        subroutine setInitCorrThreshold(var)
            use icuState
            implicit none
            real*4 var
            unwp%r_ccthr_min = var
        end subroutine

        subroutine setCorrThreshold(var)
            use icuState
            implicit none
            real*4 var
            unwp%r_ccthr_max = var
        end subroutine

        subroutine setCorrThresholdInc(var)
            use icuState
            implicit none
            real*4 var
            unwp%r_ccthr_incr = var
        end subroutine

        subroutine setNeuTypes(pgFlag,iFlag)
            use icuState
            implicit none
            integer pgFlag, iFlag
            unwp%i_neutypes(1) = pgFlag
            unwp%i_neutypes(2) = iFlag
        end subroutine

        subroutine setNeuThreshold(pgThresh,iThresh,cThresh)
            use icuState
            implicit none
            real*4 pgThresh,iThresh,cThresh
            unwp%r_neuthres(1,1) = pgThresh
            unwp%r_neuthres(2,1) = iThresh
            unwp%r_neuthres(2,2) = cThresh
        end subroutine

        subroutine setBootstrapSize(pts,lines)
            use icuState
            implicit none
            integer pts,lines
            unwp%i_minbootpts = pts
            unwp%i_minbootlns = lines
        end subroutine

        subroutine setNumTreeSets(num)
            use icuState
            implicit none
            integer num
            unwp%i_tree_sets = num
        end subroutine

        subroutine setTreeType(num)
            use icuState
            implicit none
            integer num
            unwp%i_tree_type=num
        end subroutine







