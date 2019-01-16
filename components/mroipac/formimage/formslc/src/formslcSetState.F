c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
c                                  Giangi Sacco
c                        NASA Jet Propulsion Laboratory
c                      California Institute of Technology
c                        (C) 2009  All Rights Reserved
c
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        subroutine setNumberGoodBytes(varInt)
                use formslcState
                implicit none
                integer varInt
                ngood1 = varInt
        end

        subroutine setNumberBytesPerLine(varInt)
                use formslcState
                implicit none
                integer varInt
                nbytes1 = varInt
        end

        subroutine setDebugFlag(varInt)
                use formslcState
                implicit none
                integer varInt
                iflag = varInt
        end

        subroutine setDeskewFlag(varInt)
                use formslcState
                implicit none
                integer*4 varInt
                ideskew = varInt
        end

        subroutine setSecondaryRangeMigrationFlag(varInt)
                use formslcState
                implicit none
                integer*4 varInt
                isrm = varInt
        end

        subroutine setFirstLine(varInt)
                use formslcState
                implicit none
                integer varInt
                ifirstline = varInt
        end

        subroutine setNumberPatches(varInt)
                use formslcState
                implicit none
                integer varInt
                npatches = varInt
        end

        subroutine setFirstSample(varInt)
                use formslcState
                implicit none
                integer varInt
                ifirst = varInt
        end

        subroutine setAzimuthPatchSize(varInt)
                use formslcState
                implicit none
                integer varInt
                nnn = varInt
        end

        subroutine setNumberValidPulses(varInt)
                use formslcState
                implicit none
                integer varInt
                na_valid = varInt
        end

        subroutine setCaltoneLocation(varFloat)
                use formslcState
                implicit none
                real*4 varFloat
                caltone1 = varFloat
        end

        subroutine setStartRangeBin(varInt)
                use formslcState
                implicit none
                integer varInt
                isave = varInt
        end

        subroutine setNumberRangeBin(varInt)
                use formslcState
                implicit none
                integer varInt
                nlinesaz = varInt
        end

        subroutine setDopplerCentroidCoefficients(array1d,dim1)
                use formslcState
                implicit none
                integer dim1,i
                real*8, dimension(dim1):: array1d
                do i = 1, dim1
                        dopplerCoefficients(i) = array1d(i)
                enddo
        end

        subroutine setPlanetRadiusOfCurvature(varDouble)
                use formslcState
                implicit none
                real*8 varDouble
                re = varDouble
        end

        subroutine setBodyFixedVelocity(varFloat)
                use formslcState
                implicit none
                real*4 varFloat
                vel1 = varFloat
        end

        subroutine setSpacecraftHeight(varDouble)
                use formslcState
                implicit none
                real*8 varDouble
                ht1 = varDouble
        end

        subroutine setPlanetGravitationalConstant(varDouble)
                use formslcState
                implicit none
                real*8 varDouble
                gm = varDouble
        end

        subroutine setPointingDirection(varInt)
                use formslcState
                implicit none
                integer varInt
                i_lrl = varInt
        end

        subroutine setAntennaSCHVelocity(array1d,dim1)
                use formslcState
                implicit none
                integer dim1,i
                real*8, dimension(dim1):: array1d
                do i = 1, dim1
                        r_platvel1(i) = array1d(i)
                enddo
        end

        subroutine setAntennaSCHAcceleration(array1d,dim1)
                use formslcState
                implicit none
                integer dim1,i
                real*8, dimension(dim1):: array1d
                do i = 1, dim1
                        r_platacc1(i) = array1d(i)
                enddo
        end

        subroutine setRangeFirstSample(varDouble)
                use formslcState
                implicit none
                real*8 varDouble
                r001 = varDouble
        end

        subroutine setPRF(varFloat)
                use formslcState
                implicit none
                real*4 varFloat
                prf1 = varFloat
        end

        subroutine setInPhaseValue(varFloat)
                use formslcState
                implicit none
                real*4 varFloat
                xmi1 = varFloat
        end

        subroutine setQuadratureValue(varFloat)
                use formslcState
                implicit none
                real*4 varFloat
                xmq1 = varFloat
        end

        subroutine setIQFlip(varInt)
                use formslcState
                implicit none
                integer*4 varInt
                iflip = varInt
        end

        subroutine setAzimuthResolution(varFloat)
                use formslcState
                implicit none
                real*4 varFloat
                azres = varFloat
        end

        subroutine setNumberAzimuthLooks(varInt)
                use formslcState
                implicit none
                integer varInt
                nlooks = varInt
        end

        subroutine setRangeSamplingRate(varFloat)
                use formslcState
                implicit none
                real*4 varFloat
                fs = varFloat
        end

        subroutine setChirpSlope(varFloat)
                use formslcState
                implicit none
                real*4 varFloat
                slope = varFloat
        end

        subroutine setRangePulseDuration(varFloat)
                use formslcState
                implicit none
                real*4 varFloat
                pulsedur = varFloat
        end

        subroutine setRangeChirpExtensionPoints(varInt)
                use formslcState
                implicit none
                integer varInt
                nextend = varInt
        end

        subroutine setRadarWavelength(varDouble)
                use formslcState
                implicit none
                real*8 varDouble
                wavl = varDouble
        end

        subroutine setRangeSpectralWeighting(varFloat)
                use formslcState
                implicit none
                real*4 varFloat
                rhww = varFloat
        end

        subroutine setSpectralShiftFractions(array1d,dim1)
                use formslcState
                implicit none
                integer dim1,i
                real*4, dimension(dim1):: array1d
                do i = 1, dim1
                        spectralShiftFrac(i) = array1d(i)
                enddo
        end

        subroutine setLinearResamplingCoefficiets(array1d,dim1)
                use formslcState
                implicit none
                integer dim1,i
                real*8, dimension(dim1):: array1d
                do i = 1, dim1
                        linearResampCoeff(i) = array1d(i)
                enddo
        end

        subroutine setLinearResamplingDeltas(array1d,dim1)
                use formslcState
                implicit none
                integer dim1,i
                real*8, dimension(dim1):: array1d
                do i = 1, dim1
                        linearResampDeltas(i) = array1d(i)
                enddo
        end

