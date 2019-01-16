#ifndef icumoduleFortTrans_h
#define icumoduleFortTrans_h

  #if defined(NEEDS_F77_TRANSLATION)

        #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)

              #define icu_f icu_

              #define setWidth_f setwidth_
              #define setStartSample_f setstartsample_
              #define setEndSample_f setendsample_
              #define setStartingLine_f setstartingline_
              #define setLength_f setlength_
              #define setAzimuthBufferSize_f setazimuthbuffersize_
              #define setOverlap_f setoverlap_
              #define setFilteringFlag_f setfilteringflag_
              #define setUnwrappingFlag_f setunwrappingflag_
              #define setFilterType_f setfiltertype_
              #define setLPRangeWinSize_f setlprangewinsize_
              #define setLPAzimuthWinSize_f setlpazimuthwinsize_
              #define setFilterExponent_f setfilterexponent_
              #define setUseAmplitudeFlag_f setuseamplitudeflag_
              #define setCorrelationType_f setcorrelationtype_
              #define setCorrelationBoxSize_f setcorrelationboxsize_
              #define setPhaseSigmaBoxSize_f setphasesigmaboxsize_
              #define setPhaseVarThreshold_f setphasevarthreshold_
              #define setInitCorrThreshold_f setinitcorrthreshold_
              #define setCorrThreshold_f setcorrthreshold_
              #define setCorrThresholdInc_f setcorrthresholdinc_
              #define setNeuTypes_f setneutypes_
              #define setNeuThreshold_f setneuthreshold_
              #define setBootstrapSize_f setbootstrapsize_
              #define setNumTreeSets_f setnumtreesets_
              #define setTreeType_f settreetype_

        #else
            #error Unknown translation for FORTRAN external symbols
        #endif
  #endif
#endif //icumoduleFortTrans_h
