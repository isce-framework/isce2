      FUNCTION IDXCHG(X,Y,I1,I2,I3,I4)
!C THIS FUNCTION DETERMINES WHETHER OR NOT THE EXCHANGE OF TWO
!C TRIANGLES IS NECESSARY ON THE BASIS OF MAX-MIN-ANGLE CRITERION
!C BY C. L. LAWSON.
!C THE INPUT PARAMETERS ARE
!C    X,Y  = ARRAY CONTAINING THE COORDINATES OF THE DATA
!C           POINTS,
!C    I1,I2,I3,I4 = POINT NUMBERS OF FOUR POINTS P1, P2,
!C           P3, AND P4 THAT FORM A QUADRILATERAL WITH P3
!C           AND P4 CONNECTED DIAGONALLY.
!C  THIS FUNCTION RETURNS AN INTEGER VALUE 1 (ONE) WHEN AN EX-
!C  CHANGE IS NECESSARY, AND 0 (ZERO) OTHERWISE.
!C DECLARATION STATEMENTS
      DIMENSION   X(*),Y(*)
      EQUIVALENCE (C2SQ,C1SQ),(A3SQ,B2SQ),(B3SQ,A1SQ),
     1            (A4SQ,B1SQ),(B4SQ,A2SQ),(C4SQ,C3SQ)
!C PRELIMINARY PROCESSING
   10 X1=X(I1)
      Y1=Y(I1)
      X2=X(I2)
      Y2=Y(I2)
      X3=X(I3)
      Y3=Y(I3)
      X4=X(I4)
      Y4=Y(I4)
!C CALCULATION
   20 IDX=0
      U3=(Y2-Y3)*(X1-X3)-(X2-X3)*(Y1-Y3)
      U4=(Y1-Y4)*(X2-X4)-(X1-X4)*(Y2-Y4)
      IF(U3*U4.LE.0.0)    GO TO 30
      U1=(Y3-Y1)*(X4-X1)-(X3-X1)*(Y4-Y1)
      U2=(Y4-Y2)*(X3-X2)-(X4-X2)*(Y3-Y2)
      A1SQ=(X1-X3)**2+(Y1-Y3)**2
      B1SQ=(X4-X1)**2+(Y4-Y1)**2
      C1SQ=(X3-X4)**2+(Y3-Y4)**2
      A2SQ=(X2-X4)**2+(Y2-Y4)**2
      B2SQ=(X3-X2)**2+(Y3-Y2)**2
      C3SQ=(X2-X1)**2+(Y2-Y1)**2
      S1SQ=U1*U1/(C1SQ*AMAX1(A1SQ,B1SQ))
      S2SQ=U2*U2/(C2SQ*AMAX1(A2SQ,B2SQ))
      S3SQ=U3*U3/(C3SQ*AMAX1(A3SQ,B3SQ))
      S4SQ=U4*U4/(C4SQ*AMAX1(A4SQ,B4SQ))
      IF(AMIN1(S1SQ,S2SQ).LT.AMIN1(S3SQ,S4SQ))    IDX=1
   30 IDXCHG=IDX
      RETURN
      END


      SUBROUTINE IDTANG(NDP,THRES,XD,YD,ZD,NT,IPT,NL,IPL,IWL,IWP,WK)
!C THIS SUBROUTINE PERFORMS TRIANGULATION.  IT DIVIDES THE X-Y
!C PLANE INTO A NUMBER OF TRIANGLES ACCORDING TO GIVE DATA
!C POINTS IN THE PLANE, DETERMINES LINE SEGMENTS THAT FORM THE
!C BORDER OF DATA AREA, AND DETERMINES THE TRIANGLE NUMBERS
!C CORRESPONDING TO THE BORDER LINE SEGMENTS.
!C AT COMPLETION, POINT NUMBERS OF THE VERTEXES OF EACH TRIANGLE
!C ARE LISTED COUNTER-CLOCKWISE.  POINT NUMBERS OF THE END POINTS
!C OF EACH BORDER LINE SEGMENT ARE LISTED COUNTER-CLOCKWISE,
!C LISTING ORDER OF THE LINE SEGMENTS BEING COUNTER-CLOCKWISE.
!C THE LUN CONSTANT IN THE DATA INITIALIZATION STATEMENT IS THE
!C LOGICAL UNIT NUMBER OF THE STANDARD OUTPUT UNIT AND IS,
!C THEREFORE, SYSTEM DEPENDENT.
!C THIS SUBROUTINE CALLS THE IDXCHG FUNCTION.
!C THE INPUT PARAMETERS ARE
!C     NDP  = NUMBER OF DATA POINTS,
!C     XD  = ARRAY OF DIMENSION NDP CONTAINING THE
!C           X COORDINATES OF THE DATA POINTS
!C     YD  = ARRAY OF DIMENSION NDP CONTAINING THE
!C           Y COORDINATES OF THE DATA POINTS,
!C THE OUTPUT PARAMETERS ARE
!C     NT  = NUMBER OF TRIANGLES,
!C    IPT  = INTEGER ARRAY OF DIMENSION 6*NDP-15, WHERE THE
!C           POINT NUMBERS OF THE VERTEXES OF THE (IT)TH
!C           TRIANGLE ARE TO BE STORED AS THE (3*IT-2)ND,
!C           (3*IT-1)ST, AND (3*IT)TH ELEMENTS,
!C           IT=1,2,...,NT,
!C     NL  = NUMBER OF BORDER LINE SEGMENTS,
!C    IPL  = INTEGER ARRAY OF DIMENSION 6*NDP, WHERE THE
!C           POINT NUMBERS OF THE END POINTS OF THE (IL)TH
!C           BORDER LINE SEGMENT AND ITS RESPECTIVE TRIANGLE
!C           NUMBER ARE TO BE STORED AS THE (3*IL-2)ND,
!C           (3*IL-1)ST, AND (3*IL)TH ELEMENTS,
!C           IL=1,2,...,NL.
!C THE OTHER PARAMETERS ARE
!C     IWL = INTEGER ARRAY OF DIMENSION 18*NDP USED
!C           INTERNALLY AS A WORK AREA,
!C     IWK = INTEGER ARRAY OF DIMENSION NDP USED
!C           INTERNALLY AS A WORK AREA,
!C      WK = ARRAY OF DIMENSION NDP USED INTERNALLY AS A 
!C           WORK AREA.
!C DECLARATION STATEMENTS
      DIMENSION XD(*),YD(*),ZD(*),IPT(*),IPL(*),
     1          IWL(*),IWP(*),WK(*)
      DIMENSION ITF(2),I_IDV(1000)
      DATA RATIO/1.0E-6/, NREP/100/, LUN/6/
!C STATEMENT FUNCTIONS
      DSQF(U1,V1,U2,V2)=(U2-U1)**2+(V2-V1)**2
      SIDE(U1,V1,U2,V2,U3,V3)=(V3-V1)*(U2-U1)-(U3-U1)*(V2-V1)
!C PRELIMINARY PROCESSING
!c      write(*,*)'entered idtang'
   10 NDP0=NDP
      NDPM1=NDP0-1
      I_ID = 0
      IF(NDP0.LT.4)  GO TO 90
!C DETERMINES THE CLOSEST PAIR OF DATA POINTS AND THEIR MIDPOINT
   20 DSQMN=DSQF(XD(1),YD(1),XD(2),YD(2))
      IPMN1=1
      IPMN2=2
      DO 22   IP1=1,NDPM1
        X1=XD(IP1)
        Y1=YD(IP1)
        IP1P1=IP1+1
        DO K1=1,I_ID
          IF(I_IDV(K1) .EQ. IP1)THEN
             GO TO 22
          ENDIF
        ENDDO
        DO 21  IP2=IP1P1,NDP0
          DO K1=1,I_ID
            IF(I_IDV(K1) .EQ. IP2)THEN
              GO TO 21
            ENDIF
          ENDDO
          DSQI=DSQF(X1,Y1,XD(IP2),YD(IP2))
          IF(DSQI.LE.THRES)THEN
             I_ID = I_ID + 1
             I_IDV(I_ID) = IP2
             GO TO 21
          ENDIF
          IF(DSQI.GE.DSQMN)    GO TO 21
          DSQMN=DSQI
          IPMN1=IP1
          IPMN2=IP2
   21   CONTINUE
   22 CONTINUE
!C NOW REMOVE ALL IDENTICAL POINTS FROM THE DATA ARRAY
!      IF(I_ID .GT. 0)THEN
!c        write(*,*) ' '
!c        WRITE(*,*) 'Number of duplicate points = ',i_id
!      ENDIF
      K3 = NDP0
      K4 = 0
      NDP0 = NDP0 - I_ID
      NDP = NDP0
      DO K1=K3,NDP0+1,-1
        IF(K1 .NE. I_IDV(I_ID))THEN
            K4 = K4 + 1 
            XD(I_IDV(K4)) = XD(K1)
            YD(I_IDV(K4)) = YD(K1)
            ZD(I_IDV(K4)) = ZD(K1)
        ELSE
            I_ID = I_ID - 1
        ENDIF
      ENDDO
      DSQ12=DSQMN
      XDMP=(XD(IPMN1)+XD(IPMN2))/2.0
      YDMP=(YD(IPMN1)+YD(IPMN2))/2.0
!C SORTS THE OTHER (NDP-2) DATA POINTS IN ASCENDING ORDER OF
!C DISTANCE FROM THE MIDPOINT AND STORES THE SORTED DATA POINT
!C NUMBERS IN THE IWP ARRAY.
   30 JP1=2
      DO 31   IP1=1,NDP0
        IF(IP1.EQ.IPMN1.OR.IP1.EQ.IPMN2)       GO TO 31
        JP1=JP1+1
        IWP(JP1)=IP1
        WK(JP1)=DSQF(XDMP,YDMP,XD(IP1),YD(IP1))
   31 CONTINUE
      DO 33   JP1=3,NDPM1
        DSQMN=WK(JP1)
        JPMN=JP1
        DO 32   JP2=JP1,NDP0
          IF(WK(JP2).GE.DSQMN)      GO TO 32
          DSQMN=WK(JP2)
          JPMN=JP2
   32   CONTINUE
        ITS=IWP(JP1)
        IWP(JP1)=IWP(JPMN)
        IWP(JPMN)=ITS
        WK(JPMN)=WK(JP1)
   33 CONTINUE
!C IF NECESSARY, MODIFIES THE ORDERING IN SUCH A WAY THAT THE
!C FIRST THREE DATA POINTS ARE NOT COLLINEAR.
   35 AR=DSQ12*RATIO
      X1=XD(IPMN1)
      Y1=YD(IPMN1)
      DX21=XD(IPMN2)-X1
      DY21=YD(IPMN2)-Y1
      IF(DX21 .EQ. 0 .AND. DY21 .EQ. 0)THEN
        WRITE(*,*) 'IPMN1,IPMN2 = ',IPMN1,IPMN2
        WRITE(*,*) 'XD(1),YD(1) = ',XD(IPMN1),YD(IPMN1)
        WRITE(*,*) 'XD(2),YD(2) = ',XD(IPMN2),YD(IPMN2)
      ENDIF  
      DO 36   JP=3,NDP0
        IP=IWP(JP)
        IF(ABS((YD(IP)-Y1)*DX21-(XD(IP)-X1)*DY21).GT.AR)
     1               GO TO 37
   36 CONTINUE
      GO TO 92
   37 IF(JP.EQ.3)    GO TO 40
      JPMX=JP
      JP=JPMX+1
      DO 38  JPC=4,JPMX
        JP=JP-1
        IWP(JP)=IWP(JP-1)
   38 CONTINUE
      IWP(3)=IP
!C FORMS THE FIRST TRIANGLE.  STORES POINT NUMBER OF THE VER-
!C TEXES OF THE TRIANGLE IN THE IPT ARRAY, AND STORES POINT NUM-
!C BERS OF THE BORDER LINE SEGMENTS AND THE TRIANGLE NUMBER IN
!C THE IPL ARRAY.
   40 IP1=IPMN1
      IP2=IPMN2
      IP3=IWP(3)
      IF(SIDE(XD(IP1),YD(IP1),XD(IP2),YD(IP2),XD(IP3),YD(IP3))
     1     .GE.0.0)       GO TO 41
      IP1=IPMN2
      IP2=IPMN1
   41 NT0=1
      NTT3=3
      IPT(1)=IP1
      IPT(2)=IP2
      IPT(3)=IP3
      NL0=3
      NLT3=9
      IPL(1)=IP1
      IPL(2)=IP2
      IPL(3)=1
      IPL(4)=IP2
      IPL(5)=IP3
      IPL(6)=1
      IPL(7)=IP3
      IPL(8)=IP1
      IPL(9)=1
!C ADDS THE REMAINING (NDP-3) DATA POINTS, ONE BY ONE.
   50 DO 79   JP1=4,NDP0
        IP1=IWP(JP1)
        X1=XD(IP1)
        Y1=YD(IP1)
!C - DETERMINES THE VISIBLE BORDER LINE SEGMENTS.
        IP2=IPL(1)
        JPMN=1
        DXMN=XD(IP2)-X1
        DYMN=YD(IP2)-Y1
        DSQMN=DXMN**2+DYMN**2
        ARMN=DSQMN*RATIO
        JPMX=1
        DXMX=DXMN
        DYMX=DYMN
        DSQMX=DSQMN
        ARMX=ARMN
        DO 52  JP2=2,NL0
         IP2=IPL(3*JP2-2)
         DX=XD(IP2)-X1
         DY=YD(IP2)-Y1
         AR=DY*DXMN-DX*DYMN
         IF(AR.GT.ARMN)       GO TO 51
         DSQI=DX**2+DY**2
         IF(AR.GE.(-ARMN).AND.DSQI.GE.DSQMN)        GO TO 51
         JPMN=JP2
         DXMN=DX
         DYMN=DY
         DSQMN=DSQI
         ARMN=DSQMN*RATIO
   51 AR=DY*DXMX-DX*DYMX
      IF(AR.LT.(-ARMX))     GO TO 52
      DSQI=DX**2+DY**2
      IF(AR.LE.ARMX.AND.DSQI.GE.DSQMX)     GO TO 52
      JPMX=JP2
      DXMX=DX
      DYMX=DY
      DSQMX=DSQI
      ARMX=DSQMX*RATIO
   52 CONTINUE
      IF(JPMX.LT.JPMN)  JPMX=JPMX+NL0
      NSH=JPMN-1
      IF(NSH.LE.0)      GO TO 60
!C - SHIFTS (ROTATES) THE IPL ARRAY TO HAVE THE INVISIBLE BORDER
!C - LINE SEGMENTS CONTAINED IN THE FIRST PART OF THE IPL ARRAY.
      NSHT3=NSH*3
      DO 53  JP2T3=3,NSHT3,3
        JP3T3=JP2T3+NLT3
        IPL(JP3T3-2)=IPL(JP2T3-2)
        IPL(JP3T3-1)=IPL(JP2T3-1)
        IPL(JP3T3)  =IPL(JP2T3)
   53 CONTINUE
      DO 54  JP2T3=3,NLT3,3
        JP3T3=JP2T3+NSHT3
        IPL(JP2T3-2)=IPL(JP3T3-2)
        IPL(JP2T3-1)=IPL(JP3T3-1)
        IPL(JP2T3)  =IPL(JP3T3)
   54 CONTINUE
      JPMX=JPMX-NSH
!C - ADDS TRIANGLES TO THE IPT ARRAY, UPDATES BORDER LINE
!C - SEGMENTS IN THE IPL ARRAY, AND SETS FLAGS FOR THE BORDER
!C - LINE SEGMENTS TO BE REEXAMINED IN THE IWL ARRAY.
   60   JWL=0
        DO 64   JP2=JPMX,NL0
          JP2T3=JP2*3
          IPL1=IPL(JP2T3-2)
          IPL2=IPL(JP2T3-1)
          IT  =IPL(JP2T3)
!C - - ADDS A TRIANGLE THE IPT ARRAY.
          NT0=NT0+1
          NTT3=NTT3+3
          IPT(NTT3-2)=IPL2
          IPT(NTT3-1)=IPL1
          IPT(NTT3)  =IP1
!C - - UPDATES BORDER LINE SEGMENTS IN THE IPL ARRAY.
          IF(JP2.NE.JPMX)      GO TO 61
          IPL(JP2T3-1)=IP1
          IPL(JP2T3)  =NT0
   61     IF(JP2.NE.NL0)       GO TO 62
          NLN=JPMX+1
          NLNT3=NLN*3
          IPL(NLNT3-2)=IP1
          IPL(NLNT3-1)=IPL(1)
          IPL(NLNT3)  =NT0
!C - - DETERMINES THE VERTEX THAT DOES NOT LIE ON THE BORDER
!C - - LINE SEGMENTS.
   62     ITT3=IT*3
          IPTI=IPT(ITT3-2)
          IF(IPTI.NE.IPL1.AND.IPTI.NE.IPL2)   GO TO 63
          IPTI=IPT(ITT3-1)
          IF(IPTI.NE.IPL1.AND.IPTI.NE.IPL2)   GO TO 63
          IPTI=IPT(ITT3)
!C - - CHECKS IF THE EXCHANGE IS NECESSARY.
   63     IF(IDXCHG(XD,YD,IP1,IPTI,IPL1,IPL2).EQ.0)     GO TO 64
!C - - MODIFIES THE IPT ARRAY WHEN NECESSARY
          IPT(ITT3-2)=IPTI
          IPT(ITT3-1)=IPL1
          IPT(ITT3)  =IP1
          IPT(NTT3-1)=IPTI
          IF(JP2.EQ.JPMX)        IPL(JP2T3)=IT
          IF(JP2.EQ.NL0.AND.IPL(3).EQ.IT)      IPL(3)=NT0
!C - - SETS FLAGS IN THE IWL ARRAY.
          JWL=JWL+4
          IWL(JWL-3)=IPL1
          IWL(JWL-2)=IPTI
          IWL(JWL-1)=IPTI
          IWL(JWL)  =IPL2
   64 CONTINUE
      NL0=NLN
      NLT3=NLNT3
      NLF=JWL/2
      IF(NLF.EQ.0)      GO TO 79
!C - IMPROVES TRIANGULATION.
   70   NTT3P3=NTT3+3
        DO 78  IREP=1,NREP
         DO 76   ILF=1,NLF
            ILFT2=ILF*2
            IPL1=IWL(ILFT2-1)
            IPL2=IWL(ILFT2)
!C - - LOCATES THE IPT ARRAY TWO TRIANGLES ON BOTH SIDES OF
!C - - THE FLAGGED LINE SEGMENT.
            NTF=0
            DO 71  ITT3R=3,NTT3,3
              ITT3=NTT3P3-ITT3R
              IPT1=IPT(ITT3-2)
              IPT2=IPT(ITT3-1)
              IPT3=IPT(ITT3)
              IF(IPL1.NE.IPT1.AND.IPL1.NE.IPT2.AND.
     1            IPL1.NE.IPT3)      GO TO 71
              IF(IPL2.NE.IPT1.AND.IPL2.NE.IPT2.AND.
     1            IPL2.NE.IPT3)      GO TO 71
                 NTF=NTF+1
                 ITF(NTF)=ITT3/3
                 IF(NTF.EQ.2)      GO TO 72
   71          CONTINUE
               IF(NTF.LT.2)        GO TO 76
!C - - DETERMINES THE VERTEXES OF THE TRIANGLES THAT DO NOT LIE
!C - - ON THE LINE SEGMENT.
   72     IT1T3=ITF(1)*3
          IPTI1=IPT(IT1T3-2)
          IF(IPTI1.NE.IPL1.AND.IPTI1.NE.IPL2)   GO TO 73
          IPTI1=IPT(IT1T3-1)
          IF(IPTI1.NE.IPL1.AND.IPTI1.NE.IPL2)   GO TO 73
          IPTI1=IPT(IT1T3)
   73     IT2T3=ITF(2)*3
          IPTI2=IPT(IT2T3-2)
          IF(IPTI2.NE.IPL1.AND.IPTI2.NE.IPL2)   GO TO 74
          IPTI2=IPT(IT2T3-1)
          IF(IPTI2.NE.IPL1.AND.IPTI2.NE.IPL2)   GO TO 74
          IPTI2=IPT(IT2T3)
!C - - CHECKS IF THE EXCHANGE IS NECESSARY.
   74     IF(IDXCHG(XD,YD,IPTI1,IPTI2,IPL1,IPL2).EQ.0)
     1       GO TO 76
!C - - MODIFIES THE IPT ARRAY WHEN NECESSARY
          IPT(IT1T3-2)=IPTI1
          IPT(IT1T3-1)=IPTI2
          IPT(IT1T3)  =IPL1
          IPT(IT2T3-2)=IPTI2
          IPT(IT2T3-1)=IPTI1
          IPT(IT2T3)  =IPL2
!C - - SETS NEW FLAGS.
           JWL=JWL+8
           IWL(JWL-7)=IPL1
           IWL(JWL-6)=IPTI1
           IWL(JWL-5)=IPTI1
           IWL(JWL-4)=IPL2
           IWL(JWL-3)=IPL2
           IWL(JWL-2)=IPTI2
           IWL(JWL-1)=IPTI2
           IWL(JWL)  =IPL1
           DO 75   JLT3=3,NLT3,3
             IPLJ1=IPL(JLT3-2)
             IPLJ2=IPL(JLT3-1)
             IF((IPLJ1.EQ.IPL1.AND.IPLJ2.EQ.IPTI2).OR.
     1          (IPLJ2.EQ.IPL1.AND.IPLJ1.EQ.IPTI2))
     2                         IPL(JLT3)=ITF(1)
             IF((IPLJ1.EQ.IPL2.AND.IPLJ2.EQ.IPTI1).OR.
     1          (IPLJ2.EQ.IPL2.AND.IPLJ1.EQ.IPTI1))
     2                         IPL(JLT3)=ITF(2)
   75       CONTINUE
   76     CONTINUE
          NLFC=NLF
          NLF=JWL/2
          IF(NLF.EQ.NLFC)       GO TO 79
!C - - RESETS THE IWL ARRAY FOR THE NEXT ROUND.
          JWL=0
          JWL1MN=(NLFC+1)*2
          NLFT2=NLF*2
          DO 77  JWL1=JWL1MN,NLFT2,2
            JWL=JWL+2
            IWL(JWL-1)=IWL(JWL1-1)
            IWL(JWL)  =IWL(JWL1)
   77     CONTINUE
          NLF=JWL/2
   78   CONTINUE
   79 CONTINUE
!C REARRANGES THE IPT ARRAY SO THAT THE VERTEXES OF EACH TRIANGLE
!C ARE LISTED COUNTER-CLOCKWISE.
   80 DO 81  ITT3=3,NTT3,3
        IP1=IPT(ITT3-2)
        IP2=IPT(ITT3-1)
        IP3=IPT(ITT3)
        IF(SIDE(XD(IP1),YD(IP1),XD(IP2),YD(IP2),XD(IP3),YD(IP3))
     1           .GE.0.0)       GO TO 81
        IPT(ITT3-2)=IP2
        IPT(ITT3-1)=IP1
   81 CONTINUE
      NT=NT0
      NL=NL0
      RETURN
!C ERROR EXIT
   90 WRITE (LUN,2090)   NDP0
      GO TO 93
   91 WRITE (LUN,2091)   NDP0,IP1,IP2,X1,Y1
      GO TO 93
   92 WRITE (LUN,2092)   NDP0
   93 WRITE (LUN,2093)
      NT=0
      RETURN
!C FORMAT STATEMENTS
 2090 FORMAT(1X/23H ***   NDP LESS THAN 4./8H   NDP =,I5)
 2091 FORMAT(1X/29H ***   IDENTICAL DATA POINTS./
     1   8H   NDP =,I5,5X,5HIP1 =,I5,5X,5HIP2 =,I5,
     2   5X,4HXD =,E12.4,5X,4HYD =,E12.4)
 2092 FORMAT(1X/33H ***   ALL COLLINEAR DATA POINTS./
     1   8H   NDP =,I5)
 2093 FORMAT(35H ERROR DETECTED IN ROUTINE   IDTANG/)
      END

      SUBROUTINE IDSFFT(MD,NCP,NDP,XD,YD,ZD,NXI,NYI,XI,YI,ZI,
     1                  THRES,IWK,WK,iwrksize)
!C THIS SUBROUTINE PERFORMS SMOOTH SURFACE FITTING WHEN THE PRO-
!C JECTIONS OF THE DATA POINTS IN THE X-Y PLANE ARE IRREGULARLY
!C DISTRIBUTED IN THE PLANE.
!C THE INPUT PARAMETERS ARE
!C     MD  = MODE OF COMPUTATION (MUST BE 1, 2, OR 3),
!C         = 1 FOR NEW NCP AND/OR NEW XD-YD,
!C         = 2 FOR OLD NCP, OLD XD-YD, NEW XI-YI,
!C         = 3 FOR OLD NCP, OLD XD-YD, OLD XI-YI,
!C    NCP  = NUMBER OF ADDITIONAL DATA POINTS USED FOR ESTI-
!C           MATING PARTIAL DERIVATIVES AT EACH DATA POINT
!C           (MUST BE 2 OR GREATER, BUT SMALLER THAN NDP),
!C    NDP  = NUMBER OF DATA POINTS (MUST BE 4 OR GREATER),
!C     XD  = ARRAY OF DIMENSION NDP CONTAINING THE X, 
!C           COORDINATES OF THE DATA POINTS
!C     YD  = ARRAY OF DIMENSION NDP CONTAINING THE Y, 
!C           COORDINATES OF THE DATA POINTS
!C     ZD  = ARRAY OF DIMENSION NDP CONTAINING THE Z, 
!C           COORDINATES OF THE DATA POINTS
!C    NXI  = NUMBER OF OUTPUT GRID POINTS IN THE X COORDINATE
!C           (MUST BE 1 OR GREATER)
!C    NYI  = NUMBER OF OUTPUT GRID POINTS IN THE Y COORDINATE
!C           (MUST BE 1 OR GREATER)
!C     XI  = ARRAY OF DIMENSION NXI CONTAINING THE X 
!C           COORDINATES OF THE OUTPUT GRID POINTS
!C     YI  = ARRAY OF DIMENSION NXI CONTAINING THE Y
!C           COORDINATES OF THE OUTPUT GRID POINTS.
!C THE OUTPUT PARAMETER IS
!C     ZI  = DOUBLY-DIMENSIONED ARRAY OF DIMENSION (NXI,NYI),
!C           WHERE THE INTERPOLATED Z VALUES AT THE OUTPUT
!C           GRID POINTS ARE TO BE STORED.
!C THE OTHER PARAMETERS ARE
!C     IWK = INTEGER ARRAY OF DIMENSION
!C              MAX0(31,27+NCP)*NDP+NXI*NYI
!C           USED INTERNALLY AS A WORK AREA,
!C      WK = ARRAY OF DIMENSION 5*NDP USED INTERNALLY AS A 
!C           WORK AREA.
!C THE VERY FIRST CALL TO THIS SUBROUTINE AND THE CALL WITH A NEW
!C NCP VALUE, A NEW NDP VALUE, AND/OR NEW CONTENTS OF THE XD AND 
!C YD ARRAYS MUST BE MADE WITH MD=1.  THE CALL WITH MD-2 MUST BE 
!C PRECEDED BY ANOTHER CALL WITH THE SAME NCP AND NDP VALUES AND 
!C WITH THE SAME CONTENTS OF THE XD AND YD ARRAYS.  THE CALL WITH
!C MD-3 MUST BE PRECEDED BY ANOTHER CALL WITH THE SAME NCP, NDP,
!C NXI, AND NYI VALUES AND WITH THE SAME CONTENTS OF THE XD, YD,
!C XI, AND YI ARRAYS.  BETWEEN THE CALL WITH MD=2 OR MD=3 AND ITS
!C PRECEDING CALL, THE IWK AND WK ARRAYS MUST NOT BE DISTURBED.
!C USE OF A VALUE BETWEEN 3 AND 5 (INCLUSIVE) FOR NCP IS RECOM-
!C MENDED UNLESS THERE ARE EVIDENCES THAT DICTATE OTHERWISE.
!C THE LUN CONSTANT IN THE DATA INITIALIZATION STATEMENT IS THE
!C LOGICAL UNIT NUMBER OF THE STANDARD OUTPUT UNIT AND IS,
!C THEREFORE, SYSTEM DEPENDENT.
!C THIS SUBROUTINE CALLS THE IDCLDP, IDGRID, IDPDRV, IDPTIP, AND
!C IDTANG SUBROUTINES.
!C DECLARATION STATEMENTS
      DIMENSION XD(iwrksize),YD(iwrksize),ZD(iwrksize),
     1     XI(iwrksize),YI(iwrksize),
     1     ZI(iwrksize),IWK(*),WK(*)
!c      COMMON/IDPI/ITPV
      DATA LUN/6/
!C SETTING OF SOME INPUT PARAMETERS TO LOCAL VARIABLES.
!C (FOR MD=1,2,3)

!c the following line is a Kludge to trick the HP optimizer
!c
      if(k .ne.k) write(*,*)'md, ncp,ndp,nxi,nyi: ',md, ncp,ndp,nxi,nyi
!c
!c
   10 MD0=MD
      NCP0=NCP
      NDP0=NDP
      NXI0=NXI
      NYI0=NYI
!C ERROR CHECK.  (FOR MD=1,2,3)
   20 IF(MD0.LT.1.OR.MD0.GT.3)           then
         WRITE (LUN,*) 'Value adjustment = ',MD0,NCP0,NDP0,NXI0,NYI0
         RETURN
      end if
      IF(NCP0.LT.2.OR.NCP0.GE.NDP0)      then
         WRITE (LUN,*) 'Value adjustment = ',MD0,NCP0,NDP0,NXI0,NYI0
         RETURN
      end if
      IF(NDP0.LT.4)                      then
         WRITE (LUN,*) 'Value adjustment = ',MD0,NCP0,NDP0,NXI0,NYI0
         RETURN
      end if
      IF(NXI0.LT.1.OR.NYI0.LT.1)         then
         WRITE (LUN,*) 'Value adjustment = ',MD0,NCP0,NDP0,NXI0,NYI0
         RETURN
      end if
      IF(MD0.GE.2) then
         NCPPV=IWK(1)
         NDPPV=IWK(2)
         NT=IWK(5)
         NL=IWK(6)
         IF(NCP0.NE.NCPPV)  then
            WRITE (LUN,*) 'Value adjustment = ',MD0,NCP0,NDP0,NXI0,NYI0
            RETURN
         end if
         IF(NDP0.NE.NDPPV)  then
            WRITE (LUN,*) 'Value adjustment = ',MD0,NCP0,NDP0,NXI0,NYI0
            RETURN
         end if
      else
         IWK(1)=NCP0
         IWK(2)=NDP0
      end if
      IF(MD0.GE.3)  then
         NXIPV=IWK(3)
         NYIPV=IWK(4)
         IF(NXI0.NE.NXIPV)  then
            WRITE (LUN,*) 'Value adjustment = ',MD0,NCP0,NDP0,NXI0,NYI0
            RETURN
         end if
         IF(NYI0.NE.NYIPV)  then
            WRITE (LUN,*) 'Value adjustment = ',MD0,NCP0,NDP0,NXI0,NYI0
            RETURN
         end if
      else
         IWK(3)=NXI0
         IWK(4)=NYI0
      end if
!C     ALLOCATION OF STORAGE AREAS IN THE IWK ARRAY.  (FOR MD=1,2,3)

!c         write(*,*)'in jw initialization',ndp0
      JWIPT=16
      JWIWL=6*NDP0+1
      JWNGP0=JWIWL-1
      JWIPL=24*NDP0+1
      JWIWP=30*NDP0+1
      JWIPC=27*NDP0+1
      JWIGP0=MAX0(31,27+NCP0)*NDP0
!c         write(*,*)'bb',jwipt,jwipl,jwiwl,jwiwp


!C TRIANGULATES THE X-Y PLANE.   (FOR MD=1)
 40   IF(MD0.LE.1)   then

!c********extra comments*********
!c      write(*,*) ' '
!c      write(*,*) 'Starting Triangulation...'
!c********extra comments*********      

      
!c      write(*,*)NDP0,THRES,XD,YD,ZD,NT
!c      write(*,*)'aa',jwipt,jwipl,jwiwl,jwiwp
!c      write(*,*)'a'
!c      write(*,*)IWK(JWIPT)
!c      write(*,*)IWK(JWIPL)
!c      write(*,*)IWK(JWIWL)
!c      write(*,*)IWK(JWIWP)
!c      write(*,*)'b'
!c      write(*,*) WK(1)
         CALL IDTANG(NDP0,THRES,XD,YD,ZD,NT,IWK(JWIPT),NL,IWK(JWIPL),
     1        IWK(JWIWL),IWK(JWIWP),WK)

!c********extra comments*********
!c       write(*,*) ' '
!c       write(*,*) 'Finished Triangulation ....'
!c       write(*,*) 'Number of triangles and edges = ',nt,nl
!c********extra comments*********      

         NDP = NDP0
         IWK(2) = NDP
         NDPPV = NDP
         
         
         IWK(5)=NT
         IWK(6)=NL

!c     Consistency check
!c
!c        IF(NT.EQ.0)    then
!c          write(*,*)'NT.eq.0 untrapped'
!c          write(*,*)ndp0
!c          return
!c        end if

         IF(NT.EQ.0)    RETURN
      end if
!C DETERMINES NCP POINTS CLOSEST TO EACH DATA POINT.  (FOR MD=1)
      IF(MD0.LE.1) then

!c********extra comments*********
!c         write(*,*) ' '
!c         write(*,*) 'Closest points determination...'
!c********extra comments*********      

         CALL IDCLDP(NDP0,XD,YD,NCP0,IWK(JWIPC))

!c********extra comments*********
!c       write(*,*) ' '
!c       write(*,*) 'Finished determining closest points...'
!c********extra comments*********      

!c     Consistency check
!c
!c        if(IWK(JWIPC).EQ.0) then
!c          write(*,*)'IWK(JWIPC).EQ.0 untrapped'
!c          write(*,*)ndp0
!c          stop
!c        end if
!c
!c        do i=0,(NDP0*NCP0-1)
!c          if((IWK(JWIPC+i).LE.0).or.(IWK(JWIPC+i).GT.NDP0)) then
!c            write(*,*)'IWK(JWIPC+i).LE.0 .or ... untrapped'
!c            write(*,*)IWK(JWIPC+i),jwipc,i,NDP0,NCP0
!c            stop
!c          end if
!c        end do

         IF(IWK(JWIPC).EQ.0)      RETURN
      end if

!C SORTS OUTPUT GRID POINTS IN ASCENDING ORDER OF THE TRIANGLE
!C NUMBER OF THE BORDER LINE SEGMENT NUMBER.  (FOR MD=1,2)
      IF(MD0.NE.3)  then

!c********extra comments*********
!c       write(*,*) ' '
!c       write(*,*) 'Assigning grid points to triangles...'
!c********extra comments*********      

         CALL IDGRID(XD,YD,NT,IWK(JWIPT),NL,IWK(JWIPL),NXI0,NYI0,
     1        XI,YI,IWK(JWNGP0+1),IWK(JWIGP0+1))

!c********extra comments*********
!c        write(*,*) ' '
!c        write(*,*) 'Finished grid point to triangle assignment...'
!c********extra comments*********      

!C ESTIMATES PARTIAL DERIVATIVES AT ALL DATA POINTS.
!C (FOR MD=1,2,3)

!c********extra comments*********
!c        write(*,*) ' '
!c        write(*,*) 'Estimate partial derivatives at data points...'
!c********extra comments*********      

!c     Consistency check
!c
!c       do i=0,(NDP0*NCP0-1)
!c         IF((IWK(JWIPC+i).LE.0).or.(IWK(JWIPC+i).GT.NDP0)) then
!c           write(*,*)'yyIWK(JWIPC+i).LE.0 .or ... untrapped'
!c           write(*,*)IWK(JWIPC+i),jwipc,i,NDP0,NCP0
!c           write(*,*) 'md, md0',md,md0
!c           stop
!c         end if
!c       end do

      end if


!c     Consistency check
!c
!c     do i=0,(NDP0*NCP0-1)
!c       IF((IWK(JWIPC+i).LE.0).or.(IWK(JWIPC+i).GT.NDP0)) then
!c         write(*,*)'xxIWK(JWIPC+i).LE.0 .or ... untrapped'
!c         write(*,*)IWK(JWIPC+i),jwipc,i,NDP0,NCP0
!c         write(*,*) 'md, md0',md,md0
!c         stop
!c       end if
!c     end do

      CALL IDPDRV(NDP0,XD,YD,ZD,NCP0,IWK(JWIPC),WK)

!c********extra comments*********
!c        write(*,*) ' '
!c        write(*,*) 'Finished partial derivative computation...'
!c********extra comments*********      

!c********extra comments*********
!c        write(*,*) ' '
!c        write(*,*) 'Starting interpolation to grid points...'
!c********extra comments*********      

!C INTERPOLATES THE ZI VALUES.  (FOR MD=1,2,3)
      ITPV=0
      JIG0MX=0
      JIG1MN=NXI0*NYI0+1
      NNGP=NT+2*NL
      DO JNGP=1,NNGP
         ITI=JNGP
         IF(JNGP.GT.NT)    then
            IL1=(JNGP-NT+1)/2
            IL2=(JNGP-NT+2)/2
            IF(IL2.GT.NL)      IL2=1
            ITI=IL1*(NT+NL)+IL2
         end if
 81      JWNGP=JWNGP0+JNGP
         NGP0=IWK(JWNGP)
         IF(NGP0.NE.0) then
            JIG0MN=JIG0MX+1
            JIG0MX=JIG0MX+NGP0
            DO  JIGP=JIG0MN,JIG0MX
               JWIGP=JWIGP0+JIGP
               IZI=IWK(JWIGP)
               IYI=(IZI-1)/NXI0+1
               IXI=IZI-NXI0*(IYI-1)
!c              setting 'itpv=0' before each call will cause
!c              IDPTIP to recalculate rather than cache values between calls
!c              itpv=0
               CALL IDPTIP(XD,YD,ZD,NT,IWK(JWIPT),NL,IWK(JWIPL),WK,
     1              ITI,XI(IXI),YI(IYI),ZI(IZI),itpv)
            end do
         end if
 86      JWNGP=JWNGP0+2*NNGP+1-JNGP
         NGP1=IWK(JWNGP)
         IF(NGP1.NE.0)  then
            JIG1MX=JIG1MN-1
            JIG1MN=JIG1MN-NGP1
            DO  JIGP=JIG1MN,JIG1MX
               JWIGP=JWIGP0+JIGP
               IZI=IWK(JWIGP)
               IYI=(IZI-1)/NXI0+1
               IXI=IZI-NXI0*(IYI-1)
!c              setting 'itpv=0' before each call will cause
!c              IDPTIP to recalculate rather than cache values between calls
!c              itpv=0
               CALL IDPTIP(XD,YD,ZD,NT,IWK(JWIPT),NL,IWK(JWIPL),WK,
     1              ITI,XI(IXI),YI(IYI),ZI(IZI),itpv)
            end do
         end if
      end do

!c********extra comments*********
!c        write(*,*) ' '
!c        write(*,*) 'Completed interpolation...'
!c********extra comments*********      

      RETURN
!C ERROR EXIT
!C FORMAT STATEMENT FOR ERROR MESSAGE
 2090 FORMAT(1X/' *** IMPROPER INPUT PARAMETER VALUE(S).'/
     1   '   MD =',I4,10X,'NCP =',I6,10X,'NDP =',I6,
     2   10X,'NXI =',I6,10X,'NYI =',I6/
     3   35H ERROR DETECTED IN ROUTINE   IDSFFT/)
      END

      SUBROUTINE IDPTIP(XD,YD,ZD,NT,IPT,NL,IPL,PDD,ITI,XII,YII,
     1     ZII,itpv)
!C     THIS SUBROUTINE PERFORMS PUNCTUAL INTERPOLATION OR EXTRAPOLA-
!C     TION, I.E., DETERMINES THE Z VALUE AT A POINT.
!C     THE INPUT PARAMETERS ARE
!C     XD,YD,ZD = ARRAYS OF DIMENSION NDP CONTAINING THE X, 
!C     Y, AND Z COORDINATES OF THE DATA POINTS, WHERE
!C     NDP IS THE NUMBER OF THE DATA POINTS,
!C     NT  = NUMBER OF TRIANGLES.
!C     IPT = INTEGER ARRAY OF DIMENSION 3*NT CONTAINING THE
!C     POINT NUMBERS OF THE VERTEXES OF THE TRIANGLES,
!C     NL  = NUMBER OF BORDER LINE SEGMENTS,
!C     IPL = INTEGER ARRAY OF DIMENSION 3*NL CONTAINING THE 
!C     POINT NUMBERS OF THE END POINTS OF THE BORDER
!C     LINE SEGMENTS AND THEIR RESPECTIVE TRIANGLE
!C     NUMBERS,
!C     PDD = ARRAY OF DIMENSION 5*NDF CONTAINING THE PARTIAL 
!C     DERIVATIVES AT THE DATA POINTS,
!C     ITI  = TRIANGLE NUMBER OF THE TRIANGLE IN WHICH LIES
!C     THE POINT FOR WHICH INTERPOLATION IS TO BE PERFORMED,
!C     XII,YII = X AND Y COORDINATES OF THE POINT FOR WHICH
!C     INTERPOLATION IS TO BE PERFORMED.
!C     THE OUTPUT PARAMETERS IS
!C     ZII  = INTERPOLATED Z VALUE.
!C     DECLARATION STATEMENTS
      DIMENSION XD(*), YD(*), ZD(*),IPT(*), IPL(*),
     1     PDD(*)
!c     COMMON/IDPI/ITPV
      DIMENSION X(3),Y(3),Z(3),PD(15),
     1     ZU(3),ZV(3),ZUU(3),ZUV(3),ZVV(3)
      REAL      LU,LV
!c     EQUIVALENCE (P5,P50)

!c     The code segments in the "IF(IT0.ne.ITPV)" blocks
!c     below compute values that are assumed to
!c     presist between calls to this subroutine.
!c     This is implementing a cache for the single most recently
!c     computed polynomial coefficients.
!c     Fortran requires persistence variables to be explicity
!c     identified with SAVE attribute.

!c     The lack of these SAVE statements was a bug that
!c     manifest as core dump when compiled with optimization.

      save X, Y
      save X0, Y0
      save AP, BP
      save CP, DP

      save P00, P01, P02, P03, P04, P05
      save P10, P11, P12, P13, P14
      save P20, P21, P22, P23
      save P30, P31, P32
      save P40, P41
      save P50

!C     PRELIMINARY PROCESSING
      if(k .ne. k) write(*,*) 'iti ',iti
 10   IT0=ITI
      NTL=NT+NL
      IF(IT0.LE.NTL) then       ! goto 20
!C     CALCULATION OF ZII BY INTERPOLATION.
!C     CHECKS IF THE NECESSARY COEFFICIENTS HAVE BEEN CALCULATED.
 20      IF(IT0.ne.ITPV) then   
!C     LOADS COORDINATE AND PARTIAL DERIVATIVE VALUES AT THE
!C     VERTEXES.
            JIPT=3*(IT0-1)
            JPD=0
            DO I=1,3
               JIPT=JIPT+1
               IDP=IPT(JIPT)
               X(I)=XD(IDP)
               Y(I)=YD(IDP)
               Z(I)=ZD(IDP)
               JPDD=5*(IDP-1)
               DO KPD=1,5
                  JPD=JPD+1
                  JPDD=JPDD+1
                  PD(JPD)=PDD(JPDD)
               end do
            end do
!C     DETERMINES THE COEFFICIENTS FOR THE COORDINATE SYSTEM
!C     TRANSFORMATION FROM THE X-Y SYSTEM TO THE U-V SYSTEM
!C     AND VICE VERSA.
            X0=X(1)
            Y0=Y(1)
            A=X(2)-X0
            B=X(3)-X0
            C=Y(2)-Y0
            D=Y(3)-Y0
            AD=A*D
            BC=B*C
            DLT=AD-BC
            IF(DLT .EQ. 0)THEN
               WRITE(*,*) 'X = ',X(1),X(2),X(3)
               WRITE(*,*) 'Y = ',Y(1),Y(2),Y(3)
               WRITE(*,*) 'A,C = ',A,C
               WRITE(*,*) 'B,D = ',B,D
               WRITE(*,*) 'AD,BC = ',AD,BC
               WRITE(*,*) 'ITI = ',ITI
            ENDIF
            AP= D/DLT
            BP=-B/DLT
            CP=-C/DLT
            DP= A/DLT
!C     CONVERTS THE PARTIAL DERIVATIVES AT THE VERTEXES OF THE
!C     TRIANGLE FOR THE U-V COORDINATE SYSTEM.
 25         AA=A*A
            ACT2=2.0*A*C
            CC=C*C
            AB=A*B
            ADBC=AD+BC
            CD=C*D
            BB=B*B
            BDT2=2.0*B*D
            DD=D*D
            DO I=1,3
               JPD=5*I
               ZU(I)=A*PD(JPD-4)+C*PD(JPD-3)
               ZV(I)=B*PD(JPD-4)+D*PD(JPD-3)
               ZUU(I)=AA*PD(JPD-2)+ACT2*PD(JPD-1)+CC*PD(JPD)
               ZUV(I)=AB*PD(JPD-2)+ADBC*PD(JPD-1)+CD*PD(JPD)
               ZVV(I)=BB*PD(JPD-2)+BDT2*PD(JPD-1)+DD*PD(JPD)
            end do
!C     CALCULATES THE COEFFICIENTS OF THE POLYNOMIAL.
            P00=Z(1)
            P10=ZU(1)
            P01=ZV(1)
            P20=0.5*ZUU(1)
            P11=ZUV(1)
            P02=0.5*ZVV(1)
            H1=Z(2)-P00-P10-P20
            H2=ZU(2)-P10-ZUU(1)
            H3=ZUU(2)-ZUU(1)
            P30= 10.0*H1-4.0*H2+0.5*H3
            P40=-15.0*H1+7.0*H2    -H3
            P50=  6.0*H1-3.0*H2+0.5*H3
            H1=Z(3)-P00-P01-P02
            H2=ZV(3)-P01-ZVV(1)
            H3=ZVV(3)-ZVV(1)
            P03= 10.0*H1-4.0*H2+0.5*H3
            P04=-15.0*H1+7.0*H2    -H3
            P05=  6.0*H1-3.0*H2+0.5*H3
            LU=SQRT(AA+CC)
            LV=SQRT(BB+DD)
            THXU=ATAN2(C,A)
            THUV=ATAN2(D,B)-THXU
            CSUV=COS(THUV)
            P41=5.0*LV*CSUV/LU*P50
            P14=5.0*LU*CSUV/LV*P05
            H1=ZV(2)-P01-P11-P41
            H2=ZUV(2)-P11-4.0*P41
            P21= 3.0*H1-H2
            P31=-2.0*H1+H2
            H1=ZU(3)-P10-P11-P14
            H2=ZUV(3)-P11-4.0*P14
            P12= 3.0*H1-H2
            P13=-2.0*H1+H2
            THUS=ATAN2(D-C,B-A)-THXU
            THSV=THUV-THUS
            AA= SIN(THSV)/LU
            BB=-COS(THSV)/LU
            CC= SIN(THUS)/LV
            DD= COS(THUS)/LV
            AC=AA*CC
            AD=AA*DD
            BC=BB*CC
            G1=AA*AC*(3.0*BC+2.0*AD)
            G2=CC*AC*(3.0*AD+2.0*BC)
            H1=-AA*AA*AA*(5.0*AA*BB*P50+(4.0*BC+AD)*P41)
     1           -CC*CC*CC*(5.0*CC*DD*P05+(4.0*AD+BC)*P14)
            H2=0.5*ZVV(2)-P02-P12
            H3=0.5*ZUU(3)-P20-P21
            P22=(G1*H2+G2*H3-H1)/(G1+G2)
            P32=H2-P22
            P23=H3-P22
            ITPV=IT0
!C     CONVERTS XII AND YII TO U-V SYSTEM.
         end if
         DX=XII-X0
         DY=YII-Y0
         U=AP*DX+BP*DY
         V=CP*DX+DP*DY
!C     EVALUATES THE POLYNOMIAL.
         P0=P00+V*(P01+V*(P02+V*(P03+V*(P04+V*P05))))
         P1=P10+V*(P11+V*(P12+V*(P13+V*P14)))
         P2=P20+V*(P21+V*(P22+V*P23))
         P3=P30+V*(P31+V*P32)
         P4=P40+V*P41
         ZII=P0+U*(P1+U*(P2+U*(P3+U*(P4+U*P50))))
!c     ZII=P0+U*(P1+U*(P2+U*(P3+U*(P4+U*P5))))
         RETURN
      else
         IL1=IT0/NTL
         IL2=IT0-IL1*NTL
         IF(IL1.EQ.IL2)      then
!C     CALCULATION OF ZII BY EXTRAPOLATION IN THE RECTANGLE.
!C     CHECKS IF THE NECESSARY COEFFICIENTS HAVE BEEN CALCULATED.
 40         IF(IT0.ne.ITPV) then
!C     LOADS COORDINATE AND PARTIAL DERIVATIVE VALUES AT THE END
!C     POINTS OF THE BORDER LINE SEGMENT.
               JIPL=3*(IL1-1)
               JPD=0
               DO I=1,2
                  JIPL=JIPL+1
                  IDP=IPL(JIPL)
                  X(I)=XD(IDP)
                  Y(I)=YD(IDP)
                  Z(I)=ZD(IDP)
                  JPDD=5*(IDP-1)
                  DO KPD=1,5
                     JPD=JPD+1
                     JPDD=JPDD+1
                     PD(JPD)=PDD(JPDD)
                  end do
               end do
!C     DETERMINES THE COEFFICIENTS FOR THE COORDINATE SYSTEM
!C     TRANSFORMATION FROM THE X-Y SYSTEM TO THE U-V SYSTEM
!C     AND VICE VERSA.
               X0=X(1)
               Y0=Y(1)
               A=Y(2)-Y(1)
               B=X(2)-X(1)
               C=-B
               D=A
               AD=A*D
               BC=B*C
               DLT=AD-BC
               AP= D/DLT
               BP=-B/DLT
               CP=-BP
               DP= AP
!C     CONVERTS THE PARTIAL DERIVATIVES AT THE END POINTS OF THE
!C     BORDER LINE SEGMENT FOR THE U-V COORDINATE SYSTEM.
               AA=A*A
               ACT2=2.0*A*C
               CC=C*C
               AB=A*B
               ADBC=AD+BC
               CD=C*D
               BB=B*B
               BDT2=2.0*B*D
               DD=D*D
               DO I=1,2
                  JDP=5*I
                  ZU(I)=A*PD(JPD-4)+C*PD(JPD-3)
                  ZV(I)=B*PD(JPD-4)+D*PD(JPD-3)
                  ZUU(I)=AA*PD(JPD-2)+ACT2*PD(JPD-1)+CC*PD(JPD)
                  ZUV(I)=AB*PD(JPD-2)+ADBC*PD(JPD-1)+CD*PD(JPD)
                  ZVV(I)=BB*PD(JPD-2)+BDT2*PD(JPD-1)+DD*PD(JPD)
               end do
!C     CALCULATES THE COEFFICIENTS OF THE POLYNOMIAL
               P00=Z(1)
               P10=ZU(1)
               P01=ZV(1)
               P20=0.5*ZUU(1)
               P11=ZUV(1)
               P02=0.5*ZVV(1)
               H1=Z(2)-P00-P01-P02
               H2=ZV(2)-P01-ZVV(1)
               H3=ZVV(2)-ZVV(1)
               P03= 10.0*H1-4.0*H2+0.5*H3
               P04=-15.0*H1+7.0*H2    -H3
               P05=  6.0*H1-3.0*H2+0.5*H3
               H1=ZU(2)-P10-P11
               H2=ZUV(2)-P11
               P12= 3.0*H1-H2
               P13=-2.0*H1+H2
               P21=0.0
               P23=-ZUU(2)+ZUU(1)
               P22=-1.5*P23
               ITPV=IT0
            end if
!C     CONVERTS XII AND YII TO U-V SYSTEM.
            DX=XII-X0
            DY=YII-Y0
            U=AP*DX+BP*DY
            V=CP*DX+DP*DY
!C     EVALUATES THE POLYNOMIAL.
            P0=P00+V*(P01+V*(P02+V*(P03+V*(P04+V*P05))))
            P1=P10+V*(P11+V*(P12+V*P13))
            P2=P20+V*(P21+V*(P22+V*P23))
            ZII=P0+U*(P1+U*P2)
            RETURN
         else
!C     CALCULATION OF ZII BY EXTRAPOLATION IN THE TRIANGLE.
!C     CHECKS IF THE NECESSARY COEFFICIENTS HAVE BEEN CALCULATED.
 60         IF(IT0.ne.ITPV)  then
!C     LOADS COORDINATE AND PARTIAL DERIVATIVE VALUES AT THE VERTEX
!C     OF THE TRIANGLE.
               JIPL=3*IL2-2
               IDP=IPL(JIPL)
               write(*,*)'before patch'
               i=1              !added to make compiler happy, may not be correct
               X(I)=XD(IDP)
               Y(I)=YD(IDP)
               Z(I)=ZD(IDP)
               JPDD=5*(IDP-1)
               DO  KPD=1,5
                  JPDD=JPDD+1
                  PD(KPD)=PDD(JPDD)
               end do
!C     CALCULATES THE COEFFICIENTS OF THE POLYNOMIAL.
               P00=Z(1)
               P10=PD(1)
               P01=PD(2)
               P20=0.5*PD(3)
               P11=PD(4)
               P02=0.5*PD(5)
               ITPV=IT0
!C     CONVERTS XII AND YII TO U-V SYSTEM.
            else
               U=XII-X(1)
               V=YII-Y(1)
!C     EVALUATES THE POLYNOMIAL
               P0=P00+V*(P01+V*P02)
               P1=P10+V*P11
               ZII=P0+U*(P1+U*P20)
            end if
            RETURN
         end if
      end if
      END


      SUBROUTINE IDPDRV(NDP,XD,YD,ZD,NCP,IPC,PD)
!C THIS SUBROUTINE ESTIMATES PARTIAL DERIVATIVES OF THE FIRST AND
!C SECOND ORDER AT THE DATA POINTS.
!C THE INPUT PARAMETERS ARE
!C     NDP = NUMBER OF DATA POINTS,
!C     XD,YD,ZD = ARRAYS OF DIMENSION NDP CONTAINING THE X, 
!C           Y, AND Z COORDINATES OF THE DATA POINTS,
!C     NCP = NUMBER OF ADDITIONAL DATA POINTS USED FOR ESTI-
!C           MATING PARTIAL DERIVATIVES AT EACH DATA POINT,
!C     IPC = INTEGER ARRAY OF DIMENSION NCP*NDP CONTAINING
!C           THE POINT NUMBERS OF NCP DATA POINTS CLOSEST TO
!C           EACH OF THE NDP DATA POINTS.
!C THE OUTPUT PARAMETER IS
!C     PD   = ARRAY OF DIMENSION 5*NDP, WHERE THE ESTIMATED
!C            ZX, ZY, ZXX, ZXY, AND ZYY VALUES AT THE DATA 
!C            POINTS ARE TO BE STORED.
!C DECLARATION STATEMENTS
      DIMENSION   XD(*),YD(*),ZD(*),IPC(*),PD(*)
      REAL        NMX,NMY,NMZ,NMXX,NMXY,NMYX,NMYY
!C PRELIMINARY PROCESSING
   10 NDP0=NDP
      NCP0=NCP
      NCPM1=NCP0-1
!c      write(*,*) 'ndp0,ndp,ncp0,ncpm1 = ',ndp0,ndp,ncp0,ncpm1
!c      write(*,*) 'xd(1),yd(1) = ',xd(1),yd(1)
!c      write(*,*) 'xd(ndp0),yd(ndp0) = ',xd(ndp0),yd(ndp0)
!C ESTIMATION OF ZX AND ZY
  20  DO 24  IP0=1,NDP0
        X0=XD(IP0)
        Y0=YD(IP0)
        Z0=ZD(IP0)
!c        write(*,*) 'idpdrv: x0,y0,z0 = ',x0,y0,z0
        NMX=0.0
        NMY=0.0
        NMZ=0.0
        JIPC0=NCP0*(IP0-1)
        DO 23  IC1=1,NCPM1
          JIPC=JIPC0+IC1
          IPI=IPC(JIPC)
!c          write(*,*) 'idpdrv:ic1,jipc0,jipc,ipi = ',ic1,jipc0,jipc,ipi
          DX1=XD(IPI)-X0
          DY1=YD(IPI)-Y0
          DZ1=ZD(IPI)-Z0
          IC2MN=IC1+1
!c          write(*,*) 'idpdrv: dx1,dy1,dz1 = ',dx1,dy1,dz1
!c          write(*,*) 'idpdrv: ic1,ic2mn = ',ic1,ic2mn
          DO 22 IC2=IC2MN,NCP0
            JIPC=JIPC0+IC2
            IPI=IPC(JIPC)
!c            write(*,*) 'idpdrv:ic2,jipc0,jipc,ipi = ',ic2,jipc0,jipc,ipi
            DX2=XD(IPI)-X0
            DY2=YD(IPI)-Y0
!c            write(*,*) 'idpdrv: dx2,dy2= ',dx2,dy2
            DNMZ=DX1*DY2-DY1*DX2
!c            write(*,*) 'idpdrv: before gt22 ,ic2,dnmz = ',ic2,dnmz
            IF(DNMZ.EQ.0.0)    GO TO 22
            DZ2=ZD(IPI)-Z0
            DNMX=DY1*DZ2-DZ1*DY2
            DNMY=DZ1*DX2-DX1*DZ2
            IF(DNMZ.GE.0.0)    GO TO 21
            DNMX=-DNMX
            DNMY=-DNMY
            DNMZ=-DNMZ
  21        NMX=NMX+DNMX
            NMY=NMY+DNMY
!c            write(*,*) 'nmz,dnmz = ',nmz,dnmz
            NMZ=NMZ+DNMZ
  22      CONTINUE
  23    CONTINUE
        JPD0=5*IP0
!c        write(*,*) 'idpdrv: ip0,jpd0,nmx,nmy,nmz= '
!c        write(*,*)  ip0,jpd0,nmx,nmy,nmz
        PD(JPD0-4)=-NMX/NMZ
        PD(JPD0-3)=-NMY/NMZ
  24  CONTINUE
!C ESTIMATION OF ZXX, ZXY, AND ZYY
   30 DO 34  IP0=1,NDP0
        JPD0=JPD0+5
        X0=XD(IP0)
        JPD0=5*IP0
        Y0=YD(IP0)
        ZX0=PD(JPD0-4)
        ZY0=PD(JPD0-3)
        NMXX=0.0
        NMXY=0.0
        NMYX=0.0
        NMYY=0.0
        NMZ =0.0
        JIPC0=NCP0*(IP0-1)
        DO 33 IC1=1,NCPM1
          JIPC=JIPC0+IC1
          IPI=IPC(JIPC)
          DX1=XD(IPI)-X0
          DY1=YD(IPI)-Y0
          JPD=5*IPI
          DZX1=PD(JPD-4)-ZX0
          DZY1=PD(JPD-3)-ZY0
          IC2MN=IC1+1
          DO 32  IC2=IC2MN,NCP0
            JIPC=JIPC0+IC2
            IPI=IPC(JIPC)
            DX2=XD(IPI)-X0
            DY2=YD(IPI)-Y0
            DNMZ =DX1*DY2 -DY1*DX2
            IF(DNMZ.EQ.0.0)    GO TO 32
            JPD=5*IPI
            DZX2=PD(JPD-4)-ZX0
            DZY2=PD(JPD-3)-ZY0
            DNMXX=DY1*DZX2-DZX1*DY2
            DNMXY=DZX1*DX2-DX1*DZX2
            DNMYX=DY1*DZY2-DZY1*DY2
            DNMYY=DZY1*DX2-DX1*DZY2
            IF(DNMZ.GE.0.0)    GO TO 31
            DNMXX=-DNMXX
            DNMXY=-DNMXY
            DNMYX=-DNMYX
            DNMYY=-DNMYY
            DNMZ=-DNMZ
  31        NMXX=NMXX+DNMXX
            NMXY=NMXY+DNMXY
            NMYX=NMYX+DNMYX
            NMYY=NMYY+DNMYY
            NMZ =NMZ +DNMZ
  32      CONTINUE
  33    CONTINUE
        PD(JPD0-2)=-NMXX/NMZ
        PD(JPD0-1)=-(NMXY+NMYX)/(2.0*NMZ)
        PD(JPD0)  =-NMYY/NMZ
  34  CONTINUE
      RETURN
      END

      
      SUBROUTINE IDCLDP(NDP,XD,YD,NCP,IPC)
!C THIS SUBROUTINE SELECTS SEVERAL DATA POINTS THAT ARE CLOSEST
!C TO EACH OF THE DATA POINT.
!C THE INPUT PARAMETERS ARE
!C     NDP = NUMBER OF DATA POINTS,
!C     XD,YD = ARRAYS OF DIMENSIONS NDP CONTAINING THE X AND Y
!C           COORDINATES OF THE DATA POINTS,
!C     NCP = NUMBER OF DATA POINTS CLOSEST TO EACH DATA
!C           POINTS.
!C THE OUTPUT PARAMETERS IS
!C     IPC = INTEGER ARRAY OF DIMENSION NCP*NDP, WHERE THE
!C           POINT NUMBER OF NCP DATA POINTS CLOSEST TO 
!C           EACH OF THE NDP DATA POINTS ARE TO BE STORED.
!C THIS SUBROUTINES ARBITRARILY SETS A RESTRICTION THAT NCP MUST
!C NOT EXCEED 25.
!C THE LUN CONSTANT IN THE DATA INITIALIZATION STATEMENT IS THE
!C LOGICAL UNIT NUMBER OF THE STANDARD OUTPUT UNIT AND IS,
!C THEREFORE, SYSTEM DEPENDENT.
!C DECLARATION STATEMENTS
      DIMENSION   XD(*),YD(*),IPC(*)
      DIMENSION   DSQ0(25),IPC0(25)
      DATA  NCPMX/25/, LUN/6/
!C STATEMENT FUNCTION
      DSQF(U1,V1,U2,V2)=(U2-U1)**2+(V2-V1)**2
!C PRELIMINARY PROCESSING
   10 NDP0=NDP
      NCP0=NCP
      IF(NDP0.LT.2)  GO TO 90
      IF(NDP0.LT.1.OR.NCP0.GT.NCPMX.OR.NCP0.GE.NDP0)    GO TO 90
!C CALCULATION
   20 DO 59  IP1=1,NDP0
!C - SELECTS NCP POINTS.
         X1=XD(IP1)
         Y1=YD(IP1)
         J1=0
         DSQMX=0.0
         DO 22  IP2=1,NDP0
           IF(IP2.EQ.IP1)   GO TO 22
           DSQI=DSQF(X1,Y1,XD(IP2),YD(IP2))
           J1=J1+1
           DSQ0(J1)=DSQI
           IPC0(J1)=IP2
           IF(DSQI.LE.DSQMX)    GO TO 21
           DSQMX=DSQI
           JMX=J1
   21      IF(J1.GE.NCP0)   GO TO 23
   22    CONTINUE
   23    IP2MN=IP2+1
         IF(IP2MN.GT.NDP0)  GO TO 30
         DO 25  IP2=IP2MN,NDP0
           IF(IP2.EQ.IP1)   GO TO 25
           DSQI=DSQF(X1,Y1,XD(IP2),YD(IP2))
           IF(DSQI.GE.DSQMX)    GO TO 25
           DSQ0(JMX)=DSQI
           IPC0(JMX)=IP2
           DSQMX=0.0
           DO 24  J1=1,NCP0
             IF (DSQ0(J1).LE.DSQMX)  GO TO 24
             DSQMX=DSQ0(J1)
             JMX=J1
   24      CONTINUE
   25    CONTINUE
!C - CHECKS IF ALL THE NCP+1 POINTS ARE COLLINEAR.
   30   IP2=IPC0(1)
        DX12=XD(IP2)-X1
        DY12=YD(IP2)-Y1
        DO 31   J3=2,NCP0
          IP3=IPC0(J3)
          DX13=XD(IP3)-X1
          DY13=YD(IP3)-Y1
          IF((DY13*DX12-DX13*DY12).NE.0.0)    GO TO 50
   31   CONTINUE
!C - SEARCHES FOR THE CLOSEST NONCOLLINEAR POINT.
   40   NCLPT=0
        DO 43  IP3=1,NDP0
          IF(IP3.EQ.IP1)       GO TO 43
          DO 41  J4=1,NCP0
            IF(IP3.EQ.IPC0(J4))       GO TO 43
   41     CONTINUE
          DX13=XD(IP3)-X1
          DY13=YD(IP3)-Y1
          IF((DY13*DX12-DX13*DY12).EQ.0.0)    GO TO 43
          DSQI=DSQF(X1,Y1,XD(IP3),YD(IP3))
          IF(NCLPT.EQ.0)       GO TO 42
          IF(DSQI.GE.DSQMN)    GO TO 43
   42     NCLPT=1
          DSQMN=DSQI
          IP3MN=IP3
   43   CONTINUE
        IF(NCLPT.EQ.0)THEN
           WRITE(*,*) 'IP1,XD(IP1),YD(IP1),X1,Y1'
           WRITE(*,*) IP1,XD(IP1),YD(IP1),X1,Y1
           WRITE(*,*) 'IP2,XD(IP2),YD(IP2) ',IP2,XD(IP2),YD(IP2)
           WRITE(*,*) 'DX12,DY12 = ',DX12,DY12
           WRITE(*,*) 'NCLPT,NDP0,NCP0 = ',NCLPT,NDP0,NCP0
           WRITE(*,*) 'IPC0 = ',(IPC0(II),II=1,NCP0)
!C           STOP
!C changed to try to continue despite co-linear points EJF 2006/1/24
           GO TO 91
        ENDIF
        DSQMX=DSQMN
        IPC0(JMX)=IP3MN
!C - REPLACES THE LOCAL ARRAY FOR THE OUTPUT ARRAY
   50   J1=(IP1-1)*NCP0
        DO 51  J2=1,NCP0
          J1=J1+1
          IPC(J1)=IPC0(J2)
   51    CONTINUE
   59  CONTINUE
       RETURN
!C ERROR EXIT
   90 WRITE (LUN,2090)
      GO TO 92
   91 WRITE (LUN,2091)
   92 WRITE (LUN,2092)   NDP0,NCP0
      IPC(1)=0
      RETURN
!C FORMAT STATEMENTS FOR ERROR MESSAGES
 2090 FORMAT(1X/41H ***   IMPROPER INPUT PARAMETER VALUE(S).)
 2091 FORMAT(1X/33H ***   ALL COLLINEAR DATA POINTS.)
 2092 FORMAT(8H   NDP =,I5,5X,5HNCP =,I5/,
     1   35H ERROR DETECTED IN ROUTINE   IDCLDP/)
      END

        
      SUBROUTINE CONVEX_HULL(XC,YC,NPTS,XCH,YCH,NPCH,ACH,PCH)

        IMPLICIT NONE

        REAL XC(*)                      !X COORDINATE OF POINTS
        REAL YC(*)                      !Y COORDINATE OF POINTS
        INTEGER NPTS                    !NUMBER OF POINTS
        REAL XCH(*)                     !X COORDINATE POINT IN CONVEX HULL
        REAL YCH(*)                     !Y COORDINATE POINT IN CONVEX HULL
        INTEGER NPCH                    !NUMBER OF POINTS IN CONVEX HULL
        REAL ACH                        !AREA OF CONVEX HULL
        REAL PCH                        !PERIMETER OF CONVEX HULL

        REAL SC,CP,TAXICABMHI,TAXICABJ,OXCH(100),OYCH(100)
        INTEGER SL,A,B,MAXI,MINI,NMAX,MAXI1,MAXI2,UI1,UI2,I1,I2,LI,J,I
        INTEGER K,MI(2),HULLINDEX,MHI,ONPCH

!C       FIRST FIND THE LOWEST AND HIGHEST POINTS IN THE DATA SET. IF MORE
!C       THAN ONE POINT IS AT THE LOWEST POINT TAKE THE LEFT MOST POINT AND
!C       IF MORE THAN ONE POINT IS AT THE HIGHEST POINT TAKE BOTH THE LEFT 
!C       MOST AND RIGHT MOST POINTS.

!C       TO SPEED UP THE SEARCH THE LOWEST AND HIGHEST POINT WILL BE FOUND
!C       SIMULTANEOUSLY NECESSITATING THE DISTINGUIHMENT OF ODD AND EVEN 
!C       NUMBER OF POINTS.       

        IF(MOD(NPTS,2) .EQ. 0)THEN
          SL = 1
          A = -1
          B = 0
        ELSE
          SL = 2
          A = -2
          B = -1
        ENDIF

        MAXI = 1
        MINI = 1
        NMAX = 1

        DO J=SL,NINT(FLOAT(NPTS)/2.)
        
          I1 = 2*J + A
          I2 = 2*J + B

          IF(YC(I1) .GT. YC(I2))THEN   !REDUCE NUMBER OF COMPARES
            LI = I2
            UI1 = I1
            UI2 = I1
          ELSEIF(YC(I1) .EQ. YC(I2))THEN
            IF(XC(I1) .LT. XC(I2))THEN
              LI = I1
            ELSE
              LI = I2
            ENDIF
            UI1 = I1
            UI2 = I2
          ELSE 
            LI = I1
            UI1 = I2
            UI2 = I2
          ENDIF

!C       FIND MINIMUM

          IF(YC(LI) .LT. YC(MINI))THEN
            MINI = LI
          ELSEIF(YC(LI) .EQ. YC(MINI))THEN
            IF(XC(LI) .LT. XC(MINI))THEN
              MINI = LI
            ENDIF
          ENDIF  

!C       FIND MAXIMA
        
          DO K=UI1,UI2

            IF(YC(K) .GT. YC(MAXI))THEN
              MAXI = K
              NMAX = 1
            ELSEIF(YC(K) .EQ. YC(MAXI))THEN
              IF(NMAX .EQ. 1)THEN
                IF(XC(K) .LT. XC(MAXI))THEN
                   MAXI1 = K
                   MAXI2 = MAXI
                   NMAX = 2
                ELSE
                   MAXI1 = MAXI
                   MAXI2 = K
                   NMAX = 2
                ENDIF
              ELSE
                IF(XC(K) .LT. XC(MAXI1))THEN
                   MAXI1 = K
                ELSEIF(XC(K) .GT. XC(MAXI2))THEN
                   MAXI2 = K
                ENDIF
              ENDIF
            ENDIF

          ENDDO  !K LOOP    
               
        ENDDO    !J LOOP            

!C       FIND THE POINTS  IN THE CONVEX HULL USING JARVIS MARCH ALGORITHM

        IF(NMAX .EQ. 1)THEN
          MAXI1 = MAXI
          MAXI2 = MAXI
        ENDIF
        MI(1) = MAXI1
        MI(2) = MAXI2  
        NPCH = 1
        ONPCH = 0
        XCH(1) = XC(MINI)
        YCH(1) = YC(MINI)

        DO K=1,2   !K=2 FINDS COUNTER CLOCKWISE PART OF HULL 1 CLOCKWISE

          SC = -2*K + 3
          HULLINDEX = MINI

          DO WHILE(YC(HULLINDEX) .LT. YC(MI(K)))
 
             MHI = HULLINDEX
             DO J=1,NPTS        !FIND MINIMAL POINT IN POLAR ANGLE
        
               IF(J .NE. HULLINDEX)THEN
              
                 CP = SC*((YC(MHI) - YC(HULLINDEX))*(XC(J) - XC(HULLINDEX)) -(YC(J) - YC(HULLINDEX))*(XC(MHI) - XC(HULLINDEX)))
           
                 IF(CP .LT. 0)THEN        
                    MHI = J
                 ELSEIF(CP .EQ. 0)THEN
                    TAXICABMHI = ABS((YC(MHI) - YC(HULLINDEX))) + ABS((XC(MHI) - XC(HULLINDEX)))
                    TAXICABJ = ABS((YC(J) - YC(HULLINDEX))) + ABS((XC(J) - XC(HULLINDEX)))
                    IF(TAXICABJ .GT. TAXICABMHI)THEN
                      MHI = J
                    ENDIF
                 ENDIF

               ENDIF        !NOT THE VERTEX ITSELF

             ENDDO

!C       RECORD NEW MEMBER OF CONVEX HULL AND ITS POSITION

            HULLINDEX = MHI
            IF(K .EQ. 1)THEN
              NPCH = NPCH + 1
              XCH(NPCH) = XC(MHI)
              YCH(NPCH) = YC(MHI)
            ELSE
              ONPCH = ONPCH + 1
              OXCH(ONPCH) = XC(MHI)
              OYCH(ONPCH) = YC(MHI)
            ENDIF

          ENDDO !WHILE LOOP

        ENDDO   !K LOOP

        IF(NMAX .EQ. 1)THEN
          ONPCH = ONPCH - 1
        ENDIF 

        K = 0
        DO J=NPCH+1,NPCH+ONPCH
           XCH(J) = OXCH(ONPCH-K)
           YCH(J) = OYCH(ONPCH-K)
           K = K + 1 
        ENDDO
        NPCH = NPCH + ONPCH 

!C       FINALLY COMPUTE THE AREA IN THE CONVEX HULL  

        ACH = 0
        PCH = 0
        HULLINDEX = 1 
        DO J=2,NPCH-1
 
          CP = (YCH(J) - YCH(HULLINDEX))*(XCH(J+1) - XCH(HULLINDEX)) -
     +         (YCH(J+1) - YCH(HULLINDEX))*(XCH(J) - XCH(HULLINDEX))

          ACH = ACH + ABS(CP)

          PCH = PCH + SQRT((XCH(J) - XCH(J-1))**2 + (YCH(J) - YCH(J-1))**2)

        ENDDO

        ACH = .5*ACH

        J = NPCH
        PCH = PCH + SQRT((XCH(J) - XCH(J-1))**2 + (YCH(J) - YCH(J-1))**2)
     +            + SQRT((XCH(J) - XCH(1))**2 + (YCH(J) - YCH(1))**2) 

        END


        SUBROUTINE IDGRID(XD, YD, NT, IPT, NL, IPL, NXI, NYI, XI, YI,
     *  NGP, IGP)
!C THIS SUBROUTINE ORGANIZES GRID POINTS FOR SURFACE FITTING BY
!C SORTING THEM IN ASCENDING ORDER OF TRIANGLE NUMBERS AND OF THE 
!C BORDER LINE SEGMENT NUMBER.
!C THE INPUT PARAMETERS ARE
!C      XD,YD = ARRAYS OF DIMENSION NDP CONTAINING THE X AND Y
!C             COORDINATES OF THE DATA POINTS, WHERE NDP IS THE 
!C             NUMBER OF THE DATA POINTS
!C      NT   = NUMBER OF TRIANGLES.
!C      IPT  = INTEGER ARRAY OF DIMENSION 3*NT CONTAINING THE
!C             POINT NUMBERS OF THE VERTEXES OF THE TRIANGLES,
!C      NL   = NUMBER OF BORDER LINE SEGMENTS,
!C      IPL  = INTEGER ARRAY OF DIMENSION 3*NL CONTAINING THE 
!C             POINT NUMBERS OF THE END POINTS OF THE BORDER
!C             LINE SEGMENTS AND THEIR RESPECTIVE TRIANGLE
!C             NUMBERS,
!C      NXI  = NUMBER OF GRID POINTS IN THE X COORDINATE,
!C      NYI  = NUMBER OF GRID POINTS IN THE Y COORDINATE,
!C      XI,YI = ARRAYS OF DIMENSION NXI AND NYI CONTAINING
!C            THE X AND Y COORDINATES OF THE GRID POINTS,
!C            RESPECTIVELY.
!C THE OUTPUT PARAMETERS ARE
!C      NGP  = INTEGER ARRAY OF DIMENSION 2*(NT+2*NL) WHERE THE
!C             NUMBER OF GRID POINTS THAT BELONG TO EACH OF THE
!C             TRIANGLES OR OF THE BORDER LINE SEGMENTS ARE TO
!C             BE STORED,
!C      IGP  = INTEGER ARRAY OF DIMENSION NXI*NYI WHERE THE
!C             GRID POINTS ARE TO BE STORED IN ASCENDING
!C             ORDER OF THE TRIANGLE NUMBER AND THE BORDER LINE 
!C             SEGMENTS NUMBER.
!C DECLARATION STATEMENTS
      DIMENSION XD(*), YD(*), IPT(*), IPL(*), XI(*),
     *  YI(*), NGP(*), IGP(*)
!C STATEMENT FUNCTIONS
      SIDE(U1,V1,U2,V2,U3,V3) = (U1-U3)*(V2-V3) - (V1-V3)*(U2-U3)
      SPDT(U1,V1,U2,V2,U3,V3) = (U1-U2)*(U3-U2) + (V1-V2)*(V3-V2)
!C PRELIMINARY PROCESSING
      NT0 = NT
      NL0 = NL
      NXI0 = NXI
      NYI0 = NYI
      NXINYI = NXI0*NYI0
      XIMN = AMIN1(XI(1),XI(NXI0))
      XIMX = AMAX1(XI(1),XI(NXI0))
      YIMN = AMIN1(YI(1),YI(NYI0))
      YIMX = AMAX1(YI(1),YI(NYI0))
!C DETERMINES GRID POINTS INSIDE THE DATA AREA.
       JNGP0 = 0
       JNGP1 = 2*(NT0+2*NL0) + 1
       JIGP0 = 0
       JIGP1 = NXINYI + 1
       DO 160 IT0=1,NT0
         NGP0 = 0
         NGP1 = 0
         IT0T3 = IT0*3
         IP1 = IPT(IT0T3-2)
         IP2 = IPT(IT0T3-1)
         IP3 = IPT(IT0T3)
         X1 = XD(IP1)
         Y1 = YD(IP1)
         X2 = XD(IP2)
         Y2 = YD(IP2)
         X3 = XD(IP3)
         Y3 = YD(IP3)
         XMN = AMIN1(X1,X2,X3)
         XMX = AMAX1(X1,X2,X3)
         YMN = AMIN1(Y1,Y2,Y3)
         YMX = AMAX1(Y1,Y2,Y3)
         INSD = 0
         DO 20 IXI=1,NXI0
            IF (XI(IXI).GE.XMN .AND. XI(IXI).LE.XMX) GO TO 10
            IF (INSD.EQ.0) GO TO 20
            IXIMX = IXI - 1
            GO TO 30
   10       IF (INSD.EQ.1) GO TO 20
            INSD = 1
            IXIMN = IXI
   20     CONTINUE
          IF (INSD.EQ.0) GO TO 150
          IXIMX = NXI0
   30     DO 140 IYI=1,NYI0
            YII = YI(IYI)
            IF (YII.LT.YMN .OR. YII.GT.YMX)  GO TO 140
            DO 130 IXI=IXIMN,IXIMX
            XII = XI(IXI)
            L = 0
            IF (SIDE(X1,Y1,X2,Y2,XII,YII))  130, 40, 50
   40       L = 1
   50       IF (SIDE(X2,Y2,X3,Y3,XII,YII))  130, 60, 70
   60       L = 1
   70       IF (SIDE(X3,Y3,X1,Y1,XII,YII))  130, 80, 90
   80       L = 1
   90       IZI = NXI0*(IYI-1) + IXI
            IF (L.EQ.1) GO TO 100
            NGP0 = NGP0 + 1
            JIGP0 = JIGP0 + 1
            IGP(JIGP0) = IZI
            GO TO 130
  100       IF (JIGP1.GT.NXINYI)  GO TO 120
            DO 110 JIGP1I=JIGP1,NXINYI
              IF (IZI.EQ.IGP(JIGP1I))  GO TO 130
  110       CONTINUE
  120       NGP1 = NGP1 + 1
            JIGP1 = JIGP1 - 1
            IGP(JIGP1) = IZI
  130     CONTINUE
  140   CONTINUE
  150   JNGP0 = JNGP0 + 1
        NGP(JNGP0) = NGP0
        JNGP1 = JNGP1 - 1
        NGP(JNGP1) = NGP1
  160 CONTINUE
!C DETERMINES GRID POINTS OUTSIDE THE DATA AREA.
!C - IN SEMI-INFINITE RECTANGULAR AREA.
      DO 450 IL0=1,NL0
         NGP0 = 0
         NGP1 = 0
         IL0T3 = IL0*3
         IP1 = IPL(IL0T3-2)
         IP2 = IPL(IL0T3-1)
         X1 = XD(IP1)
         Y1 = YD(IP1)
         X2 = XD(IP2)
         Y2 = YD(IP2)
         XMN = XIMN
         XMX = XIMX
         YMN = YIMN
         YMX = YIMX
         IF (Y2.GE.Y1) XMN = AMIN1(X1,X2)
         IF (Y2.LE.Y1) XMX = AMAX1(X1,X2)
         IF (X2.LE.X1) YMN = AMIN1(Y1,Y2)
         IF (X2.GE.X1) YMX = AMAX1(Y1,Y2)
         INSD = 0
         DO 180 IXI=1,NXI0
           IF (XI(IXI).GE.XMN .AND. XI(IXI).LE.XMX) GO TO 170
           IF (INSD.EQ.0) GO TO 180
           IXIMX = IXI - 1
           GO TO 190
  170      IF (INSD.EQ.1) GO TO 180
           INSD = 1
           IXIMN = IXI
  180    CONTINUE
         IF (INSD.EQ.0) GO TO 310
         IXIMX = NXI0
  190    DO 300 IYI=1,NYI0
           YII = YI(IYI)
           IF (YII.LT.YMN .OR. YII.GT.YMX) GO TO 300
           DO 290 IXI=IXIMN,IXIMX
             XII = XI(IXI)
             L = 0 
            IF (SIDE(X1,Y1,X2,Y2,XII,YII))  210, 200, 290
  200       L = 1
  210       IF (SPDT(X2,Y2,X1,Y1,XII,YII))  290, 220, 230
  220       L = 1
  230       IF (SPDT(X1,Y1,X2,Y2,XII,YII))  290, 240, 250
  240       L = 1
  250       IZI = NXI0*(IYI-1) + IXI
            IF (L.EQ.1) GO TO 260
            NGP0 = NGP0 + 1
            JIGP0 = JIGP0 + 1
            IGP(JIGP0) = IZI
            GO TO 290
  260       IF (JIGP1.GT.NXINYI) GO TO 280
            DO 270 JIGP1I=JIGP1,NXINYI
              IF (IZI.EQ.IGP(JIGP1I)) GO TO 290
  270       CONTINUE
  280       NGP1 = NGP1 + 1
            JIGP1 = JIGP1 - 1
            IGP(JIGP1) = IZI
  290     CONTINUE
  300   CONTINUE
  310   JNGP0 = JNGP0 + 1
        NGP(JNGP0) = NGP0
        JNGP1 = JNGP1 - 1
        NGP(JNGP1) = NGP1
!C - IN SEMI-INFINITE TRIANGULAR AREA.
        NGP0 = 0
        NGP1 = 0
        ILP1 = MOD(IL0,NL0) + 1
        ILP1T3 = ILP1*3
        IP3 = IPL(ILP1T3-1)
        X3 = XD(IP3)
        Y3 = YD(IP3)
        XMN = XIMN
        XMX = XIMX
        YMN = YIMN
        YMX = YIMX
        IF (Y3.GE.Y2 .AND. Y2.GE.Y1) XMN = X2
        IF (Y3.LE.Y2 .AND. Y2.LE.Y1) XMX = X2
        IF (X3.LE.X2 .AND. X2.LE.X1) YMN = Y2
        IF (X3.GE.X2 .AND. X2.GE.X1) YMX = Y2
        INSD = 0
        DO 330 IXI=1,NXI0
          IF (XI(IXI).GE.XMN .AND. XI(IXI).LE.XMX) GO TO 320
          IF (INSD.EQ.0) GO TO 330
          IXIMX = IXI - 1
          GO TO 340
  320     IF (INSD.EQ.1) GO TO 330
          INSD = 1
          IXIMN = IXI
  330   CONTINUE
        IF (INSD.EQ.0) GO TO 440
        IXIMX = NXI0
  340   DO 430 IYI=1,NYI0
          YII = YI(IYI)
          IF (YII.LT.YMN .OR. YII.GT.YMX)  GO TO 430
          DO 420 IXI=IXIMN,IXIMX
            XII = XI(IXI)
            L = 0
            IF (SPDT(X1,Y1,X2,Y2,XII,YII))  360, 350, 420
  350       L = 1
  360       IF (SPDT(X3,Y3,X2,Y2,XII,YII))  380, 370, 420
  370       L = 1
  380       IZI = NXI0*(IYI-1) + IXI
            IF (L.EQ.1) GO TO 390
            NGP0 = NGP0 + 1
            JIGP0 = JIGP0 + 1
            IGP(JIGP0) = IZI
            GO TO 420
  390       IF (JIGP1.GT.NXINYI)  GO TO 410
            DO 400 JIGP1I=JIGP1,NXINYI
              IF (IZI.EQ.IGP(JIGP1I))  GO TO 420
  400       CONTINUE
  410       NGP1 = NGP1 + 1
            JIGP1 = JIGP1 - 1
            IGP(JIGP1) = IZI
  420     CONTINUE
  430   CONTINUE
  440   JNGP0 = JNGP0 + 1
        NGP(JNGP0) = NGP0
        JNGP1 = JNGP1 - 1
        NGP(JNGP1) = NGP1
  450 CONTINUE
      RETURN
      END


