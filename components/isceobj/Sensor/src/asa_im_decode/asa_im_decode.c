
/***********************************************************************************************************************************

  asa_im_decode.c

  Decodes Envisat ASAR Image Mode Level 0 data

  compiled on a Sun and SGI with command gcc -O2 asa_im_decode.c -o asa_im_decode

  v1.0, Feb/Mar 2004, Sean M. Buckley
  v1.05, Mar 25, 2004, Sean M. Buckley, now fills missing lines with zeroes in float mode and 0.*127.5+127.5 + .5 = 128 for byte mode
  v1.1, 17 Feb 2005, Zhenhong Li:
  1. This program can run on a little endian machine as well as a big endian machine!
  2. On Linux, compiled with command gcc -O2 asa_im_decode.c -o asa_im_decode
  3. On Unix, compiled with command gcc -O2 asa_im_decode.c -o asa_im_decode
  v1.1.1, 8 Nov 2005, Vikas Gudipati, now runs correctly on 64-bit compilers.

 ***********************************************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * The <code>EPR_DataTypeId</code> enumeration lists all possible data
 * types for field elements in ENVISAT dataset records.
 */
// added by Z. Li on 16/02/2005
// extracted from ESA BEAM epr_api package
enum EPR_DataTypeId
{
    /** The ID for unknown types. */
    e_tid_unknown = 0,
    /** An array of unsigned 8-bit integers, C type is <code>uchar*</code> */
    e_tid_uchar   = 1,
    /** An array of signed 8-bit integers, C type is <code>char*</code> */
    e_tid_char    = 2,
    /** An array of unsigned 16-bit integers, C type is <code>ushort*</code> */
    e_tid_ushort  = 3,
    /** An array of signed 16-bit integers, C type is <code>short*</code> */
    e_tid_short   = 4,
    /** An array of unsigned 32-bit integers, C type is <code>ulong*</code> */
    e_tid_ulong   = 5,
    /** An array of signed 32-bit integers, C type is <code>long*</code> */
    e_tid_long    = 6,
    /** An array of 32-bit floating point numbers, C type is <code>float*</code> */
    e_tid_float   = 7,
    /** An array of 64-bit floating point numbers, C type is <code>double*</code> */
    e_tid_double  = 8,
    /** A zero-terminated ASCII string, C type is <code>char*</code> */
    e_tid_string  = 11,
    /** An array of unsigned character, C type is <code>uchar*</code> */
    e_tid_spare  = 13,
    /** A time (MJD) structure, C type is <code>EPR_Time</code> */
    e_tid_time    = 21
};

/* structures */

struct mphStruct {
    char product[ 62 ];
    char procStage[ 1 ];
    char refDoc[ 23 ];
    char spare1[ 40 ];
    char acquisitionStation[ 20 ];
    char procCenter[ 6 ];
    char procTime[ 27 ];
    char softwareVer[ 14 ];
    char spare2[ 40 ];
    char sensingStart[ 27 ];
    char sensingStop[ 27 ];
    char spare3[ 40 ];
    char phase[ 1 ];
    int cycle;
    int relOrbit;
    int absOrbit;
    char stateVectorTime[ 27 ];
    double deltaUt1;
    double xPosition;
    double yPosition;
    double zPosition;
    double xVelocity;
    double yVelocity;
    double zVelocity;
    char vectorSource[ 2 ];
    char spare4[ 40 ];
    char utcSbtTime[ 27 ];
    unsigned int satBinaryTime;
    unsigned int clockStep;
    char spare5[ 32 ];
    char leapUtc[ 27 ];
    int leapSign;
    int leapErr;
    char spare6[ 40];
    int productErr;
    int totSize;
    int sphSize;
    int numDsd;
    int dsdSize;
    int numDataSets;
    char spare7[ 40 ];
};

struct dsdStruct {
    char dsName[ 28 ];
    char dsType[ 1 ];
    char filename[ 62 ];
    int dsOffset;
    int dsSize;
    int numDsr;
    int dsrSize;
};

struct sphStruct {
    char sphDescriptor[ 28 ];
    double startLat;
    double startLon;
    double stopLat;
    double stopLon;
    double satTrack;
    char spare1[ 50 ];
    int ispErrorsSignificant;
    int missingIspsSignificant;
    int ispDiscardedSignificant;
    int rsSignificant;
    char spare2[ 50 ];
    int numErrorIsps;
    double errorIspsThresh;
    int numMissingIsps;
    double missingIspsThresh;
    int numDiscardedIsps;
    double discardedIspsThresh;
    int numRsIsps;
    double rsThresh;
    char spare3[ 100 ];
    char txRxPolar[ 5 ];
    char swath[ 3 ];
    char spare4[ 41 ];
    struct dsdStruct dsd[ 4 ];
};

struct sphAuxStruct {
    char sphDescriptor[ 28 ];
    char spare1[ 51 ];
    struct dsdStruct dsd[ 1 ];
};

struct dsrTimeStruct {
    int days;
    int seconds;
    int microseconds;
};

struct calPulseStruct {
    float nomAmplitude[ 32 ];
    float nomPhase[ 32 ];
};

struct nomPulseStruct {
    float pulseAmpCoeff[ 4 ];
    float pulsePhsCoeff[ 4 ];
    float pulseDuration;
};

struct dataConfigStruct {
    char echoCompMethod[ 4 ];
    char echoCompRatio[ 3 ];
    char echoResampFlag[ 1 ];
    char initCalCompMethod[ 4 ];
    char initCalCompRatio[ 3 ];
    char initCalResampFlag[ 1 ];
    char perCalCompMethod[ 4 ];
    char perCalCompRatio[ 3 ];
    char perCalResampFlag[ 1 ];
    char noiseCompMethod[ 4 ];
    char noiseCompRatio[ 3 ];
    char noiseResampFlag[ 1 ];
};

struct swathConfigStruct {
    unsigned short numSampWindowsEcho[ 7 ];
    unsigned short numSampWindowsInitCal[ 7 ];
    unsigned short numSampWindowsPerCal[ 7 ];
    unsigned short numSampWindowsNoise[ 7 ];
    float resampleFactor[ 7 ];
};

struct swathIdStruct {
    unsigned short swathNum[ 7 ];
    unsigned short beamSetNum[ 7 ];
};

struct timelineStruct {
    unsigned short swathNums[ 7 ];
    unsigned short mValues[ 7 ];
    unsigned short rValues[ 7 ];
    unsigned short gValues[ 7 ];
};

/* problems begin with field 132 - check the double statement */
struct testStruct {
    float operatingTemp;
    float rxGainDroopCoeffSmb[ 16 ]; /* this needs to be converted to a double array of eight elements */
    //double rxGainDroopCoeffSmb[ 8 ]; /* Something wrong here, why?*/
};

struct insGadsStruct {     /* see pages 455-477 for the 142 fields associated with this gads - got length of 121712 bytes */
    struct dsrTimeStruct dsrTime;
    unsigned int dsrLength;
    float radarFrequency;
    float sampRate;
    float offsetFreq;
    struct calPulseStruct calPulseIm0TxH1;
    struct calPulseStruct calPulseIm0TxV1;
    struct calPulseStruct calPulseIm0TxH1a;
    struct calPulseStruct calPulseIm0TxV1a;
    struct calPulseStruct calPulseIm0RxH2;
    struct calPulseStruct calPulseIm0RxV2;
    struct calPulseStruct calPulseIm0H3;
    struct calPulseStruct calPulseIm0V3;
    struct calPulseStruct calPulseImTxH1[ 7 ];
    struct calPulseStruct calPulseImTxV1[ 7 ];
    struct calPulseStruct calPulseImTxH1a[ 7 ];
    struct calPulseStruct calPulseImTxV1a[ 7 ];
    struct calPulseStruct calPulseImRxH2[ 7 ];
    struct calPulseStruct calPulseImRxV2[ 7 ];
    struct calPulseStruct calPulseImH3[ 7 ];
    struct calPulseStruct calPulseImV3[ 7 ];
    struct calPulseStruct calPulseApTxH1[ 7 ];
    struct calPulseStruct calPulseApTxV1[ 7 ];
    struct calPulseStruct calPulseApTxH1a[ 7 ];
    struct calPulseStruct calPulseApTxV1a[ 7 ];
    struct calPulseStruct calPulseApRxH2[ 7 ];
    struct calPulseStruct calPulseApRxV2[ 7 ];
    struct calPulseStruct calPulseApH3[ 7 ];
    struct calPulseStruct calPulseApV3[ 7 ];
    struct calPulseStruct calPulseWvTxH1[ 7 ];
    struct calPulseStruct calPulseWvTxV1[ 7 ];
    struct calPulseStruct calPulseWvTxH1a[ 7 ];
    struct calPulseStruct calPulseWvTxV1a[ 7 ];
    struct calPulseStruct calPulseWvRxH2[ 7 ];
    struct calPulseStruct calPulseWvRxV2[ 7 ];
    struct calPulseStruct calPulseWvH3[ 7 ];
    struct calPulseStruct calPulseWvV3[ 7 ];
    struct calPulseStruct calPulseWsTxH1[ 5 ];
    struct calPulseStruct calPulseWsTxV1[ 5 ];
    struct calPulseStruct calPulseWsTxH1a[ 5 ];
    struct calPulseStruct calPulseWsTxV1a[ 5 ];
    struct calPulseStruct calPulseWsRxH2[ 5 ];
    struct calPulseStruct calPulseWsRxV2[ 5 ];
    struct calPulseStruct calPulseWsH3[ 5 ];
    struct calPulseStruct calPulseWsV3[ 5 ];
    struct calPulseStruct calPulseGmTxH1[ 5 ];
    struct calPulseStruct calPulseGmTxV1[ 5 ];
    struct calPulseStruct calPulseGmTxH1a[ 5 ];
    struct calPulseStruct calPulseGmTxV1a[ 5 ];
    struct calPulseStruct calPulseGmRxH2[ 5 ];
    struct calPulseStruct calPulseGmRxV2[ 5 ];
    struct calPulseStruct calPulseGmH3[ 5 ];
    struct calPulseStruct calPulseGmV3[ 5 ];
    struct nomPulseStruct nomPulseIm[ 7 ];
    struct nomPulseStruct nomPulseAp[ 7 ];
    struct nomPulseStruct nomPulseWv[ 7 ];
    struct nomPulseStruct nomPulseWs[ 5 ];
    struct nomPulseStruct nomPulseGm[ 5 ];
    float azPatternIs1[ 101 ];
    float azPatternIs2[ 101 ];
    float azPatternIs3Ss2[ 101 ];
    float azPatternIs4Ss3[ 101 ];
    float azPatternIs5Ss4[ 101 ];
    float azPatternIs6Ss5[ 101 ];
    float azPatternIs7[ 101 ];
    float azPatternSs1[ 101 ];
    float rangeGateBias;
    float rangeGateBiasGm;
    float adcLutI[ 255 ];
    float adcLutQ[ 255 ];
    char spare1[ 648 ];
    float full8LutI[ 256 ];
    float full8LutQ[ 256 ];
    float fbaq4LutI[ 4096 ];
    float fbaq3LutI[ 2048 ];
    float fbaq2LutI[ 1024 ];
    float fbaq4LutQ[ 4096 ];
    float fbaq3LutQ[ 2048 ];
    float fbaq2LutQ[ 1024 ];
    float fbaq4NoAdc[ 4096 ];
    float fbaq3NoAdc[ 2048 ];
    float fbaq2NoAdc[ 1024 ];
    float smLutI[ 16 ];
    float smLutQ[ 16 ];
    struct dataConfigStruct dataConfigIm;
    struct dataConfigStruct dataConfigAp;
    struct dataConfigStruct dataConfigWs;
    struct dataConfigStruct dataConfigGm;
    struct dataConfigStruct dataConfigWv;
    struct swathConfigStruct swathConfigIm;
    struct swathConfigStruct swathConfigAp;
    struct swathConfigStruct swathConfigWs;
    struct swathConfigStruct swathConfigGm;
    struct swathConfigStruct swathConfigWv;
    unsigned short perCalWindowsEc;
    unsigned short perCalWindowsMs;
    struct swathIdStruct swathIdIm;
    struct swathIdStruct swathIdAp;
    struct swathIdStruct swathIdWs;
    struct swathIdStruct swathIdGm;
    struct swathIdStruct swathIdWv;
    unsigned short initCalBeamSetWv;
    unsigned short beamSetEc;
    unsigned short beamSetMs;
    unsigned short calSeq[ 32 ];
    struct timelineStruct timelineIm;
    struct timelineStruct timelineAp;
    struct timelineStruct timelineWs;
    struct timelineStruct timelineGm;
    struct timelineStruct timelineWv;
    unsigned short mEc;
    char spare2[ 44 ];
    float refElevAngleIs1;
    float refElevAngleIs2;
    float refElevAngleIs3Ss2;
    float refElevAngleIs4Ss3;
    float refElevAngleIs5Ss4;
    float refElevAngleIs6Ss5;
    float refElevAngleIs7;
    float refElevAngleSs1;
    char spare3[ 64 ];
    float calLoopRefIs1[ 128 ];
    float calLoopRefIs2[ 128 ];
    float calLoopRefIs3Ss2[ 128 ];
    float calLoopRefIs4Ss3[ 128 ];
    float calLoopRefIs5Ss4[ 128 ];
    float calLoopRefIs6Ss5[ 128 ];
    float calLoopRefIs7[ 128 ];
    float calLoopRefSs1[ 128 ];
    char spare4[ 5120 ];
    struct testStruct im;
    struct testStruct ap;
    struct testStruct ws;
    struct testStruct gm;
    struct testStruct wv;
    float swstCalP2;
    char spare5[ 72 ];
};

typedef struct {
    int samples;
    int lines;
}ImageOutput ;

// added by Z. Li on 16/02/2005
typedef enum   EPR_DataTypeId      EPR_EDataTypeId;
typedef int            boolean;
typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned long  ulong;



/* function prototypes */

struct mphStruct readMph( const char *mphPtr, const int printMphIfZero );
struct sphStruct readSph( const char *sphPtr, const int printSphIfZero, const struct mphStruct mph );
struct sphAuxStruct readSphAux( const char *sphPtr, const int printSphIfZero, const struct mphStruct mph );
struct insGadsStruct readInsGads( const char *gadsPtr, const int printInsGadsIfZero );
void printInsGads( const struct insGadsStruct );


// added by Z. Li on 16/02/2005
int is_bigendian();
void byte_swap_short(short *buffer, uint number_of_swaps);
void byte_swap_ushort(ushort* buffer, uint number_of_swaps);
void byte_swap_long(long *buffer, uint number_of_swaps);
void byte_swap_ulong(ulong* buffer, uint number_of_swaps);
void byte_swap_float(float* buffer, uint number_of_swaps);

/* new byte_swap_int type added below*/
void byte_swap_int(int *buffer, uint number_of_swaps);
void byte_swap_uint(uint *buffer, uint number_of_swaps);
/* new byte_swap_uint type added above*/

void swap_endian_order(EPR_EDataTypeId data_type_id, void* elems, uint num_elems);
void byte_swap_InsGads( struct insGadsStruct* InsGads );



/**********************************************************************************************************************************/

void asa_im_decode(char *imFileName, char *insFileName, char *outFileName, char * timesOutFileName, int outType, unsigned short windowStartTimeCodeword0, int daysToRemove, int *samples, int *lines)
{

    /* variable definitions */

    FILE *imFilePtr;
    FILE *outFilePtr;
    FILE *blockIdFilePtr;
    FILE *insFilePtr;
    FILE * dsrfp;
    //char imFileName[ 200 ];
    //char outFileName[ 200 ];
    char blockIdFileName[ 200 ] = "blockId";
    //char insFileName[ 200 ];
    char *mphPtr;
    char *sphPtr;
    char *gadsPtr;

    unsigned char onBoardTimeLSB;
    unsigned char auxTxMonitorLevel;
    unsigned char mdsrBlockId[ 200 ];
    unsigned char mdsrCheck[ 63 ];
    unsigned char beamAdjDeltaCodeword;
    unsigned char compressionRatio;
    unsigned char echoFlag;
    unsigned char noiseFlag;
    unsigned char calFlag;
    unsigned char calType;
    unsigned char spare;
    unsigned char antennaBeamSetNumber;
    unsigned char TxPolarization;
    unsigned char RxPolarization;
    unsigned char calibrationRowNumber;
    unsigned char chirpPulseBandwidthCodeword;
    unsigned char mdsrLineChar[ 20000 ];

    int printImMphIfZero      = 1;
    int printImSphIfZero      = 1;
    int printImMdsrIfZero  = 1;
    int printInsMphIfZero  = 1;
    int printInsSphIfZero  = 1;
    int printInsGadsIfZero = 1;
    int printBlockIdIfZero = 1;
    int noAdcIfZero     = 1;
    int firstTimeEqualsZero   = 0;
    int mphSize         = 1247;  /* fixed size */
    int outSamples      = 0;
    int outLines        = 0;
    int sampleShift     = 0;
    int bytesRead    = 0;
    int nonOverlappingLineIfZero = 0;
    //int outType       = 4;
    int i;
    int ii;
    int j;
    int k;
    int m;
    int n;
    int numFiles = 1;
    int mdsrDsrTimeDays;
    int mdsrDsrTimeSeconds;
    int mdsrDsrTimeMicroseconds;
    int mdsrGsrtTimeDays;
    int mdsrGsrtTimeSeconds;
    int mdsrGsrtTimeMicroseconds;
    int mdsrLineInt;

    unsigned int modePacketCount;
    unsigned int modePacketCountOld;
    unsigned int onBoardTimeIntegerSeconds = 0;

    short upConverterLevel;
    short downConverterLevel;

    unsigned short resamplingFactor;
    unsigned short onBoardTimeMSW;
    unsigned short onBoardTimeLSW;
    unsigned short mdsrIspLength;
    unsigned short mdsrCrcErrs;
    unsigned short mdsrRsErrs;
    unsigned short mdsrSpare1;
    unsigned short mdsrPacketIdentification;
    unsigned short mdsrPacketSequenceControl;
    unsigned short mdsrPacketLength;
    unsigned short mdsrPacketDataHeader[ 15 ];
    unsigned short onBoardTimeFractionalSecondsInt = 0;
    unsigned short TxPulseLengthCodeword;
    unsigned short priCodeword;
    unsigned short priCodewordOld;
    unsigned short priCodewordOldOld;
    unsigned short windowStartTimeCodeword;
    //unsigned short windowStartTimeCodeword0;
    unsigned short windowStartTimeCodewordOld;
    unsigned short windowStartTimeCodewordOldOld;
    unsigned short windowLengthCodeword;
    unsigned short dataFieldHeaderLength;
    unsigned short modeID;
    unsigned short cyclePacketCount;

    float LUTi[ 4096 ];
    float LUTq[ 4096 ];
    float mdsrLine[ 20000 ];
    double dateAux[2];
    double onBoardTimeFractionalSeconds;
    double TxPulseLength;
    double beamAdjDelta;
    double chirpPulseBandwidth;
    double c = 299792458.;
    double timeCode;
    double pri;
    double windowStartTime;
    double windowLength;

    struct mphStruct mph;
    struct mphStruct mphIns;
    struct sphStruct sph;
    struct sphAuxStruct sphIns;
    struct insGadsStruct insGads;

    int is_littlendian;

    /* usage note

    //printf( "\n*** asa_im_decode v1.0 by smb ***\n\n" );
    printf( "\n*** asa_im_decode v1.1 by smb ***\n\n" );

    if ( (argc-1) < 5 ) {
    printf( "Decodes Envisat ASAR Image Mode Level 0 data.\n\n" );
    printf( "Usage: asa_im_decode <asa_im> <asa_ins> <out> <outType> <swst>\n\n" );
    printf( "       asa_im      input image file(s) (multiple files if merging along-track)\n" );
    printf( "       asa_ins     input auxilary instrument characterization data file\n" );
    printf( "       out         output raw data file\n" );
    printf( "       outType     output file type (1=byte,4=float)\n" );
    printf( "       swst        window start time codeword to which to set all lines (0=use first line start time)\n\n" );
    printf( "Notes:\n\n" );
    printf( "out is a complex file with no headers (byte/float I1, byte/float Q1, byte/float I2, byte/float Q2, ...)\n\n" );
    printf( "if outType is byte, then the decoded floats are multiplied by 127.5, shifted by 127.5, rounded to the nearest integer and limited to the range 0-255\n\n" );
    printf( "starting range computed as (rank*pri+windowStartTime)*c/2 where rank is the number of pri between transmitted pulse and return echo\n\n" );
    printf( "calibration/noise lines are replaced with previous echo data line\n\n" );
    printf( "missing lines within a data set and between adjacent along-track data sets are filled with zeroes in float mode and 0.*127.5+127.5 + .5 = 128 for byte mode\n\n" );
    printf( "auxilary data files can be found at http://envisat.esa.int/services/auxiliary_data/asar/\n\n" );
    printf( "Envisat ASAR Product Handbook, Issue 1.1, 1 December 2002 can be found at http://envisat.esa.int/dataproducts/asar/CNTR6-3-6.htm#eph.asar.asardf.0pASA_IM__0P\n\n" );
    return 0;
    }*/

    /* These are passed in now */
    /* read in command-line arguments


       numFiles = (argc-1) - 4;
       sscanf( argv[ numFiles+1 ], "%s", insFileName );
       sscanf( argv[ numFiles+2 ], "%s", outFileName );
       sscanf( argv[ numFiles+3 ], "%d", &outType );
       sscanf( argv[ numFiles+4 ], "%hd", &windowStartTimeCodeword0 );


       debug
       numFiles = 1;
       sscanf( "D:\\data\\scign\\ASAR_RAW\\09786-2925-20040113\\ASA_IM__0CNPDK20040113_180720_000000152023_00213_09786_1579.N1", "%s", insFileName );
       sscanf( "D:\\data\\scign\\ASARAux\\ASA_INS_AXVIEC20031209_113421_20030211_000000_20041231_000000", "%s", insFileName );
       sscanf( "D:\\temp\\tmp_IMAGERY.raw", "%s", outFileName );
       sscanf( "1", "%d", &outType );
       sscanf( "0", "%hd", &windowStartTimeCodeword0 );
       printImMphIfZero    = 0;
       printImSphIfZero    = 0;
       printImMdsrIfZero   = 1;
       printInsMphIfZero   = 0;
       printInsSphIfZero   = 0;
       printInsGadsIfZero  = 0;
       printBlockIdIfZero  = 0;
       */

    /* modified the messages below EJF 2005/11/9 */
    if (is_bigendian())
    {
        printf("Running on big-endian CPU...\n");
        is_littlendian = 0;
    }
    else
    {
        printf("Running on little-endian CPU...\n");
        is_littlendian = 1;
    }


    /* open files */

    outFilePtr = fopen( outFileName, "wb" );
    if ( outFilePtr == NULL ) {
        printf( "*** ERROR - cannot open file: %s\n", outFileName );
        printf( "\n" );
        exit( -1 );
    }

    if ( printBlockIdIfZero == 0 ) {
        blockIdFilePtr = fopen( blockIdFileName, "wb" );
        if ( blockIdFilePtr == NULL ) {
            printf( "*** ERROR - cannot open file: %s\n", blockIdFileName );
            printf( "\n" );
            exit( -1 );
        }
    }

    insFilePtr = fopen( insFileName, "rb" );
    if ( insFilePtr == NULL ) {
        printf( "*** ERROR - cannot open file: %s\n", insFileName );
        printf( "\n" );
        exit( -1 );
    }


    if((dsrfp=fopen(timesOutFileName, "wb"))==NULL) {
        printf("Cannot open file: %s\n",timesOutFileName);
    }
    /* read MPH of ins file */

    printf( "Reading MPH of ins file...\n\n" );

    mphPtr = ( char * ) malloc( sizeof( char ) * mphSize );

    if ( mphPtr == NULL ){
        printf( "ERROR - mph allocation memory\n" );
        exit( -1 );
    }

    if ( (fread( mphPtr, sizeof( char ), mphSize, insFilePtr ) ) != mphSize ){
        printf( "ERROR - mph read error\n\n" );
        exit( -1 );
    }

    mphIns = readMph( mphPtr, printInsMphIfZero ); /* extract information from MPH */
    free ( mphPtr );


    /* read SPH from ins file */

    printf( "Reading SPH from ins file...\n\n" );

    sphPtr = ( char * ) malloc( sizeof( char ) * mphIns.sphSize );

    if ( sphPtr == NULL ){
        printf( "ERROR - sph allocation memory\n" );
        exit( -1 );
    }

    if ( (fread( sphPtr, sizeof( char ), mphIns.sphSize, insFilePtr ) ) != mphIns.sphSize ){
        printf( "ERROR - sph read error\n\n" );
        exit( -1 );
    }

    sphIns = readSphAux( sphPtr, printInsSphIfZero, mphIns );  /* extract information from SPH */
    free ( sphPtr );


    /* read GADS from ins file */

    printf( "Reading GADS from ins file...\n\n" );

    /*gadsPtr = ( char * ) malloc( sizeof( char ) * sphIns.dsd[ 0 ].dsrSize );

      if ( gadsPtr == NULL ){
      printf( "ERROR - gads allocation memory\n" );
      exit( -1 );
      }

    //edited by Z. Li at UCL on 16/02/2005
    if ( (fread( gadsPtr, sizeof( char ), sizeof( insGads ), insFilePtr ) ) != sizeof( insGads ) ){
    printf( "sizeof( insGads ): %d\n", sizeof( insGads ) );
    printf( "ERROR - gads read error\n\n" );
    printf( "%d %d %d\n", 171648, sizeof ( insGads ), 171648-sizeof( insGads ) );
    exit( -1 );
    }

    insGads =  readInsGads( gadsPtr, printInsGadsIfZero );
    free (gadsPtr);
    */

    if ( (fread( &insGads, sizeof( insGads ), 1, insFilePtr ) ) != 1 ){
        printf( "sizeof( insGads ): %d\n", sizeof( insGads ) );
        printf( "ERROR - gads read error\n\n" );
        printf( "%d %d %d\n", 171648, sizeof ( insGads ), 171648-sizeof( insGads ) );
        exit( -1 );
    }

    if (is_littlendian)
    {
        byte_swap_InsGads( &insGads );
    }

    if ( printInsGadsIfZero == 0 ) printInsGads( insGads );


    fclose( insFilePtr );


    /* fill LUTs */

    for ( i = 0; i < 4096; i++ ) {
        if ( i < 2048 ) ii = i;
        else ii = 256*(23-(i/256))+(i%256);
        if ( noAdcIfZero == 0 ){
            LUTi[ i ] = insGads.fbaq4NoAdc[ ii ];
            LUTq[ i ] = insGads.fbaq4NoAdc[ ii ];
        }
        else {
            LUTi[ i ] = insGads.fbaq4LutI[ ii ];
            LUTq[ i ] = insGads.fbaq4LutQ[ ii ];
        }
    }


    /* begin loop over files */

    for ( ii = 0; ii < numFiles; ii++ ) {


        /* open image file */

        //sscanf( argv[ ii+1 ], "%s", imFileName );

        //debug
        // sscanf( "D:\\data\\scign\\ASAR_RAW\\09786-2925-20040113\\ASA_IM__0CNPDK20040113_180720_000000152023_00213_09786_1579.N1", "%s", imFileName );

        imFilePtr = fopen( imFileName, "rb" );
        if ( imFilePtr == NULL ) {
            printf( "*** ERROR - cannot open file: %s\n", imFileName );
            printf( "\n" );
            exit( -1 );
        }


        /* read image MPH */

        printf( "Reading image MPH...\n\n" );

        mphPtr = ( char * ) malloc( sizeof( char ) * mphSize );

        if ( mphPtr == NULL ){
            printf( "ERROR - mph allocation memory\n" );
            exit( -1 );
        }

        if ( (fread( mphPtr, sizeof( char ), mphSize, imFilePtr ) ) != mphSize ){
            printf( "ERROR - mph read error\n\n" );
            exit( -1 );
        }

        mph = readMph( mphPtr, printImMphIfZero ); /* extract information from MPH */
        free ( mphPtr );


        /* read image SPH */

        printf( "Reading image SPH...\n\n" );

        sphPtr = ( char * ) malloc( sizeof( char ) * mph.sphSize );

        if ( sphPtr == NULL ){
            printf( "ERROR - sph allocation memory\n" );
            exit( -1 );
        }

        if ( (fread( sphPtr, sizeof( char ), mph.sphSize, imFilePtr ) ) != mph.sphSize ){
            printf( "ERROR - sph read error\n\n" );
            exit( -1 );
        }

        sph = readSph( sphPtr, printImSphIfZero, mph );  /* extract information from SPH */
        free ( sphPtr );


        /* read image MDSR from file */

        printf( "Reading and decoding image MDSR...\n\n" );

        bytesRead = 0;

        for ( i = 0; i < sph.dsd[ 0 ].numDsr; i++ ) {

            if ( (i+1)%1000 == 0 ) printf( "Line %5d\n", i+1 );

            modePacketCountOld = modePacketCount;

            /* sensing time added by Level 0 processor, as converted from Satellite Binary Time (SBT) counter embedded in each ISP */
            /**
             * Represents a binary time value field in ENVISAT records.
             *
             * <p> Refer to ENVISAT documentation for the exact definition of
             * this data type.
             */
            /*
               long  days;
               ulong seconds;
               ulong microseconds;
               */

            bytesRead = bytesRead +  4 * fread( &mdsrDsrTimeDays,            4, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_long, &mdsrDsrTimeDays, 1);
            }

            /* header added to the ISP by the Front End Processor (FEP) */
            bytesRead = bytesRead +  4 * fread( &mdsrDsrTimeSeconds,         4, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ulong, &mdsrDsrTimeSeconds, 1);
            }

            bytesRead = bytesRead +  4 * fread( &mdsrDsrTimeMicroseconds,    4, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ulong, &mdsrDsrTimeMicroseconds, 1);
            }

            /* jng . save the pulsetiming in a aux file. same day in year and microsec in day
             * modified to be able to compute a more precise sensingStart
             */

            dateAux[0] = 1.*(mdsrDsrTimeDays - daysToRemove);//day is in Mod Gregorian 2000. we only need days in the year, so remove day since 2000
            dateAux[1] = 1000000.*mdsrDsrTimeSeconds + mdsrDsrTimeMicroseconds;
            fwrite(dateAux,sizeof(double),2,dsrfp);

            bytesRead = bytesRead +  4 * fread( &mdsrGsrtTimeDays,           4, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_long, &mdsrGsrtTimeDays, 1);
            }
            bytesRead = bytesRead +  4 * fread( &mdsrGsrtTimeSeconds,        4, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ulong, &mdsrGsrtTimeSeconds, 1);
            }
            bytesRead = bytesRead +  4 * fread( &mdsrGsrtTimeMicroseconds,   4, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ulong, &mdsrGsrtTimeMicroseconds, 1);
            }
            bytesRead = bytesRead +  2 * fread( &mdsrIspLength,              2, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ushort, &mdsrIspLength, 1);
            }
            bytesRead = bytesRead +  2 * fread( &mdsrCrcErrs,                2, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ushort, &mdsrCrcErrs, 1);
            }
            bytesRead = bytesRead +  2 * fread( &mdsrRsErrs,                 2, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ushort, &mdsrRsErrs, 1);
            }
            bytesRead = bytesRead +  2 * fread( &mdsrSpare1,                 2, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ushort, &mdsrSpare1, 1);
            }

            /* 6-byte ISP Packet Header */
            bytesRead = bytesRead +  2 * fread( &mdsrPacketIdentification,   2, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ushort, &mdsrPacketIdentification, 1);
            }
            bytesRead = bytesRead +  2 * fread( &mdsrPacketSequenceControl,  2, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ushort, &mdsrPacketSequenceControl, 1);
            }
            bytesRead = bytesRead +  2 * fread( &mdsrPacketLength,           2, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ushort, &mdsrPacketLength, 1);
            }

            /* 30-byte Data Field Header in Packet Data Field */
            bytesRead = bytesRead + 30 * fread( &mdsrPacketDataHeader,      30, 1, imFilePtr );
            if (is_littlendian)
            {
                swap_endian_order(e_tid_ushort, &mdsrPacketDataHeader, 15);
            }

            priCodewordOldOld = priCodewordOld;
            windowStartTimeCodewordOldOld = windowStartTimeCodewordOld;

            priCodewordOld = priCodeword;
            windowStartTimeCodewordOld = windowStartTimeCodeword;

            dataFieldHeaderLength       =                     mdsrPacketDataHeader[  0 ];
            modeID                      =                     mdsrPacketDataHeader[  1 ];
            onBoardTimeMSW              =                     mdsrPacketDataHeader[  2 ];
            onBoardTimeLSW              =                     mdsrPacketDataHeader[  3 ];
            onBoardTimeLSB              = (unsigned char) ( ( mdsrPacketDataHeader[  4 ] >>  8 ) &  255);
            modePacketCount             =                     mdsrPacketDataHeader[  5 ]*256 + ((mdsrPacketDataHeader[  6 ] >> 8 ) & 255);
            antennaBeamSetNumber        = (unsigned char) ( ( mdsrPacketDataHeader[  6 ] >>  2 ) &   63);
            compressionRatio            = (unsigned char) ( ( mdsrPacketDataHeader[  6 ]       ) &    3); /* 1 is 8/4 compression */
            echoFlag                    = (unsigned char) ( ( mdsrPacketDataHeader[  7 ] >> 15 ) &    1);
            noiseFlag                   = (unsigned char) ( ( mdsrPacketDataHeader[  7 ] >> 14 ) &    1);
            calFlag                     = (unsigned char) ( ( mdsrPacketDataHeader[  7 ] >> 13 ) &    1);
            calType                     = (unsigned char) ( ( mdsrPacketDataHeader[  7 ] >> 12 ) &    1);
            cyclePacketCount            =                 (   mdsrPacketDataHeader[  7 ]         & 4095);
            priCodeword                 =                     mdsrPacketDataHeader[  8 ];
            windowStartTimeCodeword     =                     mdsrPacketDataHeader[  9 ];
            windowLengthCodeword        =                     mdsrPacketDataHeader[ 10 ];
            upConverterLevel            = (short)         ( ( mdsrPacketDataHeader[ 11 ] >> 12 ) &   15);
            downConverterLevel          = (short)         ( ( mdsrPacketDataHeader[ 11 ] >>  7 ) &   31);
            TxPolarization              = (unsigned char) ( ( mdsrPacketDataHeader[ 11 ] >>  6 ) &    1);
            RxPolarization              = (unsigned char) ( ( mdsrPacketDataHeader[ 11 ] >>  5 ) &    1);
            calibrationRowNumber        = (unsigned char) ( ( mdsrPacketDataHeader[ 11 ]       ) &   31);
            TxPulseLengthCodeword       = (unsigned short)( ( mdsrPacketDataHeader[ 12 ] >>  6 ) & 1023);
            beamAdjDeltaCodeword        = (unsigned char) ( ( mdsrPacketDataHeader[ 12 ]       ) &   63);
            chirpPulseBandwidthCodeword = (unsigned char) ( ( mdsrPacketDataHeader[ 13 ] >>  8 ) &  255);
            auxTxMonitorLevel           = (unsigned char) ( ( mdsrPacketDataHeader[ 13 ]       ) &  255);
            resamplingFactor            =                     mdsrPacketDataHeader[ 14 ];

            if ( printImMdsrIfZero == 0 ) {
                onBoardTimeIntegerSeconds = (unsigned int) (onBoardTimeMSW*256 + ( ( onBoardTimeLSW >> 8 ) & 255 ) );
                onBoardTimeFractionalSecondsInt = (unsigned short) ((onBoardTimeLSW & 255)*256 + onBoardTimeLSB);
                /* onBoardTimeFractionalSeconds = (double) ((double)onBoardTimeFractionalSecondsInt/65536.); */
                printf( "%6d %2u %2x %8d %5d %d %d %d %d %d %d %d %4d %5d %5d %5d %2d %2d %1d %1d %2u %4d %2d %3d %3d %5d\n", i+1, dataFieldHeaderLength, modeID, onBoardTimeIntegerSeconds, onBoardTimeFractionalSecondsInt, modePacketCount, antennaBeamSetNumber, compressionRatio, echoFlag, noiseFlag, calFlag, calType, cyclePacketCount, priCodeword, windowStartTimeCodeword, windowLengthCodeword, upConverterLevel, downConverterLevel, TxPolarization, RxPolarization, calibrationRowNumber, TxPulseLengthCodeword, beamAdjDeltaCodeword, chirpPulseBandwidthCodeword, auxTxMonitorLevel, resamplingFactor );
            }


            if (( modePacketCount == modePacketCountOld+1 ) || ( firstTimeEqualsZero == 0 )) {

                /* write out data */

                if ( (echoFlag == 1) && (noiseFlag == 0) && (calFlag == 0) ){

                    if ( firstTimeEqualsZero == 0 ){
                        outSamples = ((mdsrIspLength+1-30)/64)*63 + ((mdsrIspLength+1-30)%64)-1;
                        if ( windowStartTimeCodeword0 == 0 ) windowStartTimeCodeword0 = windowStartTimeCodeword;
                        else if ( windowStartTimeCodeword0 != windowStartTimeCodeword ) printf( "Line %5d : windowStartTimeCodeword %5d : shifting this and subsequent data to %5d\n", i+1, windowStartTimeCodeword, windowStartTimeCodeword0 );
                        windowStartTimeCodewordOld = windowStartTimeCodeword;
                        firstTimeEqualsZero = 1;
                    }

                    /* check a few things  - still need to check TxPulseLength, chirpPulseBandwidthCodeword, beamAdjDeltaCodeword */

                    if ( ( i != 0 ) && ( priCodeword != priCodewordOld ) ) {
                        printf( "Line %5d : priCodeword changes from %5d to %5d : no action taken\n", i+1, priCodewordOld, priCodeword  );
                    }

                    if ( windowStartTimeCodeword != windowStartTimeCodewordOld ) {
                        printf( "Line %5d : windowStartTimeCodeword changes from %5d to %5d : shifting this and subsequent data to %5d\n", i+1, windowStartTimeCodewordOld, windowStartTimeCodeword, windowStartTimeCodeword0 );
                    }

                    /* read 64-byte blocks */
                    for ( j = 0; j < (mdsrIspLength+1-30)/64; j++ ) {
                        fread( &mdsrBlockId[ j ], sizeof( char ),  1, imFilePtr );
                        fread( &mdsrCheck,        sizeof( char ), 63, imFilePtr );
                        bytesRead = bytesRead + 64;
                        for ( k = 0; k < 63; k++ ) {
                            mdsrLine[ 2*63*j+2*k   ] = LUTi[ 256*(15-((mdsrCheck[ k ] >> 4) & 15))+mdsrBlockId[ j ] ];
                            mdsrLine[ 2*63*j+2*k+1 ] = LUTq[ 256*(15-( mdsrCheck[ k ]       & 15))+mdsrBlockId[ j ] ];
                            /* if ( i == 0 ) {
                               printf( "k,sample,blockId,i_in,q_in,i_out,q_out: %2d %4d %3d %2d %2d %15f %15f\n", k, 63*j+k,  mdsrBlockId[ j ], ((mdsrCheck[ k ] >> 4) & 15), ( mdsrCheck[ k ] & 15), mdsrLine[ 2*k ], mdsrLine[ 2*k+1 ] );
                               } */
                        }
                    }

                    /* read partial last block */
                    fread( &mdsrBlockId[ j ], sizeof( char ),  1, imFilePtr );
                    fread( &mdsrCheck,        sizeof( char ), ((mdsrIspLength+1-30)%64)-1, imFilePtr );
                    bytesRead = bytesRead + (mdsrIspLength+1-30)%64;
                    for ( k = 0; k < ((mdsrIspLength+1-30)%64)-1; k++ ) {
                        mdsrLine[ 2*63*j+2*k   ] = LUTi[ 256*(15-((mdsrCheck[ k ] >> 4) & 15))+mdsrBlockId[ j ] ];
                        mdsrLine[ 2*63*j+2*k+1 ] = LUTq[ 256*(15-( mdsrCheck[ k ]       & 15))+mdsrBlockId[ j ] ];
                        /* if ( i == 0 ) {
                           printf( "k,sample,blockId,i_in,q_in,i_out,q_out: %2d %4d %3d %2d %2d %15f %15f\n", k, 63*j+k,  mdsrBlockId[ j ], ((mdsrCheck[ k ] >> 4) & 15), ( mdsrCheck[ k ] & 15), mdsrLine[ 2*k ], mdsrLine[ 2*k+1 ] );
                           } */
                    }

                    if ( windowStartTimeCodeword != windowStartTimeCodeword0 ) {
                        sampleShift = windowStartTimeCodeword - windowStartTimeCodeword0;
                        if ( sampleShift < 0 ) {
                            for ( k = 0; k < outSamples+sampleShift; k++ ) {
                                mdsrLine[ 2*k   ] = mdsrLine[ 2*(k-sampleShift)   ];
                                mdsrLine[ 2*k+1 ] = mdsrLine[ 2*(k-sampleShift)+1 ];
                            }
                            for ( k = outSamples+sampleShift; k < outSamples; k++ ) {
                                mdsrLine[ 2*k   ] = 0.;
                                mdsrLine[ 2*k+1 ] = 0.;
                            }
                        }
                        else {
                            for ( k = outSamples-1; k >= sampleShift; k-- ) {
                                mdsrLine[ 2*k   ] = mdsrLine[ 2*(k-sampleShift)   ];
                                mdsrLine[ 2*k+1 ] = mdsrLine[ 2*(k-sampleShift)+1 ];
                            }
                            for ( k = sampleShift-1; k >= 0; k-- ) {
                                mdsrLine[ 2*k   ] = 0.;
                                mdsrLine[ 2*k+1 ] = 0.;
                            }
                        }
                    }

                }
                else {  /* skip ahead and write out previous line as a placeholder */
                    fseek( imFilePtr, mdsrIspLength+1-30, SEEK_CUR );
                    bytesRead = bytesRead + mdsrIspLength+1-30;
                }

                if ( printBlockIdIfZero == 0 ) {
                    if ( (fwrite( &mdsrBlockId, sizeof( unsigned char ), outSamples/63+1, blockIdFilePtr ) ) != outSamples/63+1 ){
                        printf( "ERROR - blockIdFile write error\n\n" );
                        exit( -1 );
                    }
                }

                if ( outType == 1 ) {
                    for ( k = 0; k < 2*outSamples; k++ ) {
                        mdsrLineInt = (mdsrLine[ k ]*127.5+127.5) + .5; /* 5 for rounding */
                        if ( mdsrLineInt <   0 ) mdsrLineInt =   0;
                        if ( mdsrLineInt > 255 ) mdsrLineInt = 255;
                        mdsrLineChar[ k ] = mdsrLineInt;
                    }
                    if ( (fwrite( &mdsrLineChar, 2*sizeof( unsigned char ), outSamples, outFilePtr ) ) != outSamples ){
                        printf( "ERROR - outFile write error\n\n" );
                        exit( -1 );
                    }
                }
                else {
                    if ( (fwrite( &mdsrLine, 2*sizeof( float ), outSamples, outFilePtr ) ) != outSamples ){
                        printf( "ERROR - outFile write error\n\n" );
                        exit( -1 );
                    }
                }

                outLines = outLines + 1;

            }
            else if ( modePacketCount > modePacketCountOld+1 ) {
                /*
                   printf( "Line %5d : missing line - no action taken - %d %d\n", i+1, modePacketCount, modePacketCountOld );
                   fseek( imFilePtr, mdsrIspLength+1-30, SEEK_CUR );
                   bytesRead = bytesRead + mdsrIspLength+1-30;
                   */

                printf( "Line %5d : missing line(s) - filling with zeroes - %d %d\n", i+1, modePacketCount, modePacketCountOld );

                for ( j = 0; j < (modePacketCount-modePacketCountOld-1); j++ ) {
                    if ( outType == 1 ) {
                        for ( k = 0; k < 2*outSamples; k++ ) {
                            mdsrLineChar[ k ] = 128; /* (0.*127.5+127.5) + .5 */
                        }
                        if ( (fwrite( &mdsrLineChar, 2*sizeof( unsigned char ), outSamples, outFilePtr ) ) != outSamples ){
                            printf( "ERROR - outFile write error\n\n" );
                            exit( -1 );
                        }
                    }
                    else {
                        for ( k = 0; k < 2*outSamples; k++ ) {
                            mdsrLine[ k ] = 0.;
                        }
                        if ( (fwrite( &mdsrLine, 2*sizeof( float ), outSamples, outFilePtr ) ) != outSamples ){
                            printf( "ERROR - outFile write error\n\n" );
                            exit( -1 );
                        }
                    }
                    outLines = outLines + 1;
                }
                modePacketCountOld = modePacketCount - 1;

                /* set up to re-read header and decode current line */
                fseek( imFilePtr, -68, SEEK_CUR );
                bytesRead = bytesRead - 68;
                modePacketCountOld = modePacketCountOld - 1;
                modePacketCount = modePacketCount - 1;
                priCodewordOld = priCodewordOldOld;
                priCodeword = priCodewordOld;
                windowStartTimeCodewordOld = windowStartTimeCodewordOldOld;
                windowStartTimeCodeword = windowStartTimeCodewordOld;
                i = i - 1;

            }
            else if ( modePacketCount < modePacketCountOld+1 ) {
                printf( "Line %5d : duplicate line\n", i+1 );
                fseek( imFilePtr, mdsrIspLength+1-30, SEEK_CUR );
                bytesRead = bytesRead + mdsrIspLength+1-30;
                modePacketCount = modePacketCountOld;
            }
            else {
                printf( "Line %5d : error - %d %d\n", i+1, modePacketCount, modePacketCountOld );
                exit( -1 );
            }

        }

        if ( (i-1+1)%1000 != 0 ) printf( "Line %5d\n\n", i-1+1 );


        /* write out a few things */
        /*
           pri = priCodeword / insGads.sampRate;
           windowStartTime = windowStartTimeCodeword0 / insGads.sampRate;
           TxPulseLength = TxPulseLengthCodeword / insGads.sampRate;
           chirpPulseBandwidth = (double)chirpPulseBandwidthCodeword*16.e6/255.;

           windowLength = windowLengthCodeword / insGads.sampRate;
           beamAdjDelta = (double)(beamAdjDeltaCodeword-32)*360./4096.;
           printf( "%s%d\n",      "swathNum:                             ", insGads.timelineIm.swathNums[ antennaBeamSetNumber-1 ] );
           printf( "%s%d\n",      "mValue:                               ", insGads.timelineIm.mValues[ antennaBeamSetNumber-1 ] );
           printf( "%s%d\n",      "rValue:                               ", insGads.timelineIm.rValues[ antennaBeamSetNumber-1 ] );
           printf( "%s%d\n",      "gValue:                               ", insGads.timelineIm.gValues[ antennaBeamSetNumber-1 ] );
           printf( "%s%.9g\n",    "(rank*pri+windowStartTime)*c/2 (m):   ", (insGads.timelineIm.rValues[ antennaBeamSetNumber-1 ]*pri+windowStartTime)*c/2. );
           printf( "%s%.9g\n",    "(last)windowStartTime*c/2 (m):        ", windowStartTime*c/2. );
           printf( "%s%.9g\n",    "windowLength*c/2 (m):                 ", windowLength*c/2. );
           printf( "%s%.9g\n",    "rangeGateBias*c/2 (m):                ", insGads.rangeGateBias*c/2. );

           printf( "\nOutput information:\n\n" );
           printf( "%s%d\n",      "number of output samples:             ", outSamples );
           printf( "%s%d\n",      "number of output lines:               ", outLines );
           printf( "%s%.9g\n",    "chirp pulse bandwidth (Hz):           ", chirpPulseBandwidth );
           printf( "%s%.9g\n",    "prf (Hz):                             ", 1./pri );
           printf( "%s%.9g\n",    "range sampling frequency (Hz):        ", insGads.sampRate );
           printf( "%s%.9g\n",    "range sample spacing (m):             ", c/(2.*insGads.sampRate));
           printf( "%s%.9g\n",    "chirp slope (Hz/s):                   ", chirpPulseBandwidth/TxPulseLength );
           printf( "%s%.9g\n",    "pulse length (s):                     ", TxPulseLength );
           printf( "%s%.9g\n",    "radar frequency (Hz):                 ", insGads.radarFrequency );
           printf( "%s%.9g\n",    "wavelength (m):                       ", c/insGads.radarFrequency );
           printf( "%s%.9g\n",    "starting range (m):                   ", (insGads.timelineIm.rValues[ antennaBeamSetNumber-1 ]*pri+windowStartTime)*c/2. );
           printf( "\n" );
           */

        fclose( imFilePtr );

    }


    /* write out a few things */

    pri = priCodeword / insGads.sampRate;
    windowStartTime = windowStartTimeCodeword0 / insGads.sampRate;
    TxPulseLength = TxPulseLengthCodeword / insGads.sampRate;
    chirpPulseBandwidth = (double)chirpPulseBandwidthCodeword*16.e6/255.;

    /*//debug  */

    printf( "%s%d\n",      "priCodeword=:             ", priCodeword );
    printf( "%s%.12f\n",   "insGads.sampRate=:        ", insGads.sampRate );
    printf( "%s%.12f\n",   "pri=:                     ", pri );
    printf( "%s%d\n",      "windowStartTimeCodeword0=:", windowStartTimeCodeword0);
    printf( "%s%.12f\n",   "windowStartTime=:         ", windowStartTime );
    printf( "%s%.12f\n",   "TxPulseLength=:           ", TxPulseLength );
    printf( "%s%.12f\n",   "chirpPulseBandwidth=:     ", chirpPulseBandwidth );
    /* //end of debug
    */

    printf( "\nOutput information:\n\n" );
    printf( "%s%d\n",      "number of output samples:             ", outSamples );
    printf( "%s%d\n",      "number of output lines:               ", outLines );
    printf( "%s%.9g\n",    "chirp pulse bandwidth (Hz):           ", chirpPulseBandwidth );
    printf( "%s%.9g\n",    "prf (Hz):                             ", 1./pri );
    printf( "%s%.9g\n",    "range sampling frequency (Hz):        ", insGads.sampRate );
    printf( "%s%.9g\n",    "range sample spacing (m):             ", c/(2.*insGads.sampRate));
    printf( "%s%.9g\n",    "chirp slope (Hz/s):                   ", chirpPulseBandwidth/TxPulseLength );
    printf( "%s%.9g\n",    "pulse length (s):                     ", TxPulseLength );
    printf( "%s%.9g\n",    "radar frequency (Hz):                 ", insGads.radarFrequency );
    printf( "%s%.9g\n",    "wavelength (m):                       ", c/insGads.radarFrequency );
    printf( "%s%.9g\n",    "starting range (m):                   ", (insGads.timelineIm.rValues[ antennaBeamSetNumber-1 ]*pri+windowStartTime)*c/2. );
    printf( "%s%.9g\n",    "rangeGateBias*c/2 (m):                ", insGads.rangeGateBias*c/2. );
    printf( "\n" );

    *samples = outSamples;
    *lines = outLines;

    /* end program */

    //fclose(blockIdFilePtr);
    fclose( outFilePtr );
    fclose( dsrfp );

    printf( "\nDone.\n\n" );

    return;
    //    return imageOutput;
} /* end main */


/**********************************************************************************************************************************/

struct mphStruct readMph( const char *mphPtr, const int printMphIfZero )
{

    struct mphStruct mph;

    if ( 1 == 0 ) {
        printf( "check:\n%s\n", mphPtr+1247 );
    }

    memcpy( mph.product,                                   mphPtr+   0+ 9,  62 );
    memcpy( mph.procStage,                                 mphPtr+  73+11,   1 );
    memcpy( mph.refDoc,                                    mphPtr+  86+ 9,  23 );
    memcpy( mph.spare1,                                    mphPtr+ 120+ 0,  40 );
    memcpy( mph.acquisitionStation,                        mphPtr+ 161+21,  20 );
    memcpy( mph.procCenter,                                mphPtr+ 204+13,   6 );
    memcpy( mph.procTime,                                  mphPtr+ 225+11,  27 );
    memcpy( mph.softwareVer,                               mphPtr+ 265+14,  14 );
    memcpy( mph.spare2,                                    mphPtr+ 295+ 0,  40 );
    memcpy( mph.sensingStart,                              mphPtr+ 336+15,  27 );
    memcpy( mph.sensingStop,                               mphPtr+ 380+14,  27 );
    memcpy( mph.spare3,                                    mphPtr+ 423+ 0,  40 );
    memcpy( mph.phase,                                     mphPtr+ 464+ 6,   1 );
    mph.cycle                   = atoi( ( char * ) strchr( mphPtr+ 472+ 0, '=' )+1 );
    mph.relOrbit                = atoi( ( char * ) strchr( mphPtr+ 483+ 0, '=' )+1 );
    mph.absOrbit                = atoi( ( char * ) strchr( mphPtr+ 500+ 0, '=' )+1 );
    memcpy( mph.stateVectorTime,                           mphPtr+ 517+19,  27 );
    mph.deltaUt1                = atof( ( char * ) strchr( mphPtr+ 565+ 0, '=' )+1 );
    mph.xPosition               = atof( ( char * ) strchr( mphPtr+ 587+ 0, '=' )+1 );
    mph.yPosition               = atof( ( char * ) strchr( mphPtr+ 614+ 0, '=' )+1 );
    mph.zPosition               = atof( ( char * ) strchr( mphPtr+ 641+ 0, '=' )+1 );
    mph.xVelocity               = atof( ( char * ) strchr( mphPtr+ 668+ 0, '=' )+1 );
    mph.yVelocity               = atof( ( char * ) strchr( mphPtr+ 697+ 0, '=' )+1 );
    mph.zVelocity               = atof( ( char * ) strchr( mphPtr+ 726+ 0, '=' )+1 );
    memcpy( mph.vectorSource,                              mphPtr+ 755+15,   2 );
    memcpy( mph.spare4,                                    mphPtr+ 774+ 0,  40 );
    memcpy( mph.utcSbtTime,                                mphPtr+ 815+14,  27 );
    mph.satBinaryTime           = atoi( ( char * ) strchr( mphPtr+ 858+ 0, '=' )+1 );
    mph.clockStep               = atoi( ( char * ) strchr( mphPtr+ 886+ 0, '=' )+1 );
    memcpy( mph.spare5,                                    mphPtr+ 913+ 0,  32 );
    memcpy( mph.leapUtc,                                   mphPtr+ 946+10,  27 );
    mph.leapSign                = atoi( ( char * ) strchr( mphPtr+ 985+ 0, '=' )+1 );
    mph.leapErr                 = atoi( ( char * ) strchr( mphPtr+1000+ 0, '=' )+1 );
    memcpy( mph.spare6,                                    mphPtr+1011+ 0,  40 );
    mph.productErr              = atoi( ( char * ) strchr( mphPtr+1052+ 0, '=' )+1 );
    mph.totSize                 = atoi( ( char * ) strchr( mphPtr+1066+ 0, '=' )+1 );
    mph.sphSize                 = atoi( ( char * ) strchr( mphPtr+1104+ 0, '=' )+1 );
    mph.numDsd                  = atoi( ( char * ) strchr( mphPtr+1132+ 0, '=' )+1 );
    mph.dsdSize                 = atoi( ( char * ) strchr( mphPtr+1152+ 0, '=' )+1 );
    mph.numDataSets             = atoi( ( char * ) strchr( mphPtr+1180+ 0, '=' )+1 );
    memcpy( mph.spare7,                                    mphPtr+1206+ 0,  40 );

    if ( printMphIfZero == 0 ) {
        printf( "%s%.62s\n", "product:                 ", mph.product );
        printf( "%s%.1s\n",  "procStage:               ", mph.procStage );
        printf( "%s%.23s\n", "refDoc:                  ", mph.refDoc );
        printf( "%s%.40s\n", "spare1:                  ", mph.spare1 );
        printf( "%s%.20s\n", "acquisitionStation:      ", mph.acquisitionStation );
        printf( "%s%.6s\n",  "procCenter:              ", mph.procCenter );
        printf( "%s%.27s\n", "procTime:                ", mph.procTime );
        printf( "%s%.14s\n", "softwareVer:             ", mph.softwareVer );
        printf( "%s%.40s\n", "spare2:                  ", mph.spare2 );
        printf( "%s%.27s\n", "sensingStart:            ", mph.sensingStart );
        printf( "%s%.27s\n", "sensingStop:             ", mph.sensingStop );
        printf( "%s%.40s\n", "spare3:                  ", mph.spare3 );
        printf( "%s%.1s\n",  "phase:                   ", mph.phase );
        printf( "%s%d\n",    "cycle:                   ", mph.cycle );
        printf( "%s%d\n",    "relOrbit:                ", mph.relOrbit );
        printf( "%s%d\n",    "absOrbit:                ", mph.absOrbit );
        printf( "%s%.27s\n", "stateVectorTime:         ", mph.stateVectorTime );
        printf( "%s%f\n",    "deltaUt1:                ", mph.deltaUt1 );
        printf( "%s%f\n",    "xPosition:               ", mph.xPosition );
        printf( "%s%f\n",    "yPosition:               ", mph.yPosition );
        printf( "%s%f\n",    "zPosition:               ", mph.zPosition );
        printf( "%s%f\n",    "xVelocity:               ", mph.xVelocity );
        printf( "%s%f\n",    "yVelocity:               ", mph.yVelocity );
        printf( "%s%f\n",    "zVelocity:               ", mph.zVelocity );
        printf( "%s%.2s\n",  "vectorSource:            ", mph.vectorSource );
        printf( "%s%.40s\n", "spare4:                  ", mph.spare4 );
        printf( "%s%.27s\n", "utcSbtTime:              ", mph.utcSbtTime );
        printf( "%s%u\n",    "satBinaryTime:           ", mph.satBinaryTime );
        printf( "%s%u\n",    "clockStep:               ", mph.clockStep );
        printf( "%s%.32s\n", "spare5:                  ", mph.spare5 );
        printf( "%s%.27s\n", "leapUtc:                 ", mph.leapUtc );
        printf( "%s%d\n",    "leapSign:                ", mph.leapSign );
        printf( "%s%d\n",    "leapErr:                 ", mph.leapErr );
        printf( "%s%.40s\n", "spare6:                  ", mph.spare6 );
        printf( "%s%d\n",    "productErr:              ", mph.productErr );
        printf( "%s%d\n",    "totSize:                 ", mph.totSize );
        printf( "%s%d\n",    "sphSize:                 ", mph.sphSize );
        printf( "%s%d\n",    "numDsd:                  ", mph.numDsd );
        printf( "%s%d\n",    "dsdSize:                 ", mph.dsdSize );
        printf( "%s%d\n",    "numDataSets:             ", mph.numDataSets );
        printf( "%s%.40s\n", "spare7:                  ", mph.spare7 );
        printf( "\n" );
    }

    return mph;

} /* end readMph */

/**********************************************************************************************************************************/


/**********************************************************************************************************************************/

struct sphStruct readSph( const char *sphPtr, const int printSphIfZero, const struct mphStruct mph )
{

    struct sphStruct sph;
    int i;

    memcpy( sph.sphDescriptor,                             sphPtr+  0+16,  28 );
    sph.startLat                = atof( ( char * ) strchr( sphPtr+ 46+ 0, '=' )+1 ) * 1.e-6;
    sph.startLon                = atof( ( char * ) strchr( sphPtr+ 78+ 0, '=' )+1 ) * 1.e-6;
    sph.stopLat                 = atof( ( char * ) strchr( sphPtr+111+ 0, '=' )+1 ) * 1.e-6;
    sph.stopLon                 = atof( ( char * ) strchr( sphPtr+142+ 0, '=' )+1 ) * 1.e-6;
    sph.satTrack                = atof( ( char * ) strchr( sphPtr+174+ 0, '=' )+1 );
    memcpy( sph.spare1,                                    sphPtr+205+ 0,  50 );
    sph.ispErrorsSignificant    = atoi( ( char * ) strchr( sphPtr+256+ 0, '=' )+1 );
    sph.missingIspsSignificant  = atoi( ( char * ) strchr( sphPtr+281+ 0, '=' )+1 );
    sph.ispDiscardedSignificant = atoi( ( char * ) strchr( sphPtr+308+ 0, '=' )+1 );
    sph.rsSignificant           = atoi( ( char * ) strchr( sphPtr+336+ 0, '=' )+1 );
    memcpy( sph.spare2,                                    sphPtr+353+ 0,  50 );
    sph.numErrorIsps            = atoi( ( char * ) strchr( sphPtr+404+ 0, '=' )+1 );
    sph.errorIspsThresh         = atof( ( char * ) strchr( sphPtr+431+ 0, '=' )+1 );
    sph.numMissingIsps          = atoi( ( char * ) strchr( sphPtr+468+ 0, '=' )+1 );
    sph.missingIspsThresh       = atof( ( char * ) strchr( sphPtr+497+ 0, '=' )+1 );
    sph.numDiscardedIsps        = atoi( ( char * ) strchr( sphPtr+536+ 0, '=' )+1 );
    sph.discardedIspsThresh     = atof( ( char * ) strchr( sphPtr+567+ 0, '=' )+1 );
    sph.numRsIsps               = atoi( ( char * ) strchr( sphPtr+608+ 0, '=' )+1 );
    sph.rsThresh                = atof( ( char * ) strchr( sphPtr+632+ 0, '=' )+1 );
    memcpy( sph.spare3,                                    sphPtr+661+ 0, 100 );
    memcpy( sph.txRxPolar,                                 sphPtr+762+13,   5 );
    memcpy( sph.swath,                                     sphPtr+782+ 7,   3 );
    memcpy( sph.spare4,                                    sphPtr+794+ 0,  41 );

    if ( 1 == 0 ) {
        printf( "check:\n%s\n", sphPtr+836+ 0 );
    }

    if ( printSphIfZero == 0 ) {
        printf( "%s%.28s\n",  "sphDescriptor:           ", sph.sphDescriptor );
        printf( "%s%f\n",     "startLat:                ", sph.startLat );
        printf( "%s%f\n",     "startLon:                ", sph.startLon );
        printf( "%s%f\n",     "stopLat:                 ", sph.stopLat );
        printf( "%s%f\n",     "stopLon:                 ", sph.stopLon );
        printf( "%s%f\n",     "satTrack:                ", sph.satTrack );
        printf( "%s%.50s\n",  "spare1:                  ", sph.spare1 );
        printf( "%s%d\n",     "ispErrorsSignificant:    ", sph.ispErrorsSignificant );
        printf( "%s%d\n",     "missingIspsSignificant:  ", sph.missingIspsSignificant );
        printf( "%s%d\n",     "ispDiscardedSignificant: ", sph.ispDiscardedSignificant );
        printf( "%s%d\n",     "rsSignificant:           ", sph.rsSignificant );
        printf( "%s%.50s\n",  "spare2:                  ", sph.spare2 );
        printf( "%s%d\n",     "numErrorIsps:            ", sph.numErrorIsps );
        printf( "%s%f\n",     "errorIspsThresh:         ", sph.errorIspsThresh );
        printf( "%s%d\n",     "numMissingIsps:          ", sph.numMissingIsps );
        printf( "%s%f\n",     "missingIspsThresh:       ", sph.missingIspsThresh );
        printf( "%s%d\n",     "numDiscardedIsps:        ", sph.numDiscardedIsps );
        printf( "%s%f\n",     "discardedIspsThresh:     ", sph.discardedIspsThresh );
        printf( "%s%d\n",     "numRsIsps:               ", sph.numRsIsps );
        printf( "%s%f\n",     "rsThresh:                ", sph.rsThresh );
        printf( "%s%.100s\n", "spare3:                  ", sph.spare3 );
        printf( "%s%.5s\n",   "txRxPolar:               ", sph.txRxPolar );
        printf( "%s%.3s\n",   "swath:                   ", sph.swath );
        printf( "%s%.41s\n",  "spare4:                  ", sph.spare4 );
    }

    for ( i = 0; i < mph.numDsd; i++ ){      /* extract DSDs from SPH */
        if ( i != 3 ) {          /* fourth is a spare DSD - see pdf page 537 */
            if (1 == 0) {
                printf( "check:\n%s\n",                       sphPtr+836+mph.dsdSize*i+  0+ 0 );
            }
            memcpy( sph.dsd[ i ].dsName,                     sphPtr+836+mph.dsdSize*i+  0+ 9, 28 );
            memcpy( sph.dsd[ i ].dsType,                     sphPtr+836+mph.dsdSize*i+ 39+ 8,  1 );
            memcpy( sph.dsd[ i ].filename,                   sphPtr+836+mph.dsdSize*i+ 49+10, 62 );
            sph.dsd[ i ].dsOffset = atoi( ( char * ) strchr( sphPtr+836+mph.dsdSize*i+123+ 0, '=' )+1 );
            sph.dsd[ i ].dsSize   = atoi( ( char * ) strchr( sphPtr+836+mph.dsdSize*i+162+ 0, '=' )+1 );
            sph.dsd[ i ].numDsr   = atoi( ( char * ) strchr( sphPtr+836+mph.dsdSize*i+199+ 0, '=' )+1 );
            sph.dsd[ i ].dsrSize  = atoi( ( char * ) strchr( sphPtr+836+mph.dsdSize*i+219+ 0, '=' )+1 );
            /* write out a few things */
            if ( printSphIfZero == 0 ) {
                printf( "%s%d%s%.28s\n",  "dsd[ ", i, " ].dsName:         ", sph.dsd[ i ].dsName );
                printf( "%s%d%s%.1s\n",   "dsd[ ", i, " ].dsType:         ", sph.dsd[ i ].dsType );
                printf( "%s%d%s%.62s\n",  "dsd[ ", i, " ].filename:       ", sph.dsd[ i ].filename );
                printf( "%s%d%s%d\n",     "dsd[ ", i, " ].dsOffset:       ", sph.dsd[ i ].dsOffset );
                printf( "%s%d%s%d\n",     "dsd[ ", i, " ].dsSize:         ", sph.dsd[ i ].dsSize );
                printf( "%s%d%s%d\n",     "dsd[ ", i, " ].numDsr:         ", sph.dsd[ i ].numDsr );
                printf( "%s%d%s%d\n",     "dsd[ ", i, " ].dsrSize:        ", sph.dsd[ i ].dsrSize );
            }
        }
    }

    if ( printSphIfZero == 0 ) {
        printf( "\n" );
    }

    return sph;

} /* end readSph */

/**********************************************************************************************************************************/


/**********************************************************************************************************************************/

struct sphAuxStruct readSphAux( const char *sphPtr, const int printSphIfZero, const struct mphStruct mph )
{

    struct sphAuxStruct sph;
    int i;

    memcpy( sph.sphDescriptor,                          sphPtr+ 0+16, 28 );
    memcpy( sph.spare1,                                 sphPtr+46+ 0, 51 );

    if ( printSphIfZero == 0 ) {
        printf( "%s%.28s\n",  "sphDescriptor:           ", sph.sphDescriptor );
        printf( "%s%.51s\n",  "spare1:                  ", sph.spare1 );
    }

    for ( i = 0; i < mph.numDsd; i++ ){      /* extract DSDs from SPH */
        memcpy( sph.dsd[ i ].dsName,                     sphPtr+ 98+mph.dsdSize*i+  0+ 9, 28 );
        memcpy( sph.dsd[ i ].dsType,                     sphPtr+ 98+mph.dsdSize*i+ 39+ 8,  1 );
        memcpy( sph.dsd[ i ].filename,                   sphPtr+ 98+mph.dsdSize*i+ 49+10, 62 );
        sph.dsd[ i ].dsOffset = atoi( ( char * ) strchr( sphPtr+ 98+mph.dsdSize*i+123+ 0, '=' )+1 );
        sph.dsd[ i ].dsSize   = atoi( ( char * ) strchr( sphPtr+ 98+mph.dsdSize*i+162+ 0, '=' )+1 );
        sph.dsd[ i ].numDsr   = atoi( ( char * ) strchr( sphPtr+ 98+mph.dsdSize*i+199+ 0, '=' )+1 );
        sph.dsd[ i ].dsrSize  = atoi( ( char * ) strchr( sphPtr+ 98+mph.dsdSize*i+219+ 0, '=' )+1 );
        /* write out a few things */
        if ( printSphIfZero == 0 ) {
            printf( "%s%d%s%.28s\n",  "dsd[ ", i, " ].dsName:         ", sph.dsd[ i ].dsName );
            printf( "%s%d%s%.1s\n",   "dsd[ ", i, " ].dsType:         ", sph.dsd[ i ].dsType );
            printf( "%s%d%s%.62s\n",  "dsd[ ", i, " ].filename:       ", sph.dsd[ i ].filename );
            printf( "%s%d%s%d\n",     "dsd[ ", i, " ].dsOffset:       ", sph.dsd[ i ].dsOffset );
            printf( "%s%d%s%d\n",     "dsd[ ", i, " ].dsSize:         ", sph.dsd[ i ].dsSize );
            printf( "%s%d%s%d\n",     "dsd[ ", i, " ].numDsr:         ", sph.dsd[ i ].numDsr );
            printf( "%s%d%s%d\n",     "dsd[ ", i, " ].dsrSize:        ", sph.dsd[ i ].dsrSize );
        }
    }

    if ( printSphIfZero == 0 ) {
        printf( "\n" );
    }

    return sph;

} /* end readSphAux */

/**********************************************************************************************************************************/


/**********************************************************************************************************************************/

void printInsGads( const struct insGadsStruct insGads )
{

    int i;

    printf( "%s%d\n",      "dsrTime.days:                       ", insGads.dsrTime.days );
    printf( "%s%d\n",      "dsrTime.seconds:                    ", insGads.dsrTime.seconds );
    printf( "%s%d\n",      "dsrTime.microseconds:               ", insGads.dsrTime.microseconds );
    printf( "%s%d\n",      "dsrLength:                          ", insGads.dsrLength );
    printf( "%s%.9g\n",    "radarFrequency:                     ", insGads.radarFrequency );
    printf( "%s%.9g\n",    "sampRate:                           ", insGads.sampRate );
    printf( "%s%.9g\n",    "offsetFreq:                         ", insGads.offsetFreq );
    printf( "%s%.9g\n",    "rangeGateBias:                      ", insGads.rangeGateBias );
    printf( "%s%.9g\n",    "rangeGateBiasGm:                    ", insGads.rangeGateBiasGm );
    printf( "%s%f\n",      "refElevAngleIs1:                    ", insGads.refElevAngleIs1 );
    printf( "%s%f\n",      "refElevAngleIs2:                    ", insGads.refElevAngleIs2 );
    printf( "%s%f\n",      "refElevAngleIs3Ss2:                 ", insGads.refElevAngleIs3Ss2 );
    printf( "%s%f\n",      "refElevAngleIs4Ss3:                 ", insGads.refElevAngleIs4Ss3 );
    printf( "%s%f\n",      "refElevAngleIs5Ss4:                 ", insGads.refElevAngleIs5Ss4 );
    printf( "%s%f\n",      "refElevAngleIs6Ss5:                 ", insGads.refElevAngleIs6Ss5 );
    printf( "%s%f\n",      "refElevAngleIs7:                    ", insGads.refElevAngleIs7 );
    printf( "%s%f\n",      "refElevAngleSs1:                    ", insGads.refElevAngleSs1 );
    printf( "%s%.9g\n",    "swstCalP2:                          ", insGads.swstCalP2 );
    printf( "%s%u\n",      "perCalWindowsEc:                    ", insGads.perCalWindowsEc );
    printf( "%s%u\n",      "perCalWindowsMs:                    ", insGads.perCalWindowsMs );
    printf( "%s%u\n",      "initCalBeamSetWv:                   ", insGads.initCalBeamSetWv );
    printf( "%s%u\n",      "beamSetEc:                          ", insGads.beamSetEc );
    printf( "%s%u\n",      "beamSetMs:                          ", insGads.beamSetMs );
    printf( "%s%u\n",      "mEc:                                ", insGads.mEc );
    printf( ".\n" );
    printf( ".\n" );
    printf( ".\n" );

    for ( i = 0; i < 4096; i++ ) printf( "%s%4d%s%15f %15f %15f\n", "fbaq4LutI,Q,NoAdc[ ", i, " ]:          ", insGads.fbaq4LutI[ i ], insGads.fbaq4LutQ[ i ], insGads.fbaq4NoAdc[ i ] );
    printf( ".\n" );
    printf( ".\n" );
    printf( ".\n" );

    printf( "\n" );

    /* exit( 0 ); */

    return;

} /* end printInsGads */

/**********************************************************************************************************************************/
/**********************************************************************************************************************************/
/**********************************************************
 ** Function: byte_swap_InsGads
 **
 ** Purpose: Convert the bytes of struct insGadsStruct for a little endian order machine
 **
 ** Comment: struct testStruct should be redefined in the future!
 **
 ** Author: Zhenhong Li at UCL
 **
 ** Created: 17/02/2005
 **
 ** Modified:
 **
 ;**********************************************************/
void byte_swap_InsGads( struct insGadsStruct* InsGads )
{
    swap_endian_order(e_tid_long, &(*InsGads).dsrTime, 3);
    swap_endian_order(e_tid_ulong, &(*InsGads).dsrLength, 1);
    swap_endian_order(e_tid_float, &(*InsGads).radarFrequency, 1);
    swap_endian_order(e_tid_float, &(*InsGads).sampRate, 1);
    swap_endian_order(e_tid_float, &(*InsGads).offsetFreq, 1);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseIm0TxH1, 64);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseIm0TxV1, 64);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseIm0TxH1a, 64);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseIm0TxV1a, 64);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseIm0RxH2, 64);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseIm0RxV2, 64);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseIm0H3, 64);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseIm0V3, 64);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseImTxH1, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseImTxV1, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseImTxH1a, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseImTxV1a, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseImRxH2, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseImRxV2, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseImH3, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseImV3, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseApTxH1, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseApTxV1, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseApTxH1a, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseApTxV1a, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseApRxH2, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseApRxV2, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseApH3, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseApV3, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWvTxH1, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWvTxV1, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWvTxH1a, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWvTxV1a, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWvRxH2, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWvRxV2, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWvH3, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWvV3, 448);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWsTxH1, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWsTxV1, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWsTxH1a, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWsTxV1a, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWsRxH2, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWsRxV2, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWsH3, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseWsV3, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseGmTxH1, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseGmTxV1, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseGmTxH1a, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseGmTxV1a, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseGmRxH2, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseGmRxV2, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseGmH3, 320);
    swap_endian_order(e_tid_float, &(*InsGads).calPulseGmV3, 320);
    swap_endian_order(e_tid_float, &(*InsGads).nomPulseIm, 63);
    swap_endian_order(e_tid_float, &(*InsGads).nomPulseAp, 63);
    swap_endian_order(e_tid_float, &(*InsGads).nomPulseWv, 63);
    swap_endian_order(e_tid_float, &(*InsGads).nomPulseWs, 45);
    swap_endian_order(e_tid_float, &(*InsGads).nomPulseGm, 45);
    swap_endian_order(e_tid_float, &(*InsGads).azPatternIs1, 101);
    swap_endian_order(e_tid_float, &(*InsGads).azPatternIs2, 101);
    swap_endian_order(e_tid_float, &(*InsGads).azPatternIs3Ss2, 101);
    swap_endian_order(e_tid_float, &(*InsGads).azPatternIs4Ss3, 101);
    swap_endian_order(e_tid_float, &(*InsGads).azPatternIs5Ss4, 101);
    swap_endian_order(e_tid_float, &(*InsGads).azPatternIs6Ss5, 101);
    swap_endian_order(e_tid_float, &(*InsGads).azPatternIs7, 101);
    swap_endian_order(e_tid_float, &(*InsGads).azPatternSs1, 101);
    swap_endian_order(e_tid_float, &(*InsGads).rangeGateBias, 1);
    swap_endian_order(e_tid_float, &(*InsGads).rangeGateBiasGm, 1);
    swap_endian_order(e_tid_float, &(*InsGads).adcLutI, 255);
    swap_endian_order(e_tid_float, &(*InsGads).adcLutQ, 255);
    swap_endian_order(e_tid_float, &(*InsGads).full8LutI, 256);
    swap_endian_order(e_tid_float, &(*InsGads).full8LutQ, 256);
    swap_endian_order(e_tid_float, &(*InsGads).fbaq4LutI, 4096);
    swap_endian_order(e_tid_float, &(*InsGads).fbaq3LutI, 2048);
    swap_endian_order(e_tid_float, &(*InsGads).fbaq2LutI, 1024);
    swap_endian_order(e_tid_float, &(*InsGads).fbaq4LutQ, 4096);
    swap_endian_order(e_tid_float, &(*InsGads).fbaq3LutQ, 2048);
    swap_endian_order(e_tid_float, &(*InsGads).fbaq2LutQ, 1024);
    swap_endian_order(e_tid_float, &(*InsGads).fbaq4NoAdc, 4096);
    swap_endian_order(e_tid_float, &(*InsGads).fbaq3NoAdc, 2048);
    swap_endian_order(e_tid_float, &(*InsGads).fbaq2NoAdc, 1024);
    swap_endian_order(e_tid_float, &(*InsGads).smLutI, 16);
    swap_endian_order(e_tid_float, &(*InsGads).smLutQ, 16);
    swap_endian_order(e_tid_ushort, &(*InsGads).swathConfigIm, 28);
    swap_endian_order(e_tid_float, &(*InsGads).swathConfigIm.resampleFactor, 7);
    swap_endian_order(e_tid_ushort, &(*InsGads).swathConfigAp, 28);
    swap_endian_order(e_tid_float, &(*InsGads).swathConfigAp.resampleFactor, 7);
    swap_endian_order(e_tid_ushort, &(*InsGads).swathConfigWs, 28);
    swap_endian_order(e_tid_float, &(*InsGads).swathConfigWs.resampleFactor, 7);
    swap_endian_order(e_tid_ushort, &(*InsGads).swathConfigGm, 28);
    swap_endian_order(e_tid_float, &(*InsGads).swathConfigGm.resampleFactor, 7);
    swap_endian_order(e_tid_ushort, &(*InsGads).swathConfigWv, 28);
    swap_endian_order(e_tid_float, &(*InsGads).swathConfigWv.resampleFactor, 7);
    swap_endian_order(e_tid_ushort, &(*InsGads).perCalWindowsEc, 1);
    swap_endian_order(e_tid_ushort, &(*InsGads).perCalWindowsMs, 1);
    swap_endian_order(e_tid_ushort, &(*InsGads).swathIdIm, 14);
    swap_endian_order(e_tid_ushort, &(*InsGads).swathIdAp, 14);
    swap_endian_order(e_tid_ushort, &(*InsGads).swathIdWs, 14);
    swap_endian_order(e_tid_ushort, &(*InsGads).swathIdGm, 14);
    swap_endian_order(e_tid_ushort, &(*InsGads).swathIdWv, 14);
    swap_endian_order(e_tid_ushort, &(*InsGads).initCalBeamSetWv, 1);
    swap_endian_order(e_tid_ushort, &(*InsGads).beamSetEc, 1);
    swap_endian_order(e_tid_ushort, &(*InsGads).beamSetMs, 1);
    swap_endian_order(e_tid_ushort, &(*InsGads).calSeq, 32);
    swap_endian_order(e_tid_ushort, &(*InsGads).timelineIm, 28);
    swap_endian_order(e_tid_ushort, &(*InsGads).timelineAp, 28);
    swap_endian_order(e_tid_ushort, &(*InsGads).timelineWs, 28);
    swap_endian_order(e_tid_ushort, &(*InsGads).timelineGm, 28);
    swap_endian_order(e_tid_ushort, &(*InsGads).timelineWv, 28);
    swap_endian_order(e_tid_ushort, &(*InsGads).mEc, 1);
    swap_endian_order(e_tid_float, &(*InsGads).refElevAngleIs1, 1);
    swap_endian_order(e_tid_float, &(*InsGads).refElevAngleIs2, 1);
    swap_endian_order(e_tid_float, &(*InsGads).refElevAngleIs3Ss2, 1);
    swap_endian_order(e_tid_float, &(*InsGads).refElevAngleIs4Ss3, 1);
    swap_endian_order(e_tid_float, &(*InsGads).refElevAngleIs5Ss4, 1);
    swap_endian_order(e_tid_float, &(*InsGads).refElevAngleIs6Ss5, 1);
    swap_endian_order(e_tid_float, &(*InsGads).refElevAngleIs7, 1);
    swap_endian_order(e_tid_float, &(*InsGads).refElevAngleSs1, 1);
    swap_endian_order(e_tid_float, &(*InsGads).calLoopRefIs1, 128);
    swap_endian_order(e_tid_float, &(*InsGads).calLoopRefIs2, 128);
    swap_endian_order(e_tid_float, &(*InsGads).calLoopRefIs3Ss2, 128);
    swap_endian_order(e_tid_float, &(*InsGads).calLoopRefIs4Ss3, 128);
    swap_endian_order(e_tid_float, &(*InsGads).calLoopRefIs5Ss4, 128);
    swap_endian_order(e_tid_float, &(*InsGads).calLoopRefIs6Ss5, 128);
    swap_endian_order(e_tid_float, &(*InsGads).calLoopRefIs7, 128);
    swap_endian_order(e_tid_float, &(*InsGads).calLoopRefSs1, 128);

    //struct testStruct should be redefined in the future.
    swap_endian_order(e_tid_float, &(*InsGads).im, 17);
    swap_endian_order(e_tid_float, &(*InsGads).ap, 17);
    swap_endian_order(e_tid_float, &(*InsGads).ws, 17);
    swap_endian_order(e_tid_float, &(*InsGads).gm, 17);
    swap_endian_order(e_tid_float, &(*InsGads).wv, 17);

    swap_endian_order(e_tid_float, &(*InsGads).swstCalP2, 1);
}

/**********************************************************
 ** Function: is_bigendian
 **
 ** Purpose: Test whether it is a bigendian machine
 **
 **   Return values: true: 1, false: 0
 **
 ** Comment:
 **
 ** Author: Eric J Fielding at JPL
 **
 ** Created:
 **
 ** Modified:
 **
 ;**********************************************************/
int is_bigendian()
{

    int bigendian, littleendian, test;
    unsigned char t[4];

    littleendian=256;
    bigendian=256*256;

    t[0]=0;
    t[1]=1;
    t[2]=0;
    t[3]=0;

    memcpy(&test, &t[0], 4);

    /* printf("test: %i\n",test); */
    if(test==bigendian)return(1);
    if(test==littleendian)return(0);
    printf("Error in endian test, test= %i ********\n",test);
}

/*
 * Function: byte_swap_short.c
 */
/**
 *
 * Swaps bytes within NUMBER_OF_SWAPS two-byte words,
 *   starting at address BUFFER.
 *
 * @param buffer the one element typed buffer
 * to convert for a little endian order machine
 *
 * @param number_of_swaps number of elements to convert
 *
 */
void byte_swap_short(short *buffer, uint number_of_swaps)
{
    short* temp = buffer;
    uint swap_loop;

    for (swap_loop = 0, temp = buffer; swap_loop < number_of_swaps; swap_loop++, temp++) {
        *temp = (short)(((*temp & 0x00ff) << 8) |
                ((*temp & 0xff00) >> 8));
    }
}


/*
Function: byte_swap_long.c
*/
/**
 *
 *  Swaps bytes within NUMBER_OF_SWAPS four-byte words,
 *     starting at address BUFFER.
 *
 *
 */
void byte_swap_long(long *buffer, uint number_of_swaps)
{
    long *temp = buffer;
    uint swap_loop;

    for (swap_loop = 0, temp = buffer; swap_loop < number_of_swaps; swap_loop++, temp++) {
        *temp = ((*temp & 0x000000ff) << 24) |
            ((*temp & 0x0000ff00) << 8)  |
            ((*temp & 0x00ff0000) >> 8)  |
            ((*temp & 0xff000000) >> 24);
    }
}

/* ADDED THESE LINES TO TEST THE 4-BYTE INT TYPE ON 64 BIT */
/*
Function: byte_swap_int.c
*/
/**
 *
 *  Swaps bytes within NUMBER_OF_SWAPS four-byte words,
 *     starting at address BUFFER.
 *
 *
 */
void byte_swap_int(int *buffer, uint number_of_swaps)
{
    int *temp = buffer;
    uint swap_loop;

    for (swap_loop = 0, temp = buffer; swap_loop < number_of_swaps; swap_loop++, temp++) {
        *temp = ((*temp & 0x000000ff) << 24) |
            ((*temp & 0x0000ff00) << 8)  |
            ((*temp & 0x00ff0000) >> 8)  |
            ((*temp & 0xff000000) >> 24);
    }
}
/*
Function: byte_swap_uint.c
*/
/**
 *
 *  Swaps bytes within NUMBER_OF_SWAPS four-byte words,
 *     starting at address BUFFER.
 *
 *
 */
void byte_swap_uint(uint *buffer, uint number_of_swaps)
{
    uint *temp = buffer;
    uint swap_loop;

    for (swap_loop = 0, temp = buffer; swap_loop < number_of_swaps; swap_loop++, temp++) {
        *temp = ((*temp & 0x000000ff) << 24) |
            ((*temp & 0x0000ff00) << 8)  |
            ((*temp & 0x00ff0000) >> 8)  |
            ((*temp & 0xff000000) >> 24);
    }
}
/* ADDDED NEW LINES ABOVE */
/*  ************************************************************************** */

/*
Function: byte_swap_short.c
*/
/**
 *
 * Swaps bytes within NUMBER_OF_SWAPS two-byte words,
 *   starting at address BUFFER.
 *
 * @param buffer the one element typed buffer
 * to convert for a little endian order machine
 *
 * @param number_of_swaps number of elements to convert
 *
 */
void byte_swap_ushort(ushort* buffer, uint number_of_swaps)
{
    byte_swap_short((short*) buffer, number_of_swaps);
}

/*
 *  Function: byte_swap_ulong.c
 */
/**
 *
 * Swaps bytes within NUMBER_OF_SWAPS four-byte words,
 *     starting at address BUFFER.
 *
 * @param buffer the one element typed buffer
 * to convert for a little endian order machine
 *
 * @param number_of_swaps number of elements to convert
 *
 */
void byte_swap_ulong(ulong* buffer, uint number_of_swaps)
{
    byte_swap_long((long*) buffer, number_of_swaps);
}

/*
 *  Function: byte_swap_long.c
 */
/**
 *
 * Swaps bytes within NUMBER_OF_SWAPS four-byte words,
 *     starting at address BUFFER.
 *
 * @param buffer the one element typed buffer
 * to convert for a little endian order machine
 *
 * @param number_of_swaps number of elements to convert
 *
 */
void byte_swap_float(float* buffer, uint number_of_swaps)
{
    byte_swap_int((int*) buffer, number_of_swaps);

}



/*
Function:    epr_swap_endian_order
Access:      public API
Changelog:   2002/02/04  mp nitial version
*/
/**
 * Converts bytes for a little endian order machine
 *
 * @param field the pointer at data reading in
 *
 */
void swap_endian_order(EPR_EDataTypeId data_type_id, void* elems, uint num_elems)
{
    switch (data_type_id) {
        case e_tid_uchar:
        case e_tid_char:
        case e_tid_string:
            /* no conversion required */
            break;
        case e_tid_time:
            byte_swap_uint((uint*)elems, 3);
            break;
        case e_tid_spare:
            /* no conversion required */
            break;
        case e_tid_ushort:
            byte_swap_ushort((ushort*) elems, num_elems);
            break;
        case e_tid_short:
            byte_swap_short((short*) elems, num_elems);
            break;
        case e_tid_ulong:
            byte_swap_uint((uint*) elems, num_elems);
            break;
        case e_tid_long:
            byte_swap_int((int*) elems, num_elems);
            break;
        case e_tid_float:
            byte_swap_float((float*) elems, num_elems);
            break;
        case e_tid_double:
            printf( "swap_endian_order: DOUBLE type was not yet processed\n" );
            break;
        default:
            printf( "swap_endian_order: unknown data type\n" );
    }
}
