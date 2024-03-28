/*************************************************************************

  snaphu header file
  Written by Curtis W. Chen
  Copyright 2002 Board of Trustees, Leland Stanford Jr. University
  Please see the supporting documentation for terms of use.
  No warranty.

*************************************************************************/


/**********************/
/* defined constants  */
/**********************/

#define PROGRAMNAME          "snaphu"
#define VERSION              "2.0.6"
#define BUGREPORTEMAIL       "snaphu@gmail.com"
#ifdef PI
#undef PI
#endif
#define PI                   3.14159265358979323846
#define TWOPI                6.28318530717958647692
#define SQRTHALF             0.70710678118654752440
#define MAXSTRLEN            512
#define MAXTMPSTRLEN         1024
#define MAXLINELEN           2048
#define TRUE                 1
#define FALSE                0
#define LARGESHORT           32000
#define LARGEINT             2000000000
#define LARGEFLOAT           1.0e35
#define VERYFAR              LARGEINT
#define GROUNDROW            -2
#define GROUNDCOL            -2
#define BOUNDARYROW          -4
#define BOUNDARYCOL          -4
#define MAXGROUPBASE         LARGEINT
#define ONTREE               -1
#define INBUCKET             -2
#define NOTINBUCKET          -3
#define PRUNED               -4
#define MASKED               -5
#define BOUNDARYPTR          -6
#define BOUNDARYCANDIDATE    -7
#define BOUNDARYLEVEL        LARGEINT
#define INTERIORLEVEL        (BOUNDARYLEVEL-1)
#define MINBOUNDARYSIZE      100
#define POSINCR              0
#define NEGINCR              1
#define NOCOSTSHELF          -LARGESHORT
#define MINSCALARCOST        1
#define INITARRSIZE          500
#define NEWNODEBAGSTEP       500
#define CANDIDATEBAGSTEP     500
#define NEGBUCKETFRACTION    1.0
#define POSBUCKETFRACTION    1.0
#define CLIPFACTOR           0.6666666667
#define NSOURCELISTMEMINCR   1024
#define NLISTMEMINCR         1024
#define DEF_OUTFILE          "snaphu.out"
#define DEF_SYSCONFFILE      ""     /* "/usr/local/snaphu/snaphu.conf" */
#define DEF_WEIGHTFILE       ""     /* "snaphu.weight" */
#define DEF_AMPFILE          ""     /* "snaphu.amp" */
#define DEF_AMPFILE2         ""     /* "snaphu.amp" */
#define DEF_MAGFILE          ""     /* "snaphu.mag" */
#define DEF_CORRFILE         ""     /* "snaphu.corr" */
#define DEF_ESTFILE          ""     /* "snaphu.est" */
#define DEF_COSTINFILE       ""
#define DEF_BYTEMASKFILE     ""
#define DEF_DOTILEMASKFILE   ""
#define DEF_INITFILE         ""
#define DEF_FLOWFILE         ""
#define DEF_EIFILE           ""
#define DEF_ROWCOSTFILE      ""
#define DEF_COLCOSTFILE      ""
#define DEF_MSTROWCOSTFILE   ""
#define DEF_MSTCOLCOSTFILE   ""
#define DEF_MSTCOSTSFILE     ""
#define DEF_CORRDUMPFILE     ""
#define DEF_RAWCORRDUMPFILE  ""
#define DEF_CONNCOMPFILE     ""
#define DEF_COSTOUTFILE      ""
#define DEF_LOGFILE          ""
#define MAXITERATION         5000
#define NEGSHORTRANGE        SHRT_MIN
#define POSSHORTRANGE        SHRT_MAX
#define MAXRES               SCHAR_MAX
#define MINRES               SCHAR_MIN
#define PROBCOSTP            (-99.999)
#define NULLFILE             "/dev/null"
#define DEF_ERRORSTREAM      stderr
#define DEF_OUTPUTSTREAM     stdout
#define DEF_VERBOSESTREAM    NULL
#define DEF_COUNTERSTREAM    NULL
#define DEF_INITONLY         FALSE
#define DEF_INITMETHOD       MSTINIT
#define DEF_UNWRAPPED        FALSE
#define DEF_REGROWCONNCOMPS  FALSE
#define DEF_EVAL             FALSE
#define DEF_WEIGHT           1
#define DEF_COSTMODE         TOPO
#define DEF_VERBOSE          FALSE
#define DEF_AMPLITUDE        TRUE
#define AUTOCALCSTATMAX      0
#define MAXNSHORTCYCLE       8192
#define USEMAXCYCLEFRACTION  (-123)
#define COMPLEX_DATA         1         /* file format */
#define FLOAT_DATA           2         /* file format */
#define ALT_LINE_DATA        3         /* file format */
#define ALT_SAMPLE_DATA      4         /* file format */
#define TILEINITFILEFORMAT   ALT_LINE_DATA
#define TILEINITFILEROOT     "snaphu_tileinit_"
#define ABNORMAL_EXIT        1         /* exit code */
#define NORMAL_EXIT          0         /* exit code */
#define DUMP_PATH            "/tmp/"   /* default location for writing dumps */
#define NARMS                8         /* number of arms for Despeckle() */
#define ARMLEN               5         /* length of arms for Despeckle() */
#define KEDGE                5         /* length of edge detection window */
#define ARCUBOUND            200       /* capacities for cs2 */
#define MSTINIT              1         /* initialization method */
#define MCFINIT              2         /* initialization method */
#define BIGGESTDZRHOMAX      10000.0
#define SECONDSPERPIXEL      0.000001  /* for delay between thread creations */
#define MAXTHREADS           64
#define TMPTILEDIRROOT       "snaphu_tiles_"
#define TILEDIRMODE          511
#define TMPTILEROOT          "tmptile_"
#define TMPTILECOSTSUFFIX    "cost_"
#define TMPTILEOUTFORMAT     ALT_LINE_DATA
#define REGIONSUFFIX         "_regions"
#define LOGFILEROOT          "tmptilelog_"
#define RIGHT                1
#define DOWN                 2
#define LEFT                 3
#define UP                   4
#define TILEDPSICOLFACTOR    0.8
#define TILEOVRLPWARNTHRESH  400
#define ZEROCOSTARC          -LARGEINT
#define PINGPONG             2
#define SINGLEANTTRANSMIT    1
#define NOSTATCOSTS          0
#define TOPO                 1
#define DEFO                 2
#define SMOOTH               3
#define CONNCOMPOUTTYPEUCHAR 1
#define CONNCOMPOUTTYPEUINT  4


/* SAR and geometry parameter defaults */

#define DEF_ORBITRADIUS      7153000.0
#define DEF_ALTITUDE         0.0
#define DEF_EARTHRADIUS      6378000.0
#define DEF_BASELINE         150.0
#define DEF_BASELINEANGLE    (1.25*PI)
#define DEF_BPERP            0
#define DEF_TRANSMITMODE     PINGPONG
#define DEF_NLOOKSRANGE      1
#define DEF_NLOOKSAZ         5
#define DEF_NLOOKSOTHER      1
#define DEF_NCORRLOOKS       23.8
#define DEF_NCORRLOOKSRANGE  3  
#define DEF_NCORRLOOKSAZ     15
#define DEF_NEARRANGE        831000.0
#define DEF_DR               8.0
#define DEF_DA               20.0 
#define DEF_RANGERES         10.0
#define DEF_AZRES            6.0
#define DEF_LAMBDA           0.0565647


/* scattering model defaults */

#define DEF_KDS              0.02
#define DEF_SPECULAREXP      8.0
#define DEF_DZRCRITFACTOR    2.0
#define DEF_SHADOW           FALSE
#define DEF_DZEIMIN          -4.0
#define DEF_LAYWIDTH         16 
#define DEF_LAYMINEI         1.25
#define DEF_SLOPERATIOFACTOR 1.18
#define DEF_SIGSQEI          100.0


/* decorrelation model parameters */

#define DEF_DRHO             0.005
#define DEF_RHOSCONST1       1.3
#define DEF_RHOSCONST2       0.14
#define DEF_CSTD1            0.4
#define DEF_CSTD2            0.35
#define DEF_CSTD3            0.06
#define DEF_DEFAULTCORR      0.01
#define DEF_RHOMINFACTOR     1.3


/* pdf model parameters */

#define DEF_DZLAYPEAK        -2.0
#define DEF_AZDZFACTOR       0.99
#define DEF_DZEIFACTOR       4.0 
#define DEF_DZEIWEIGHT       0.5 
#define DEF_DZLAYFACTOR      1.0
#define DEF_LAYCONST         0.9
#define DEF_LAYFALLOFFCONST  2.0
#define DEF_SIGSQSHORTMIN    1
#define DEF_SIGSQLAYFACTOR   0.1


/* deformation mode parameters */

#define DEF_DEFOAZDZFACTOR   1.0
#define DEF_DEFOTHRESHFACTOR 1.2
#define DEF_DEFOMAX          1.2
#define DEF_SIGSQCORR        0.05
#define DEF_DEFOLAYCONST     0.9


/* algorithm parameters */

#define DEF_FLIPPHASESIGN    FALSE
#define DEF_ONETILEREOPT     FALSE
#define DEF_RMTILEINIT       TRUE
#define DEF_MAXFLOW          4
#define DEF_KROWEI           65
#define DEF_KCOLEI           257
#define DEF_KPARDPSI         7
#define DEF_KPERPDPSI        7
#define DEF_THRESHOLD        0.001
#define DEF_INITDZR          2048.0
#define DEF_INITDZSTEP       100.0
#define DEF_MAXCOST          1000.0
#define DEF_COSTSCALE        100.0 
#define DEF_COSTSCALEAMBIGHT 80.0 
#define DEF_DNOMINCANGLE     0.01
#define DEF_SRCROW           -1
#define DEF_SRCCOL           -1
#define DEF_P                PROBCOSTP
#define DEF_BIDIRLPN         TRUE
#define DEF_NSHORTCYCLE      200
#define DEF_MAXNEWNODECONST  0.0008
#define DEF_MAXCYCLEFRACTION 0.00001
#define DEF_NCONNNODEMIN     0
#define DEF_MAXNFLOWCYCLES   USEMAXCYCLEFRACTION
#define DEF_INITMAXFLOW      9999
#define INITMAXCOSTINCR      200
#define NOSTATINITMAXFLOW    15
#define DEF_ARCMAXFLOWCONST  3
#define DEF_DUMPALL          FALSE
#define DUMP_INITFILE        "snaphu.init"
#define DUMP_FLOWFILE        "snaphu.flow"
#define DUMP_EIFILE          "snaphu.ei"
#define DUMP_ROWCOSTFILE     "snaphu.rowcost"
#define DUMP_COLCOSTFILE     "snaphu.colcost"
#define DUMP_MSTROWCOSTFILE  "snaphu.mstrowcost"
#define DUMP_MSTCOLCOSTFILE  "snaphu.mstcolcost"
#define DUMP_MSTCOSTSFILE    "snaphu.mstcosts"
#define DUMP_CORRDUMPFILE    "snaphu.corr"
#define DUMP_RAWCORRDUMPFILE "snaphu.rawcorr"
#define INCRCOSTFILEPOS      "snaphu.incrcostpos"
#define INCRCOSTFILENEG      "snaphu.incrcostneg"
#define DEF_CS2SCALEFACTOR   8
#define DEF_NMAJORPRUNE      LARGEINT
#define DEF_PRUNECOSTTHRESH  LARGEINT
#define DEF_EDGEMASKTOP      0
#define DEF_EDGEMASKBOT      0
#define DEF_EDGEMASKLEFT     0
#define DEF_EDGEMASKRIGHT    0
#define CONNCOMPMEMINCR      1024


/* default tile parameters */

#define DEF_NTILEROW         1
#define DEF_NTILECOL         1
#define DEF_ROWOVRLP         0
#define DEF_COLOVRLP         0
#define DEF_PIECEFIRSTROW    1
#define DEF_PIECEFIRSTCOL    1
#define DEF_PIECENROW        0
#define DEF_PIECENCOL        0
#define DEF_TILECOSTTHRESH   500
#define DEF_MINREGIONSIZE    100
#define DEF_NTHREADS         1
#define DEF_SCNDRYARCFLOWMAX 8
#define DEF_TILEEDGEWEIGHT   2.5
#define DEF_TILEDIR          ""
#define DEF_ASSEMBLEONLY     FALSE
#define DEF_RMTMPTILE        TRUE


/* default connected component parameters */
#define DEF_MINCONNCOMPFRAC  0.01
#define DEF_CONNCOMPTHRESH   300
#define DEF_MAXNCOMPS        32
#define DEF_CONNCOMPOUTTYPE  CONNCOMPOUTTYPEUCHAR


/* default file formats */

#define DEF_INFILEFORMAT              COMPLEX_DATA
#define DEF_UNWRAPPEDINFILEFORMAT     ALT_LINE_DATA
#define DEF_MAGFILEFORMAT             FLOAT_DATA
#define DEF_OUTFILEFORMAT             ALT_LINE_DATA
#define DEF_CORRFILEFORMAT            ALT_LINE_DATA
#define DEF_ESTFILEFORMAT             ALT_LINE_DATA
#define DEF_AMPFILEFORMAT             ALT_SAMPLE_DATA

/* command-line usage help strings */

#define OPTIONSHELPFULL\
 "usage:  snaphu [options] infile linelength [options]\n"\
 "options:\n"\
 "  -t              use topography mode costs (default)\n"\
 "  -d              use deformation mode costs\n"\
 "  -s              use smooth-solution mode costs\n"\
 "  -C <confstr>    parse argument string as config line as from conf file\n"\
 "  -f <filename>   read configuration parameters from file\n"\
 "  -o <filename>   write output to file\n"\
 "  -a <filename>   read amplitude data from file\n"\
 "  -A <filename>   read power data from file\n"\
 "  -m <filename>   read interferogram magnitude data from file\n"\
 "  -M <filename>   read byte mask data from file\n"\
 "  -c <filename>   read correlation data from file\n"\
 "  -e <filename>   read coarse unwrapped-phase estimate from file\n"\
 "  -w <filename>   read scalar weights from file\n"\
 "  -b <decimal>    perpendicular baseline (meters, topo mode only)\n"\
 "  -p <decimal>    Lp-norm parameter p\n"\
 "  -i              do initialization and exit\n"\
 "  -n              do not use statistical costs (with -p or -i)\n"\
 "  -u              infile is already unwrapped; initialization not needed\n"\
 "  -q              quantify cost of unwrapped input file then exit\n"\
 "  -g <filename>   grow connected components mask and write to file\n"\
 "  -G <filename>   grow connected components mask for unwrapped input\n"\
 "  -S              single-tile reoptimization after multi-tile init\n"\
 "  -k              keep temporary tile outputs\n"\
 "  -l <filename>   log runtime parameters to file\n"\
 "  -v              give verbose output\n"\
 "  --mst           use MST algorithm for initialization (default)\n"\
 "  --mcf           use MCF algorithm for initialization\n"\
 "  --aa <filename1> <filename2>    read amplitude from next two files\n"\
 "  --AA <filename1> <filename2>    read power from next two files\n"\
 "  --costinfile <filename>         read statistical costs from file\n"\
 "  --costoutfile <filename>        write statistical costs to file\n"\
 "  --tile <nrow> <ncol> <rowovrlp> <colovrlp>  unwrap as nrow x ncol tiles\n"\
 "  --nproc <integer>               number of processors used in tile mode\n"\
 "  --tiledir <dirname>             use specified directory for tiles\n"\
 "  --assemble                      assemble unwrapped tiles in tiledir\n"\
 "  --piece <firstrow> <firstcol> <nrow> <ncol>  unwrap subset of image\n" \
 "  --debug, --dumpall              dump all intermediate data arrays\n"\
 "  --copyright, --info             print copyright and bug report info\n"\
 "  -h, --help                      print this help text\n"\
 "\n"

#define OPTIONSHELPBRIEF\
 "usage:  snaphu [options] infile linelength [options]\n"\
 "most common options:\n"\
 "  -t              use topography mode costs (default)\n"\
 "  -d              use deformation mode costs\n"\
 "  -s              use smooth-solution mode costs\n"\
 "  -C <confstr>    parse argument string as config line as from conf file\n"\
 "  -f <filename>   read configuration parameters from file\n"\
 "  -o <filename>   write output to file\n"\
 "  -a <filename>   read amplitude data from file\n"\
 "  -c <filename>   read correlation data from file\n"\
 "  -M <filename>   read byte mask data from file\n"\
 "  -b <decimal>    perpendicular baseline (meters)\n"\
 "  -i              do initialization and exit\n"\
 "  -S              single-tile reoptimization after multi-tile init\n"\
 "  -l <filename>   log runtime parameters to file\n"\
 "  -u              infile is already unwrapped; initialization not needed\n"\
 "  -v              give verbose output\n"\
 "  --mst           use MST algorithm for initialization (default)\n"\
 "  --mcf           use MCF algorithm for initialization\n"\
 "  --tile <nrow> <ncol> <rowovrlp> <colovrlp>  unwrap as nrow x ncol tiles\n"\
 "  --nproc <integer>               number of processors used in tile mode\n"\
 "\n"\
 "type snaphu -h for a complete list of options\n"\
 "\n"

#define COPYRIGHT\
 "Written by Curtis W. Chen\n"\
 "Copyright 2002,2017 Board of Trustees, Leland Stanford Jr. University\n"\
 "\n"\
 "Except as noted below, permission to use, copy, modify, and\n"\
 "distribute, this software and its documentation for any purpose is\n"\
 "hereby granted without fee, provided that the above copyright notice\n"\
 "appear in all copies and that both that copyright notice and this\n"\
 "permission notice appear in supporting documentation, and that the\n"\
 "name of the copyright holders be used in advertising or publicity\n"\
 "pertaining to distribution of the software with specific, written\n"\
 "prior permission, and that no fee is charged for further distribution\n"\
 "of this software, or any modifications thereof.  The copyright holder\n"\
 "makes no representations about the suitability of this software for\n"\
 "any purpose.  It is provided \"as is\" without express or implied\n"\
 "warranty.\n"\
 "\n"\
 "THE COPYRIGHT HOLDER DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS\n"\
 "SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND\n"\
 "FITNESS, IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY\n"\
 "SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER\n"\
 "RESULTING FROM LOSS OF USE, DATA, PROFITS, QPA OR GPA, WHETHER IN AN\n"\
 "ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT\n"\
 "OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.\n"\
 "\n"\
 "The parts of this software derived from the CS2 minimum cost flow\n"\
 "solver written by A. V. Goldberg and B. Cherkassky are governed by the\n"\
 "terms of the copyright holder of that software.  Permission has been\n"\
 "granted to use and distrubute that software for strictly noncommercial\n"\
 "purposes as part of this package, provided that the following\n"\
 "copyright notice from the original distribution and URL accompany the\n"\
 "software:\n"\
 "\n"\
 "  COPYRIGHT C 1995 IG Systems, Inc.  Permission to use for\n"\
 "  evaluation purposes is granted provided that proper\n"\
 "  acknowledgments are given.  For a commercial licence, contact\n"\
 "  igsys@eclipse.net (http://www.igsystems.com/cs2).\n"\
 "\n"\
 "  This software comes with NO WARRANTY, expressed or implied. By way\n"\
 "  of example, but not limitation, we make no representations of\n"\
 "  warranties of merchantability or fitness for any particular\n"\
 "  purpose or that the use of the software components or\n"\
 "  documentation will not infringe any patents, copyrights,\n"\
 "  trademarks, or other rights.\n"\
 "\n"\
 "\n"\
 "Please send snaphu bug reports to " BUGREPORTEMAIL "\n"\
 "\n"


/********************/
/* type definitions */
/********************/

/* node data structure */
typedef struct nodeST{
  int row,col;                  /* row, col of this node */
  struct nodeST *next;          /* ptr to next node in thread or bucket */
  struct nodeST *prev;          /* ptr to previous node in thread or bucket */
  struct nodeST *pred;          /* parent node in tree */
  unsigned int level;           /* tree level */
  int group;                    /* for marking label */
  int incost,outcost;           /* costs to, from root of tree */
}nodeT;


/* boundary neighbor structure */
typedef struct neighborST{
  nodeT *neighbor;              /* neighbor node pointer */
  int arcrow;                   /* row of arc to neighbor */
  int arccol;                   /* col of arc to neighbor */
  int arcdir;                   /* direction of arc to neighbor */
}neighborT;


/* boundary data structure */
typedef struct boundaryST{
  nodeT node[1];                /* ground node pointed to by this boundary */
  neighborT* neighborlist;      /* list of neighbors of common boundary */
  nodeT **boundarylist;         /* list of nodes covered by common boundary */
  long nneighbor;               /* number of neighbor nodes of boundary */
  long nboundary;               /* number of nodes covered by boundary */
}boundaryT;

  
/* arc cost data structure */
typedef struct costST{
  short offset;                 /* offset of wrapped phase gradient from 0 */
  short sigsq;                  /* variance due to decorrelation */
  short dzmax;                  /* largest discontinuity on shelf */
  short laycost;                /* cost of layover discontinuity shelf */
}costT;


/* arc cost data structure for smooth costs */
typedef struct smoothcostST{
  short offset;                 /* offset of wrapped phase gradient from 0 */
  short sigsq;                  /* variance due to decorrelation */
}smoothcostT;


/* arc cost data structure for bidirectional scalar costs */
typedef struct bidircostST{
  short posweight;              /* weight for positive flows */
  short negweight;              /* weight for negative flows */
}bidircostT;


/* incremental cost data structure */
typedef struct incrcostST{
  short poscost;                /* cost for positive flow increment */
  short negcost;                /* cost for negative flow increment */
}incrcostT;


/* arc candidate data structure */
typedef struct candidateST{
  nodeT *from, *to;             /* endpoints of candidate arc */
  long violation;               /* magnitude of arc violation */
  int arcrow,arccol;            /* indexes into arc arrays */
  signed char arcdir;           /* direction of arc (1=fwd, -1=rev) */
}candidateT;


/* bucket data structure */
typedef struct bucketST{
  long size;                    /* number of buckets in list */
  long curr;                    /* current bucket index */
  long maxind;                  /* maximum bucket index */
  long minind;                  /* smallest (possibly negative) bucket index */
  nodeT **bucket;               /* array of first nodes in each bucket */
  nodeT **bucketbase;           /* real base of bucket array */
  signed char wrapped;          /* flag denoting wrapped circular buckets */
}bucketT;


/* secondary arc data structure */
typedef struct scndryarcST{
  int arcrow;                   /* row of arc in secondary network array */
  int arccol;                   /* col of arc in secondary network array */
  nodeT *from;                  /* secondary node at tail of arc */
  nodeT *to;                    /* secondary node at head of arc */
  signed char fromdir;          /* direction from which arc enters head */
}scndryarcT;


/* supplementary data structure for secondary nodes */
typedef struct nodesuppST{
  int row;                      /* row of node in primary network problem */
  int col;                      /* col of node in primary network problem */
  nodeT **neighbornodes;        /* pointers to neighboring secondary nodes */
  scndryarcT **outarcs;         /* pointers to secondary arcs to neighbors */
  int noutarcs;                 /* number of arcs from this node */
}nodesuppT;


/* run-time parameter data structure */
typedef struct paramST{

  /* SAR and geometry parameters */
  double orbitradius;     /* radius of platform orbit (meters) */
  double altitude;        /* SAR altitude (meters) */
  double earthradius;     /* radius of earth (meters) */
  double bperp;           /* nominal perpendiuclar baseline (meters) */
  signed char transmitmode; /* transmit mode (PINGPONG or SINGLEANTTRANSMIT) */
  double baseline;        /* baseline length (meters, always postive) */
  double baselineangle;   /* baseline angle above horizontal (rad) */
  long nlooksrange;       /* number of looks in range for input data */ 
  long nlooksaz;          /* number of looks in azimuth for input data */ 
  long nlooksother;       /* number of nonspatial looks for input data */ 
  double ncorrlooks;      /* number of independent looks in correlation est */
  long ncorrlooksrange;   /* number of looks in range for correlation */ 
  long ncorrlooksaz;      /* number of looks in azimuth for correlation */ 
  double nearrange;       /* slant range to near part of swath (meters) */
  double dr;              /* range bin spacing (meters) */
  double da;              /* azimuth bin spacing (meters) */
  double rangeres;        /* range resolution (meters) */
  double azres;           /* azimuth resolution (meters) */
  double lambda;          /* wavelength (meters) */

  /* scattering model parameters */
  double kds;             /* ratio of diffuse to specular scattering */
  double specularexp;     /* power specular scattering component */
  double dzrcritfactor;   /* fudge factor for linearizing scattering model */
  signed char shadow;     /* allow discontinuities from shadowing */
  double dzeimin;         /* lower limit for backslopes (if shadow = FALSE) */
  long laywidth;          /* width of window for summing layover brightness */
  double layminei;        /* threshold brightness for assuming layover */
  double sloperatiofactor;/* fudge factor for linearized scattering slopes */
  double sigsqei;         /* variance (dz, meters) due to uncertainty in EI */

  /* decorrelation model parameters */
  double drho;            /* step size of correlation-slope lookup table */
  double rhosconst1,rhosconst2;/* for calculating rho0 in biased rho */
  double cstd1,cstd2,cstd3;/* for calculating correlation power given nlooks */
  double defaultcorr;     /* default correlation if no correlation file */
  double rhominfactor;    /* threshold for setting unbiased correlation to 0 */

  /* pdf model parameters */
  double dzlaypeak;       /* range pdf peak for no discontinuity when bright */
  double azdzfactor;      /* fraction of dz in azimuth vs. rnage */
  double dzeifactor;      /* nonlayover dz scale factor */
  double dzeiweight;      /* weight to give dz expected from intensity */
  double dzlayfactor;     /* layover regime dz scale factor */
  double layconst;        /* normalized constant pdf of layover edge */
  double layfalloffconst; /* factor of sigsq for layover cost increase */
  long sigsqshortmin;     /* min short value for costT variance */
  double sigsqlayfactor;  /* fration of ambiguityheight^2 for layover sigma */

  /* deformation mode parameters */
  double defoazdzfactor;  /* scale for azimuth ledge in defo cost function */
  double defothreshfactor;/* factor of rho0 for discontinuity threshold */
  double defomax;         /* max discontinuity (cycles) from deformation */
  double sigsqcorr;       /* variance in measured correlation */
  double defolayconst;    /* layconst for deformation mode */

  /* algorithm parameters */
  signed char eval;       /* evaluate unwrapped input file if TRUE */
  signed char unwrapped;  /* input file is unwrapped if TRUE */
  signed char regrowconncomps;/* grow connected components and exit if TRUE */
  signed char initonly;   /* exit after initialization if TRUE */
  signed char initmethod; /* MST or MCF initialization */
  signed char costmode;   /* statistical cost mode */
  signed char dumpall;    /* dump intermediate files */
  signed char verbose;    /* print verbose output */
  signed char amplitude;  /* intensity data is amplitude, not power */
  signed char havemagnitude; /* flag: create correlation from other inputs */
  signed char flipphasesign; /* flag: flip phase and flow array signs */
  signed char onetilereopt;  /* flag: reoptimize full input after tile init */
  signed char rmtileinit; /* flag to remove temporary tile unw init soln */
  long initmaxflow;       /* maximum flow for initialization */
  long arcmaxflowconst;   /* units of flow past dzmax to use for initmaxflow */
  long maxflow;           /* max flow for tree solve looping */
  long krowei, kcolei;    /* size of boxcar averaging window for mean ei */
  long kpardpsi;          /* length of boxcar for mean wrapped gradient */
  long kperpdpsi;         /* width of boxcar for mean wrapped gradient */
  double threshold;       /* thershold for numerical dzrcrit calculation */
  double initdzr;         /* initial dzr for numerical dzrcrit calc. (m) */
  double initdzstep;      /* initial stepsize for spatial decor slope calc. */
  double maxcost;         /* min and max float values for cost arrays */
  double costscale;       /* scale factor for discretizing to integer costs */
  double costscaleambight;/* ambiguity height for auto costs caling */
  double dnomincangle;    /* step size for range-varying param lookup table */
  long srcrow,srccol;     /* source node location */
  double p;               /* power for Lp-norm solution (less than 0 is MAP) */
  signed char bidirlpn;   /* use bidirectional Lp costs if TRUE */
  long nshortcycle;       /* number of points for one cycle in short int dz */
  double maxnewnodeconst; /* number of nodes added to tree on each iteration */
  long maxnflowcycles;    /* max number of cycles to consider nflow done */
  double maxcyclefraction;/* ratio of max cycles to pixels */
  long nconnnodemin;      /* min number of nodes to keep in connected set */
  long cs2scalefactor;    /* scale factor for cs2 initialization (eg, 3-30) */
  long nmajorprune;       /* number of major iterations between tree pruning */
  long prunecostthresh;   /* cost threshold for pruning */
  long edgemasktop;       /* number of pixels to mask at top edge of input */
  long edgemaskbot;       /* number of pixels to mask at bottom edge */
  long edgemaskleft;      /* number of pixels to mask at left edge */
  long edgemaskright;     /* number of pixels to mask at right edge */
  long parentpid;         /* process identification number of parent */

  /* tiling parameters */
  long ntilerow;          /* number of tiles in azimuth */
  long ntilecol;          /* number of tiles in range */
  long rowovrlp;          /* pixels of overlap between row tiles */
  long colovrlp;          /* pixels of overlap between column tiles */
  long piecefirstrow;     /* first row (indexed from 1) for piece mode */
  long piecefirstcol;     /* first column (indexed from 1) for piece mode */
  long piecenrow;         /* number of rows for piece mode */
  long piecencol;         /* number of cols for piece mode */
  long tilecostthresh;    /* maximum cost within single reliable tile region */
  long minregionsize;     /* minimum number of pixels in a region */
  long nthreads;          /* number of parallel processes to run */
  long scndryarcflowmax;  /* max flow increment for which to keep cost data */
  double tileedgeweight;  /* weight applied to tile-edge secondary arc costs */
  signed char assembleonly; /* flag for assemble-only (no unwrap) mode */
  signed char rmtmptile;  /* flag for removing temporary tile files */
  char tiledir[MAXSTRLEN];/* directory for temporary tile files */

  /* connected component parameters */
  double minconncompfrac; /* min fraction of pixels in connected component */
  long conncompthresh;    /* cost threshold for connected component */
  long maxncomps;         /* max number of connected components */
  int conncompouttype;    /* flag for type of connected component output file */
  
}paramT;


/* input file name data structure */
typedef struct infileST{
  char infile[MAXSTRLEN];             /* input interferogram */
  char magfile[MAXSTRLEN];            /* interferogram magnitude (optional) */
  char ampfile[MAXSTRLEN];            /* image amplitude or power file */
  char ampfile2[MAXSTRLEN];           /* second amplitude or power file */
  char weightfile[MAXSTRLEN];         /* arc weights */
  char corrfile[MAXSTRLEN];           /* correlation file */
  char estfile[MAXSTRLEN];            /* unwrapped estimate */
  char costinfile[MAXSTRLEN];         /* file from which cost data is read */
  char bytemaskfile[MAXSTRLEN];       /* signed char valid pixel mask */
  char dotilemaskfile[MAXSTRLEN];     /* signed char tile unwrap mask file */
  signed char infileformat;           /* input file format */
  signed char unwrappedinfileformat;  /* input file format if unwrapped */
  signed char magfileformat;          /* interferogram magnitude file format */
  signed char corrfileformat;         /* correlation file format */
  signed char weightfileformat;       /* weight file format */
  signed char ampfileformat;          /* amplitude file format */
  signed char estfileformat;          /* unwrapped-estimate file format */
}infileT;


/* output file name data structure */
typedef struct outfileST{
  char outfile[MAXSTRLEN];            /* unwrapped output */
  char initfile[MAXSTRLEN];           /* unwrapped initialization */
  char flowfile[MAXSTRLEN];           /* flows of unwrapped solution */
  char eifile[MAXSTRLEN];             /* despckled, normalized intensity */
  char rowcostfile[MAXSTRLEN];        /* statistical azimuth cost array */
  char colcostfile[MAXSTRLEN];        /* statistical range cost array */
  char mstrowcostfile[MAXSTRLEN];     /* scalar initialization azimuth costs */
  char mstcolcostfile[MAXSTRLEN];     /* scalar initialization range costs */
  char mstcostsfile[MAXSTRLEN];       /* scalar initialization costs (all) */
  char corrdumpfile[MAXSTRLEN];       /* correlation coefficient magnitude */
  char rawcorrdumpfile[MAXSTRLEN];    /* correlation coefficient magnitude */
  char conncompfile[MAXSTRLEN];       /* connected component map or mask */
  char costoutfile[MAXSTRLEN];        /* file to which cost data is written */
  char logfile[MAXSTRLEN];            /* file to which parmeters are logged */
  signed char outfileformat;          /* output file format */
}outfileT;


/* tile parameter data structure */
typedef struct tileparamST{
  long firstcol;          /* first column of tile to process (index from 0) */
  long ncol;              /* number of columns in tile to process */
  long firstrow;          /* first row of tile to process (index from 0) */
  long nrow;              /* number of rows in tile to process */
}tileparamT;


/* connectected component size structure */
typedef struct conncompsizeST{
  unsigned int tilenum;               /* tile index */
  unsigned int icomptile;             /* conn comp index in tile */
  unsigned int icompfull;             /* conn comp index in full array */
  long npix;                          /* number of pixels in conn comp */
}conncompsizeT;


/* type for total cost of solution (may overflow long) */
typedef double totalcostT;
#define INITTOTALCOST LARGEFLOAT



/***********************/
/* function prototypes */
/***********************/

/* functions in snaphu_tile.c */

int SetupTile(long nlines, long linelen, paramT *params, 
              tileparamT *tileparams, outfileT *outfiles, 
              outfileT *tileoutfiles, long tilerow, long tilecol);
signed char **SetUpDoTileMask(infileT *infiles, long ntilerow, long ntilecol);
int GrowRegions(void **costs, short **flows, long nrow, long ncol, 
                incrcostT **incrcosts, outfileT *outfiles, 
                tileparamT *tileparams, paramT *params);
int GrowConnCompsMask(void **costs, short **flows, long nrow, long ncol, 
                      incrcostT **incrcosts, outfileT *outfiles, 
                      paramT *params);
int AssembleTiles(outfileT *outfiles, paramT *params, 
                  long nlines, long linelen);


/* functions in snaphu_solver.c */

int SetGridNetworkFunctionPointers(void);
int SetNonGridNetworkFunctionPointers(void);
long TreeSolve(nodeT **nodes, nodesuppT **nodesupp, nodeT *ground, 
               nodeT *source, candidateT **candidatelistptr, 
               candidateT **candidatebagptr, long *candidatelistsizeptr,
               long *candidatebagsizeptr, bucketT *bkts, short **flows, 
               void **costs, incrcostT **incrcosts, nodeT ***apexes, 
               signed char **iscandidate, long ngroundarcs, long nflow, 
               float **mag, float **wrappedphase, char *outfile, 
               long nnoderow, int *nnodesperrow, long narcrow, 
               int *narcsperrow, long nrow, long ncol,
               outfileT *outfiles, long nconnected, paramT *params);
int InitNetwork(short **flows, long *ngroundarcsptr, long *ncycleptr, 
                long *nflowdoneptr, long *mostflowptr, long *nflowptr, 
                long *candidatebagsizeptr, candidateT **candidatebagptr, 
                long *candidatelistsizeptr, candidateT **candidatelistptr, 
                signed char ***iscandidateptr, nodeT ****apexesptr, 
                bucketT **bktsptr, long *iincrcostfileptr, 
                incrcostT ***incrcostsptr, nodeT ***nodesptr, nodeT *ground, 
                long *nnoderowptr, int **nnodesperrowptr, long *narcrowptr,
                int **narcsperrowptr, long nrow, long ncol, 
                signed char *notfirstloopptr, totalcostT *totalcostptr,
                paramT *params);
long SetupTreeSolveNetwork(nodeT **nodes, nodeT *ground, nodeT ***apexes, 
                           signed char **iscandidate, long nnoderow,
                           int *nnodesperrow, long narcrow, int *narcsperrow,
                           long nrow, long ncol);
signed char CheckMagMasking(float **mag, long nrow, long ncol);
int MaskNodes(long nrow, long ncol, nodeT **nodes, nodeT *ground, 
              float **mag);
long MaxNonMaskFlow(short **flows, float **mag, long nrow, long ncol);
int InitNodeNums(long nrow, long ncol, nodeT **nodes, nodeT *ground);
int InitNodes(long nrow, long ncol, nodeT **nodes, nodeT *ground);
void BucketInsert(nodeT *node, long ind, bucketT *bkts);
void BucketRemove(nodeT *node, long ind, bucketT *bkts);
nodeT *ClosestNode(bucketT *bkts);
long SelectSources(nodeT **nodes, float **mag, nodeT *ground, long nflow, 
                   short **flows, long ngroundarcs, 
                   long nrow, long ncol, paramT *params,
                   nodeT ***sourcelistptr, long **nconnectedarrptr);
long ReCalcCost(void **costs, incrcostT **incrcosts, long flow, 
                long arcrow, long arccol, long nflow, long nrow, 
                paramT *params);
int SetupIncrFlowCosts(void **costs, incrcostT **incrcosts, short **flows,
                       long nflow, long nrow, long narcrow, 
                       int *narcsperrow, paramT *params);
totalcostT EvaluateTotalCost(void **costs, short **flows, long nrow, long ncol,
                             int *narcsperrow,paramT *params);
int MSTInitFlows(float **wrappedphase, short ***flowsptr, 
                 short **mstcosts, long nrow, long ncol, 
                 nodeT ***nodes, nodeT *ground, long maxflow);
int MCFInitFlows(float **wrappedphase, short ***flowsptr, short **mstcosts, 
                 long nrow, long ncol, long cs2scalefactor);


/* functions in snaphu_cost.c */
int BuildCostArrays(void ***costsptr, short ***mstcostsptr, 
                    float **mag, float **wrappedphase, 
                    float **unwrappedest, long linelen, long nlines, 
                    long nrow, long ncol, paramT *params, 
                    tileparamT *tileparams, infileT *infiles, 
                    outfileT *outfiles);
void CalcCostTopo(void **costs, long flow, long arcrow, long arccol, 
                  long nflow, long nrow, paramT *params, 
                  long *poscostptr, long *negcostptr);
void CalcCostDefo(void **costs, long flow, long arcrow, long arccol, 
                  long nflow, long nrow, paramT *params, 
                  long *poscostptr, long *negcostptr);
void CalcCostSmooth(void **costs, long flow, long arcrow, long arccol, 
                    long nflow, long nrow, paramT *params, 
                    long *poscostptr, long *negcostptr);
void CalcCostL0(void **costs, long flow, long arcrow, long arccol, 
                long nflow, long nrow, paramT *params, 
                long *poscostptr, long *negcostptr);
void CalcCostL1(void **costs, long flow, long arcrow, long arccol, 
                long nflow, long nrow, paramT *params, 
                long *poscostptr, long *negcostptr);
void CalcCostL2(void **costs, long flow, long arcrow, long arccol, 
                long nflow, long nrow, paramT *params, 
                long *poscostptr, long *negcostptr);
void CalcCostLP(void **costs, long flow, long arcrow, long arccol, 
                long nflow, long nrow, paramT *params, 
                long *poscostptr, long *negcostptr);
void CalcCostL0BiDir(void **costs, long flow, long arcrow, long arccol, 
                     long nflow, long nrow, paramT *params, 
                     long *poscostptr, long *negcostptr);
void CalcCostL1BiDir(void **costs, long flow, long arcrow, long arccol, 
                     long nflow, long nrow, paramT *params, 
                     long *poscostptr, long *negcostptr);
void CalcCostL2BiDir(void **costs, long flow, long arcrow, long arccol, 
                     long nflow, long nrow, paramT *params, 
                     long *poscostptr, long *negcostptr);
void CalcCostLPBiDir(void **costs, long flow, long arcrow, long arccol, 
                     long nflow, long nrow, paramT *params, 
                     long *poscostptr, long *negcostptr);
void CalcCostNonGrid(void **costs, long flow, long arcrow, long arccol, 
                     long nflow, long nrow, paramT *params, 
                     long *poscostptr, long *negcostptr);
long EvalCostTopo(void **costs, short **flows, long arcrow, long arccol,
                  long nrow, paramT *params);
long EvalCostDefo(void **costs, short **flows, long arcrow, long arccol,
                  long nrow, paramT *params);
long EvalCostSmooth(void **costs, short **flows, long arcrow, long arccol,
                    long nrow, paramT *params);
long EvalCostL0(void **costs, short **flows, long arcrow, long arccol,
                long nrow, paramT *params);
long EvalCostL1(void **costs, short **flows, long arcrow, long arccol,
                long nrow, paramT *params);
long EvalCostL2(void **costs, short **flows, long arcrow, long arccol,
                long nrow, paramT *params);
long EvalCostLP(void **costs, short **flows, long arcrow, long arccol,
                long nrow, paramT *params);
long EvalCostL0BiDir(void **costs, short **flows, long arcrow, long arccol,
                     long nrow, paramT *params);
long EvalCostL1BiDir(void **costs, short **flows, long arcrow, long arccol,
                     long nrow, paramT *params);
long EvalCostL2BiDir(void **costs, short **flows, long arcrow, long arccol,
                     long nrow, paramT *params);
long EvalCostLPBiDir(void **costs, short **flows, long arcrow, long arccol,
                     long nrow, paramT *params);
long EvalCostNonGrid(void **costs, short **flows, long arcrow, long arccol, 
                     long nrow, paramT *params);


/* functions in snaphu_util.c */

signed char SetBooleanSignedChar(signed char *boolptr, char *str);
int WrapPhase(float **wrappedphase, long nrow, long ncol);
int CalcWrappedRangeDiffs(float **dpsi, float **avgdpsi, float **wrappedphase,
                          long kperpdpsi, long kpardpsi,
                          long nrow, long ncol);
int CalcWrappedAzDiffs(float **dpsi, float **avgdpsi, float **wrappedphase,
                       long kperpdpsi, long kpardpsi, long nrow, long ncol);
int CycleResidue(float **phase, signed char **residue, 
                 int nrow, int ncol);
int NodeResidue(float **wphase, long row, long col);
int CalcFlow(float **phase, short ***flowsptr, long nrow, long ncol);
int IntegratePhase(float **psi, float **phi, short **flows,
                   long nrow, long ncol);
float **ExtractFlow(float **unwrappedphase, short ***flowsptr, 
                    long nrow, long ncol);
int FlipPhaseArraySign(float **arr, paramT *params, long nrow, long ncol);
int FlipFlowArraySign(short **arr, paramT *params, long nrow, long ncol);
void **Get2DMem(int nrow, int ncol, int psize, size_t size);
void **Get2DRowColMem(long nrow, long ncol, int psize, size_t size);
void **Get2DRowColZeroMem(long nrow, long ncol, int psize, size_t size);
void *MAlloc(size_t size);
void *CAlloc(size_t nitems, size_t size);
void *ReAlloc(void *ptr, size_t size);
int Free2DArray(void **array, unsigned int nrow);
int Set2DShortArray(short **arr, long nrow, long ncol, long value);
signed char ValidDataArray(float **arr, long nrow, long ncol);
signed char NonNegDataArray(float **arr, long nrow, long ncol);
signed char IsFinite(double d);
long LRound(double a);
long LMin(long a, long b);
long LClip(long a, long minval, long maxval);
long Short2DRowColAbsMax(short **arr, long nrow, long ncol);
float LinInterp1D(float *arr, double index, long nelem);
float LinInterp2D(float **arr, double rowind, double colind , 
                  long nrow, long ncol);
int Despeckle(float **mag, float ***ei, long nrow, long ncol);
float **MirrorPad(float **array1, long nrow, long ncol, long krow, long kcol);
int BoxCarAvg(float **avgarr, float **padarr, long nrow, long ncol, 
              long krow, long kcol);
char *StrNCopy(char *dest, const char *src, size_t n);
int FlattenWrappedPhase(float **wrappedphase, float **unwrappedest, 
                        long nrow, long ncol);
int Add2DFloatArrays(float **arr1, float **arr2, long nrow, long ncol);
int StringToDouble(char *str, double *d);
int StringToLong(char *str, long *l);
int CatchSignals(void (*SigHandler)(int));
void SetDump(int signum);
void KillChildrenExit(int signum);
void SignalExit(int signum);
int StartTimers(time_t *tstart, double *cputimestart);
int DisplayElapsedTime(time_t tstart, double cputimestart);
int LongCompare(const void *c1, const void *c2);

/* functions in snaphu_io.c */

int SetDefaults(infileT *infiles, outfileT *outfiles, paramT *params);
int ProcessArgs(int argc, char *argv[], infileT *infiles, outfileT *outfiles,
                long *ncolptr, paramT *params);
int CheckParams(infileT *infiles, outfileT *outfiles, 
                long linelen, long nlines, paramT *params);
int ReadConfigFile(char *conffile, infileT *infiles, outfileT *outfiles,
                   long *ncolptr, paramT *params);
int WriteConfigLogFile(int argc, char *argv[], infileT *infiles, 
                       outfileT *outfiles, long linelen, paramT *params);
long GetNLines(infileT *infiles, long linelen, paramT *params);
int WriteOutputFile(float **mag, float **unwrappedphase, char *outfile, 
                    outfileT *outfiles, long nrow, long ncol);
FILE *OpenOutputFile(char *outfile, char *realoutfile);
int Write2DArray(void **array, char *filename, long nrow, long ncol, 
                 size_t size);
int Write2DRowColArray(void **array, char *filename, long nrow, 
                       long ncol, size_t size);
int ReadInputFile(infileT *infiles, float ***magptr, float ***wrappedphaseptr,
                  short ***flowsptr, long linelen, long nlines, 
                  paramT *params, tileparamT *tileparams);
int ReadMagnitude(float **mag, infileT *infiles, long linelen, long nlines, 
                  tileparamT *tileparams);
int ReadByteMask(float **mag, infileT *infiles, long linelen, long nlines, 
                 tileparamT *tileparams, paramT *params);
int ReadUnwrappedEstimateFile(float ***unwrappedestptr, infileT *infiles, 
                              long linelen, long nlines, 
                              paramT *params, tileparamT *tileparams);
int ReadWeightsFile(short ***weightsptr,char *weightfile, 
                    long linelen, long nlines, tileparamT *tileparams);
int ReadIntensity(float ***pwrptr, float ***pwr1ptr, float ***pwr2ptr, 
                  infileT *infiles, long linelen, long nlines, 
                  paramT *params, tileparamT *tileparams);
int ReadCorrelation(float ***corrptr, infileT *infiles,
                    long linelen, long nlines, tileparamT *tileparams);
int ReadAltLineFile(float ***mag, float ***phase, char *alfile, 
                    long linelen, long nlines, tileparamT *tileparams);
int ReadAltLineFilePhase(float ***phase, char *alfile, 
                         long linelen, long nlines, tileparamT *tileparams);
int ReadComplexFile(float ***mag, float ***phase, char *rifile, 
                    long linelen, long nlines, tileparamT *tileparams);
int Read2DArray(void ***arr, char *infile, long linelen, long nlines, 
                tileparamT *tileparams, size_t elptrsize, size_t elsize);
int ReadAltSampFile(float ***arr1, float ***arr2, char *infile,
                     long linelen, long nlines, tileparamT *tileparams);
int Read2DRowColFile(void ***arr, char *filename, long linelen, long nlines, 
                     tileparamT *tileparams, size_t size);
int Read2DRowColFileRows(void ***arr, char *filename, long linelen, 
                         long nlines, tileparamT *tileparams, size_t size);
int SetDumpAll(outfileT *outfiles, paramT *params);
int SetStreamPointers(void);
int SetVerboseOut(paramT *params);
int ChildResetStreamPointers(pid_t pid, long tilerow, long tilecol,
                             paramT *params);
int DumpIncrCostFiles(incrcostT **incrcosts, long iincrcostfile, 
                      long nflow, long nrow, long ncol);
int MakeTileDir(paramT *params, outfileT *outfiles);
int ParseFilename(char *filename, char *path, char *basename);
int SetTileInitOutfile(char *outfile, long pid);


/* functions in snaphu_cs2.c  */

void SolveCS2(signed char **residue, short **mstcosts, long nrow, long ncol, 
              long cs2scalefactor, short ***flowsptr);



/*******************************************/
/* global (external) variable declarations */
/*******************************************/

/* flags used for signal handling */
extern char dumpresults_global;
extern char requestedstop_global;

/* ouput stream pointers */
/* sp0=error messages, sp1=status output, sp2=verbose, sp3=verbose counter */
extern FILE *sp0, *sp1, *sp2, *sp3;

/* node pointer for marking arc not on tree in apex array */
/* this should be treat as a constant */
extern nodeT NONTREEARC[1];

/* pointers to functions which calculate arc costs */
extern void (*CalcCost)(void **, long, long, long, long, long,
                        paramT *, long *, long *);
extern long (*EvalCost)(void **, short **, long, long, long, paramT *);

/* end of snaphu.h */




