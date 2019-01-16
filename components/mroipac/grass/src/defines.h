// From grass.c
#define SCROLL_WIDTH	21	/* width of scrollbar */
#define WIN_WIDTH_MAX 	960
#define WIN_HEIGHT_MAX	768
#define LR_NORMAL 1		/* normal display */
#define LR_REV	 -1		/* display reverse of each line, left to right */

#define PLUS 1
#define MINU 2
#define CHG  3
#define GUID 4
#define LSNR 8
#define VIST 16
#define BRPT 32
#define CUT  64
#define LAWN 128
#define TREE 128

#define REL_BEGIN 0		/* fseek relative to beginning of file */
#define REL_CUR   1		/* fseek relative to beginning of file */
#define REL_EOF   2		/* fseek relative to end of file */

#define Max(a,b)  ( ( (a) > (b) ) ? (a) : (b) )
#define Min(a,b)  ( ( (a) < (b) ) ? (a) : (b) )
#define Abs(a)    ( ((a) > 0) ? (a) : (-a) )
#define SQR(a)	  ( (a)*(a) )
#define nint(a)  ( ( (a) > (0.0) ) ? ((int)(a+0.5)) : ((int)(a-0.5)) )

#define PI	3.1415926535 
#define TWO_PI  6.283185308

#define MAX_CRAB 200000			/* maximum size of ping-pong list for growth of crabgrass */
// From trees.c
#define PLUS 1
#define MINU 2
#define CHG  3
#define GUID 4
#define LSNR 8
#define VIST 16
#define BRPT 32
#define CUT  64
#define LAWN 128
#define TREE 128

#define REL_BEGIN 0		/* fseek relative to beginning of file */
#define REL_CUR   1		/* fseek relative to current position */
#define REL_EOF   2		/* fseek relative to end of file */

#define Max(a,b)  ( ( (a) > (b) ) ? (a) : (b) )
#define Min(a,b)  ( ( (a) < (b) ) ? (a) : (b) )
#define Abs(a)    ( ((a) > 0) ? (a) : (-a) )

#define PI	3.1415926535 
#define TWO_PI  6.283185308
#define RTD	57.2957795131	/* radians to degrees */
#define DTR	.0174532925199	/* degrees to radians */
#define C	2.99792458e8
// From residue.c
//#define nint(a) ( ((nintarg=(a)) >= 0.0 )?(int)(nintarg+0.5):(int)(nintarg-0.5) )

#define PLUS 1
#define MINU 2
#define CHG  3
#define GUID 4
#define LSNR 8
#define VIST 16
#define BRPT 32
#define CUT  64
#define LAWN 128
#define TREE 128

#define REL_BEGIN 0		/* fseek relative to beginning of file */
#define REL_CUR   1		/* fseek relative to current position */
#define REL_EOF   2		/* fseek relative to end of file */
#define NEW_FILE  1		/* new flag file */
#define OLD_FILE  0		/* old flag file */

#define Max(a,b)  ( ( (a) > (b) ) ? (a) : (b) )
#define Min(a,b)  ( ( (a) < (b) ) ? (a) : (b) )
#define Abs(a)    ( ((a) > 0) ? (a) : (-a) )

#define PI	3.1415926535 
#define TWO_PI  6.283185308
#define RTD	57.2957795131	/* radians to degrees */
#define DTR	.0174532925199	/* degrees to radians */
#define C	2.99792458e8
// From corr_flag.c
#define PLUS 1
#define MINU 2
#define CHG  3
#define GUID 4
#define LSNR 8
#define VIST 16
#define BRPT 32
#define CUT  64
#define LAWN 128
#define TREE 128

#define REL_BEGIN 0		/* fseek relative to beginning of file */
#define REL_CUR   1		/* fseek relative to current position */
#define REL_EOF   2		/* fseek relative to end of file */

#define Max(a,b)  ( ( (a) > (b) ) ? (a) : (b) )
#define Min(a,b)  ( ( (a) < (b) ) ? (a) : (b) )
#define Abs(a)    ( ((a) > 0) ? (a) : (-a) )

#define PI	3.1415926535 
#define TWO_PI  6.283185308
// From phase_slope.c
#define MIN(a,b)  ( ( (a) < (b) ) ? (a) : (b) )
