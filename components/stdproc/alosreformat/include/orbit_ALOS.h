/* alos_orbit.h */
/* structure to hold orbit and attitude information derived from ALOS L1.0 LED-file */

#define ND 28		/* number of orbit data points	*/
#define NA 64		/* number of altitude data points	*/
#define HDR	1	/* orbit information from header */
#define ODR	2	/* orbit information from Delft */
#define DOR	3	/* orbit information from Doris */

struct ORB_XYZ {
	double pt;
	double px;
	double py;
	double pz;
	double vx;
	double vy;
	double vz;
	};

struct ALOS_ORB {
	int	itype;
	int 	nd;
	int 	iy;
	int 	id;
	double sec;
	double dsec;
	double pt0;
	struct ORB_XYZ *points;
}; 

struct ALOS_ATT {
	int  na;
	int  id[NA];
	int  msec[NA];
	double ap[NA];
	double ar[NA];
	double ay[NA];
	double dp[NA];
	double dr[NA];
	double dy[NA];
}; 
