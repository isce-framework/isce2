/*****************************************************
   cal2ut1.c

   Convert calendar date/time to UT1 seconds 
   after 01-01-2000 12:00:00

   Reference: "A Simple and Precise Approach to 
               Position and Velocity Estimation
               of Low Earth Orbit Satellites"

   14-April-2010     Jeff Bytof
   
*****************************************************/

#include <stdio.h>

double cal2ut1( int mode, int cal[3], double daysec ) 
{ 
   double days;
//   double deltaU =  0.0 ;     /* needed for UT1 - refine  */
//   double deltaU =  +0.1 ;   /* for 2006-09-28	0000 UTC  */
   double deltaU =  -0.0811 ;   /* for minimum residuals  */
   double sec;
   double ut1sec;

   int monthDays[] = {0,31,59,90,120,151,181,212,243,273,304,334};   
   int monthDayLeap[] = {0,31,60,91,121,152,182,213,244,274,305,335};

   int day;
   int doy;
   int month; 
   int year;

   int years[]={-7305, -6939, -6574, -6209, -5844,      /* 1980 to 2060 */
                -5478, -5113, -4748, -4383, -4017,
                -3652, -3287, -2922, -2556, -2191,
                -1826, -1461, -1095,  -730,  -365,
                    0,   366,   731,  1096,  1461,
                 1827,  2192,  2557,  2922,  3288,
                 3653,  4018,  4383,  4749,  5114,
                 5479,  5844,  6210,  6575,  6940,
                 7305,  7671,  8036,  8401,  8766,
                 9132,  9497,  9862, 10227, 10593, 
                10958, 11323, 11688, 12054, 12419,
                12784, 13149, 13515, 13880, 14245,
                14610, 14976, 15341, 15706, 16071,
                16437, 16802, 17167, 17532, 17898,
                18263, 18628, 18993, 19359, 19724,
                20089, 20454, 20820, 21185, 21550,
                21915 };

   int leaps[]={ 1,0,0,0,1,0,0,0,1,0,     /* 1980 to 2060 */
                 0,0,1,0,0,0,1,0,0,0,
                 1,0,0,0,1,0,0,0,1,0,
                 0,0,1,0,0,0,1,0,0,0,
                 1,0,0,0,1,0,0,0,1,0,
                 0,0,1,0,0,0,1,0,0,0,
                 1,0,0,0,1,0,0,0,1,0,
                 0,0,1,0,0,0,1,0,0,0,
                 1  };


   if( mode == 1 ) {     /*  year, month, day of month  */ 

      year = cal[0];
      month = cal[1];
      day = cal[2];

      days = years[ year-1980 ] - 0.5;

      if( leaps[ year-1980] == 1 ) {
         days = days + monthDayLeap[month-1];
      } else { 
         days = days + monthDays[month-1];
      }

      days = days + day - 1;
   
   } else if( mode == 2 ) {    /* year, day of year */

      year = cal[0];
      doy = cal[1];

      days = years[ year-1980] - 0.5;
      days = days + doy - 1;
   }

   sec = days*86400.0 + daysec; 

   ut1sec = sec + deltaU; 

   return ut1sec;
} 
/**********************************************
   eci2ecr.c

   Convert position and velocity vectors in

   Inertial Earth Coordinates (ECI) 
           -to-
   Rotating Earth Coordinates (ECR).

Inputs
------
   double pos[3] = ECI position vector (meters)
   double vel[3] = ECI velocity vector (meters/sec) 
   double utsec = UT seconds past 1-JAN-2000 12:00:00

Outputs
-------
   double pos_ecr[3] = ECR position vector (meters)
   double vel_ecr[3] = ECR velocity vector (meters/sec)

-------------------------------------------
Reference: 

"A Simple and Precise Approach to Position 
and Velocity Estimation of Low Earth Orbit 
Satellites"

Authors: P. Beaulne and I. Sikaneta

Defence R&D Canada Ottowa TM 2005-250
-------------------------------------------

   5 March 2010     Jeff Bytof

**********************************************/

void gmst( double, double *, double *);
void matvec( double [3][3], double [3], double [3] );

#include <math.h>

void eci2ecr( double pos[], double vel[], double utsec, 
              double pos_ecr[], double vel_ecr[] )
{
   double  a[3][3];
   double  ap[3][3];
   double  cth;
   double  cthp;
   double  sth;
   double  sthp;
   double  th;
   double  thp;
   double  vel_ecr_1[3];
   double  vel_ecr_2[3];

   int  i;

   gmst( utsec, &th, &thp );

   cth = cos(th);
   sth = sin(th);

   a[0][0] = cth;
   a[0][1] = sth;
   a[0][2] = 0.0;
   a[1][0] = -sth;
   a[1][1] = cth;
   a[1][2] = 0.0;
   a[2][0] = 0.0;
   a[2][1] = 0.0;
   a[2][2] = 1.0;

   matvec( a, pos, pos_ecr );
 
   cthp = thp*cos(th);
   sthp = thp*sin(th); 

   ap[0][0] = -sthp;
   ap[0][1] = cthp;
   ap[0][2] = 0.0;
   ap[1][0] = -cthp;
   ap[1][1] = -sthp;
   ap[1][2] = 0.0;
   ap[2][0] = 0.0;
   ap[2][1] = 0.0;
   ap[2][2] = 0.0;
 
   matvec( ap, pos, vel_ecr_1 );
   matvec( a, vel, vel_ecr_2 );

   for( i=0; i<3; i++ ) { 
      vel_ecr[i] = vel_ecr_1[i] + vel_ecr_2[i];   
   } 

   return;

}
/**********************************************
   ecr2eci.c

   Transform position and velocity vectors in 

   Rotating Earth Coordinates (ECR)
         -to-
   Inertial Earth Coordinates (ECI).

Inputs
------
   double pos[3] = ECR position vector (meters)
   double vel[3] = ECR velocity vector (meters/sec)
   double utsec = UT seconds past 1-JAN-2000 12:00:00

Outputs
-------
   double pos_eci[3] = ECI position vector (meters)
   double vel_eci[3] = ECI velocity vector (meters/sec) 

-------------------------------------------
Reference: 

"A Simple and Precise Approach to Position 
and Velocity Estimation of Low Earth Orbit 
Satellites"

Authors: P. Beaulne and I. Sikaneta

Defence R&D Canada Ottowa TM 2005-250
-------------------------------------------

   26-April-2010     Jeff Bytof

**********************************************/

void gmst( double, double *, double *);
void matvec( double [3][3], double [3], double [3] );

#include <math.h>

void ecr2eci( double pos[], double vel[], double utsec, 
              double pos_eci[], double vel_eci[] )
{
   double  a[3][3];
   double  ap[3][3];
   double  cth;
   double  cthp;
   double  sth;
   double  sthp;
   double  th;
   double  thp;
   double  vel_eci_1[3];
   double  vel_eci_2[3];

   int  i;

   gmst( utsec, &th, &thp );

   cth = cos(th);
   sth = sin(th);

   a[0][0] = cth;
   a[0][1] = -sth;
   a[0][2] = 0.0;
   a[1][0] = sth;
   a[1][1] = cth;
   a[1][2] = 0.0;
   a[2][0] = 0.0;
   a[2][1] = 0.0;
   a[2][2] = 1.0;

   matvec( a, pos, pos_eci );
 
   cthp = thp*cos(th);
   sthp = thp*sin(th); 

   ap[0][0] = -sthp;
   ap[0][1] = -cthp;
   ap[0][2] = 0.0;
   ap[1][0] = cthp;
   ap[1][1] = -sthp; 
   ap[1][2] = 0.0;
   ap[2][0] = 0.0;
   ap[2][1] = 0.0;
   ap[2][2] = 0.0;
 
   matvec( ap, pos, vel_eci_1 );
   matvec( a, vel,  vel_eci_2 );

   for( i=0; i<3; i++ ) { 
      vel_eci[i] = vel_eci_1[i] + vel_eci_2[i];   
   } 

   return;

}
/**********************************************
    gmst.c

    Calculate the Greenwich mean sidereal angle
    and its first time derivative.

-------------------------------------------
Reference: 

"A Simple and Precise Approach to Position 
and Velocity Estimation of Low Earth Orbit 
Satellites"

Authors: P. Beaulne and I. Sikaneta

Defence R&D Canada Ottowa TM 2005-250
-------------------------------------------

    March 2010    Jeff Bytof

***********************************************/

#include <math.h>

void gmst( double julsec, double *th,  double *thp )
{
   double a0 = 67310.54841;
   double a1 = 3164400184.812866;   /* 876600*3600+8640184.812866 */ 
   double a2 = 0.093104;
   double a3 = -6.2e-6;

   double rpd =  0.01745329251994329444;
   double sigma;
   double t;
   double twopi =  6.283185307179586 ;

   t = ( julsec/86400.0 ) / 36525.0;   /* convert to centuries */

   sigma = a0 + a1*t + a2*t*t + a3*t*t*t;
   sigma = sigma/240.0;      /* 240 = 360/86400 */
   sigma = sigma * rpd;

   *th = fmod( sigma, twopi );

   sigma = a1 + 2.*a2*t + 3.*a3*t*t;
   sigma = sigma/240.0;
   sigma = sigma/(36525.*86400.);
   *thp = sigma * rpd;

   return;
}
/****************************************
    matvec.c

    Multiply a matrix and a vector
    and return the product vector.

    March 2010    Jeff Bytof

*****************************************/ 

void matvec( double mat[3][3], double vin[3], double vout[3] )
{
   int  i;
   int  j;

   for( j=0; j<3; j++ ) { 
      vout[j] = 0.0;
      for( i=0; i<3; i++ ) {  
         vout[j] = vout[j] + mat[j][i]*vin[i];
      }
   }

   return;
}
/*---------------------------------------------------------------*/
