/*********************************************************************** 

  This code is derived from cs2 v3.7
  Written by Andrew V. Goldberg and Boris Cherkassky
  Modifications for use in snaphu by Curtis W. Chen 

  The cs2 code is used here with permission for strictly noncommerical
  use.  The original cs2 source code can be downloaded from
 
    http://www.igsystems.com/cs2

  The original cs2 copyright is stated as follows:

    COPYRIGHT C 1995 IG Systems, Inc.  Permission to use for
    evaluation purposes is granted provided that proper
    acknowledgments are given.  For a commercial licence, contact
    igsys@eclipse.net.
    
    This software comes with NO WARRANTY, expressed or implied. By way
    of example, but not limitation, we make no representations of
    warranties of merchantability or fitness for any particular
    purpose or that the use of the software components or
    documentation will not infringe any patents, copyrights,
    trademarks, or other rights.

  Copyright 2002 Board of Trustees, Leland Stanford Jr. University

*************************************************************************/

/* min-cost flow */
/* successive approximation algorithm */
/* Copyright C IG Systems, igsys@eclipse.com */
/* any use except for evaluation purposes requires a licence */

/* parser changed to take input from passed data */
/* main() changed to callable function */
/* outputs parsed as flow */
/* functions made static */
/* MAX and MIN macros renamed GREATEROF and LESSEROF */

#ifndef NO_CS2

/************************************** constants  &  parameters ********/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "snaphu.h"

/* for measuring time */

/* definitions of types: node & arc */

#define PRICE_MAX           1e30
#define BIGGEST_FLOW        LARGESHORT

#include "snaphu_cs2types.h"

/* parser for getting  DIMACS format input and transforming the
   data to the internal representation */

#include "snaphu_cs2parse.c"


#define N_NODE( i ) ( ( (i) == NULL ) ? -1 : ( (i) - ndp + nmin ) )
#define N_ARC( a ) ( ( (a) == NULL )? -1 : (a) - arp )


#define UNFEASIBLE          2
#define ALLOCATION_FAULT    5
#define PRICE_OFL           6

/* parameters */

#define UPDT_FREQ      0.4
#define UPDT_FREQ_S    30

#define SCALE_DEFAULT  12.0

/* PRICE_OUT_START may not be less than 1 */
#define PRICE_OUT_START 1

#define CUT_OFF_POWER    0.44
#define CUT_OFF_COEF     1.5
#define CUT_OFF_POWER2   0.75
#define CUT_OFF_COEF2    1
#define CUT_OFF_GAP      0.8
#define CUT_OFF_MIN      12
#define CUT_OFF_INCREASE 4

/*
#define TIME_FOR_PRICE_IN    5
*/
#define TIME_FOR_PRICE_IN1    2
#define TIME_FOR_PRICE_IN2    4
#define TIME_FOR_PRICE_IN3    6

#define EMPTY_PUSH_COEF      1.0
/*
#define MAX_CYCLES_CANCELLED 10
#define START_CYCLE_CANCEL   3
*/
#define MAX_CYCLES_CANCELLED 0
#define START_CYCLE_CANCEL   100
/************************************************ shared macros *******/

#define GREATEROF( x, y ) ( ( (x) > (y) ) ?  x : y )
#define LESSEROF( x, y ) ( ( (x) < (y) ) ? x : y )

#define OPEN( a )   ( a -> r_cap > 0 )
#define CLOSED( a )   ( a -> r_cap <= 0 )
#define REDUCED_COST( i, j, a ) ( (i->price) + dn*(a->cost) - (j->price) )
#define FEASIBLE( i, j, a )     ( (i->price) + dn*(a->cost) < (j->price) )
#define ADMISSIBLE( i, j, a )   ( OPEN(a) && FEASIBLE( i, j, a ) )


#define INCREASE_FLOW( i, j, a, df )\
{\
   (i) -> excess            -= df;\
   (j) -> excess            += df;\
   (a)            -> r_cap  -= df;\
  ((a) -> sister) -> r_cap  += df;\
}\

/*---------------------------------- macros for excess queue */

#define RESET_EXCESS_Q \
{\
   for ( ; excq_first != NULL; excq_first = excq_last )\
     {\
	excq_last            = excq_first -> q_next;\
        excq_first -> q_next = sentinel_node;\
     }\
}

#define OUT_OF_EXCESS_Q( i )  ( i -> q_next == sentinel_node )

#define EMPTY_EXCESS_Q    ( excq_first == NULL )
#define NONEMPTY_EXCESS_Q ( excq_first != NULL )

#define INSERT_TO_EXCESS_Q( i )\
{\
   if ( NONEMPTY_EXCESS_Q )\
     excq_last -> q_next = i;\
   else\
     excq_first  = i;\
\
   i -> q_next = NULL;\
   excq_last   = i;\
}

#define INSERT_TO_FRONT_EXCESS_Q( i )\
{\
   if ( EMPTY_EXCESS_Q )\
     excq_last = i;\
\
   i -> q_next = excq_first;\
   excq_first  = i;\
}

#define REMOVE_FROM_EXCESS_Q( i )\
{\
   i           = excq_first;\
   excq_first  = i -> q_next;\
   i -> q_next = sentinel_node;\
}

/*---------------------------------- excess queue as a stack */

#define EMPTY_STACKQ      EMPTY_EXCESS_Q
#define NONEMPTY_STACKQ   NONEMPTY_EXCESS_Q

#define RESET_STACKQ  RESET_EXCESS_Q

#define STACKQ_PUSH( i )\
{\
   i -> q_next = excq_first;\
   excq_first  = i;\
}

#define STACKQ_POP( i ) REMOVE_FROM_EXCESS_Q( i )

/*------------------------------------ macros for buckets */

node dnd, *dnode;

#define RESET_BUCKET( b )  ( b -> p_first ) = dnode;

#define INSERT_TO_BUCKET( i, b )\
{\
i -> b_next                  = ( b -> p_first );\
( b -> p_first ) -> b_prev   = i;\
( b -> p_first )             = i;\
}

#define NONEMPTY_BUCKET( b ) ( ( b -> p_first ) != dnode )

#define GET_FROM_BUCKET( i, b )\
{\
i    = ( b -> p_first );\
( b -> p_first ) = i -> b_next;\
}

#define REMOVE_FROM_BUCKET( i, b )\
{\
if ( i == ( b -> p_first ) )\
       ( b -> p_first ) = i -> b_next;\
  else\
    {\
       ( i -> b_prev ) -> b_next = i -> b_next;\
       ( i -> b_next ) -> b_prev = i -> b_prev;\
    }\
}

/*------------------------------------------- misc macros */

#define UPDATE_CUT_OFF \
{\
   if (n_bad_pricein + n_bad_relabel == 0) \
     {\
	cut_off_factor = CUT_OFF_COEF2 * pow ( (double)n, CUT_OFF_POWER2 );\
        cut_off_factor = GREATEROF ( cut_off_factor, CUT_OFF_MIN );\
        cut_off        = cut_off_factor * epsilon;\
        cut_on         = cut_off * CUT_OFF_GAP;\
      }\
     else\
       {\
	cut_off_factor *= CUT_OFF_INCREASE;\
        cut_off        = cut_off_factor * epsilon;\
        cut_on         = cut_off * CUT_OFF_GAP;\
	}\
}

#define TIME_FOR_UPDATE \
( n_rel > n * UPDT_FREQ + n_src * UPDT_FREQ_S )

#define FOR_ALL_NODES_i        for ( i = nodes; i != sentinel_node; i ++ )

#define FOR_ALL_ARCS_a_FROM_i \
for ( a = i -> first, a_stop = ( i + 1 ) -> suspended; a != a_stop; a ++ )

#define FOR_ALL_CURRENT_ARCS_a_FROM_i \
for ( a = i -> current, a_stop = ( i + 1 ) -> suspended; a != a_stop; a ++ )

#define WHITE 0
#define GREY  1
#define BLACK 2

arc     *sa, *sb;
long    d_cap;

#define EXCHANGE( a, b )\
{\
if ( a != b )\
  {\
     sa = a -> sister;\
     sb = b -> sister;\
\
     d_arc.r_cap = a -> r_cap;\
     d_arc.cost  = a -> cost;\
     d_arc.head  = a -> head;\
\
     a -> r_cap  = b -> r_cap;\
     a -> cost   = b -> cost;\
     a -> head   = b -> head;\
\
     b -> r_cap  = d_arc.r_cap;\
     b -> cost   = d_arc.cost;\
     b -> head   = d_arc.head;\
\
     if ( a != sb )\
       {\
	  b -> sister = sa;\
	  a -> sister = sb;\
	  sa -> sister = b;\
	  sb -> sister = a;\
        }\
\
     d_cap       = cap[a-arcs];\
     cap[a-arcs] = cap[b-arcs];\
     cap[b-arcs] = d_cap;\
  }\
}

#define SUSPENDED( i, a ) ( a < i -> first ) 



long n_push      =0,
     n_relabel   =0,
     n_discharge =0,
     n_refine    =0,
     n_update    =0,
     n_scan      =0,
     n_prscan    =0,
     n_prscan1   =0,
     n_prscan2   =0,
     n_bad_pricein = 0,
     n_bad_relabel = 0,
     n_prefine   =0;

long   n,                    /* number of nodes */
       m;                    /* number of arcs */

short   *cap;                 /* array containig capacities */

node   *nodes,               /* array of nodes */
       *sentinel_node,       /* next after last */
       *excq_first,          /* first node in push-queue */
       *excq_last;           /* last node in push-queue */

arc    *arcs,                /* array of arcs */
       *sentinel_arc;        /* next after last */

bucket *buckets,             /* array of buckets */
       *l_bucket;            /* last bucket */
long   linf;                 /* number of l_bucket + 1 */
double dlinf;                /* copy of linf in double mode */

int time_for_price_in;
double epsilon,              /* optimality bound */
       low_bound,            /* lowest bound for epsilon */
       price_min,            /* lowest bound for prices */
       f_scale,              /* scale factor */
       dn,                   /* cost multiplier - number of nodes  + 1 */
       mmc,                  /* multiplied maximal cost */
       cut_off_factor,       /* multiplier to produce cut_on and cut_off
				from n and epsilon */
       cut_on,               /* the bound for returning suspended arcs */
       cut_off;              /* the bound for suspending arcs */

double total_excess;         /* total excess */

long   n_rel,                /* number of relabels from last price update */
       n_ref,                /* current number of refines */
       n_src;                /* current number of nodes with excess */

int   flag_price = 0,        /* if = 1 - signal to start price-in ASAP - 
				maybe there is infeasibility because of
				susoended arcs */
      flag_updt = 0;         /* if = 1 - update failed some sources are 
				unreachable: either the problem is
				unfeasible or you have to return 
                                suspended arcs */

long  empty_push_bound;      /* maximal possible number of zero pushes
                                during one discharge */

int   snc_max;               /* maximal number of cycles cancelled
                                during price refine */

arc   d_arc;                 /* dummy arc - for technical reasons */

node  d_node,                /* dummy node - for technical reasons */
      *dummy_node;           /* the address of d_node */

/************************************************ abnormal finish **********/

static void err_end ( cc )

int cc;

{
fprintf ( sp0, "\ncs2 solver: Error %d ", cc );
if(cc==ALLOCATION_FAULT){
  fprintf(sp0,"(allocation fault)\n");
}else if(cc==UNFEASIBLE){
  fprintf(sp0,"(problem infeasible)\n");
}else if(cc==PRICE_OFL){
  fprintf(sp0,"(price overflow)\n");
}

/*
2 - problem is unfeasible
5 - allocation fault
6 - price overflow
*/

exit(ABNORMAL_EXIT);
/* exit ( cc ); */
}

/************************************************* initialization **********/

static void cs_init ( n_p, m_p, nodes_p, arcs_p, f_sc, max_c, cap_p )

long    n_p,        /* number of nodes */
        m_p;        /* number of arcs */
node    *nodes_p;   /* array of nodes */
arc     *arcs_p;    /* array of arcs */
long    f_sc;       /* scaling factor */
double  max_c;      /* maximal cost */
short    *cap_p;     /* array of capacities (changed to short by CWC) */

{
node   *i;          /* current node */
/*arc    *a;  */        /* current arc */
bucket *b;          /* current bucket */

n             = n_p;
nodes         = nodes_p;
sentinel_node = nodes + n;

m    = m_p;
arcs = arcs_p;
sentinel_arc  = arcs + m;

cap = cap_p;

f_scale = f_sc;

low_bound = 1.00001;

 dn = (double) n ; 
 /*
for ( a = arcs ; a != sentinel_arc ; a ++ )
  a -> cost *= dn; 
 */

mmc = max_c * dn;

linf   = n * f_scale + 2;
dlinf  = (double)linf;

buckets = (bucket*) CAlloc ( linf, sizeof (bucket) );
if ( buckets == NULL ) 
   err_end ( ALLOCATION_FAULT );

l_bucket = buckets + linf;

dnode = &dnd;

for ( b = buckets; b != l_bucket; b ++ )
   RESET_BUCKET ( b );

epsilon = mmc;
if ( epsilon < 1 )
  epsilon = 1;

price_min = - PRICE_MAX;

FOR_ALL_NODES_i 
  {
    i -> price  = 0;
    i -> suspended = i -> first;
    i -> q_next = sentinel_node;
  }

sentinel_node -> first = sentinel_node -> suspended = sentinel_arc;

cut_off_factor = CUT_OFF_COEF * pow ( (double)n, CUT_OFF_POWER );

cut_off_factor = GREATEROF ( cut_off_factor, CUT_OFF_MIN );

n_ref = 0;

flag_price = 0;

dummy_node = &d_node;

excq_first = NULL;

empty_push_bound = n * EMPTY_PUSH_COEF;

} /* end of initialization */

/********************************************** up_node_scan *************/

static void up_node_scan ( i )

node *i;                      /* node for scanning */

{
node   *j;                     /* opposite node */
arc    *a,                     /* ( i, j ) */
       *a_stop,                /* first arc from the next node */
       *ra;                    /* ( j, i ) */
bucket *b_old,                 /* old bucket contained j */
       *b_new;                 /* new bucket for j */
long   i_rank,
       j_rank,                 /* ranks of nodes */
       j_new_rank;             
double rc,                     /* reduced cost of (j,i) */
       dr;                     /* rank difference */

n_scan ++;

i_rank = i -> rank;

FOR_ALL_ARCS_a_FROM_i 
  {

    ra = a -> sister;

    if ( OPEN ( ra ) )
      {
	j = a -> head;
	j_rank = j -> rank;

	if ( j_rank > i_rank )
	  {
	    if ( ( rc = REDUCED_COST ( j, i, ra ) ) < 0 ) 
	        j_new_rank = i_rank;
	    else
	      {
		dr = rc / epsilon;
		j_new_rank = ( dr < dlinf ) ? i_rank + (long)dr + 1
		                            : linf;
	      }

	    if ( j_rank > j_new_rank )
	      {
		j -> rank = j_new_rank;
		j -> current = ra;

		if ( j_rank < linf )
		  {
		    b_old = buckets + j_rank;
		    REMOVE_FROM_BUCKET ( j, b_old )
		  }

		b_new = buckets + j_new_rank;
		INSERT_TO_BUCKET ( j, b_new )  
	      }
	  }
      }
  } /* end of scanning arcs */

i -> price -= i_rank * epsilon;
i -> rank = -1;
}


/*************************************************** price_update *******/

static void  price_update ()

{

register node   *i;

double remain;                 /* total excess of unscanned nodes with
                                  positive excess */
bucket *b;                     /* current bucket */
double dp;                     /* amount to be subtracted from prices */

n_update ++;

FOR_ALL_NODES_i 
  {

    if ( i -> excess < 0 )
      {
	INSERT_TO_BUCKET ( i, buckets );
	i -> rank = 0;
      }
    else
      {
        i -> rank = linf;
      }
  }

remain = total_excess;
if ( remain < 0.5 ) return;

/* main loop */

for ( b = buckets; b != l_bucket; b ++ )
  {

    while ( NONEMPTY_BUCKET ( b ) )
       {
	 GET_FROM_BUCKET ( i, b )

	 up_node_scan ( i );

	 if ( i -> excess > 0 )
	   {
	     remain -= (double)(i -> excess);
             if ( remain <= 0  ) break; 
	   }

       } /* end of scanning the bucket */

    if ( remain <= 0  ) break; 
  } /* end of scanning buckets */

if ( remain > 0.5 ) flag_updt = 1;

/* finishup */
/* changing prices for nodes which were not scanned during main loop */

dp = ( b - buckets ) * epsilon;

FOR_ALL_NODES_i 
  {

    if ( i -> rank >= 0 )
    {
      if ( i -> rank < linf )
	REMOVE_FROM_BUCKET ( i, (buckets + i -> rank) );

      if ( i -> price > price_min )
	i -> price -= dp;
    }
  }

} /* end of price_update */



/****************************************************** relabel *********/

static int relabel ( i )

register node *i;         /* node for relabelling */

{
register arc    *a,       /* current arc from  i  */
                *a_stop,  /* first arc from the next node */
                *a_max;   /* arc  which provides maximum price */
register double p_max,    /* current maximal price */
                i_price,  /* price of node  i */
                dp;       /* current arc partial residual cost */

p_max = price_min;
i_price = i -> price;

for ( 
      a = i -> current + 1, a_stop = ( i + 1 ) -> suspended;
      a != a_stop;
      a ++
    )
  {
    if ( OPEN ( a )
	 &&
	 ( ( dp = ( ( a -> head ) -> price ) - dn*( a -> cost ) ) > p_max )
       )
      {
	if ( i_price < dp )
	  {
	    i -> current = a;
	    return ( 1 );
	  }

	p_max = dp;
	a_max = a;
      }
  } /* 1/2 arcs are scanned */


for ( 
      a = i -> first, a_stop = ( i -> current ) + 1;
      a != a_stop;
      a ++
    )
  {
    if ( OPEN ( a )
	 &&
	 ( ( dp = ( ( a -> head ) -> price ) - dn*( a -> cost ) ) > p_max )
       )
      {
	if ( i_price < dp )
	  {
	    i -> current = a;
	    return ( 1 );
	  }

	p_max = dp;
	a_max = a;
      }
  } /* 2/2 arcs are scanned */

/* finishup */

if ( p_max != price_min )
  {
    i -> price   = p_max - epsilon;
    i -> current = a_max;
  }
else
  { /* node can't be relabelled */
    if ( i -> suspended == i -> first )
      {
	if ( i -> excess == 0 )
	  {
	    i -> price = price_min;
	  }
	else
	  {
	    if ( n_ref == 1 )
	      {
		err_end ( UNFEASIBLE );
	      }
	    else
	      {
		err_end ( PRICE_OFL );
	      }
	  }
      }
    else /* node can't be relabelled because of suspended arcs */
      {
	flag_price = 1;
      }
   }


n_relabel ++;
n_rel ++;

return ( 0 );

} /* end of relabel */


/***************************************************** discharge *********/


static void discharge ( i )

register node *i;         /* node to be discharged */

{

register arc  *a;       /* an arc from  i  */

arc  *b,                /* an arc from j */
     *ra;               /* reversed arc (j,i) */
register node *j;       /* head of  a  */
register long df;       /* amoumt of flow to be pushed through  a  */
excess_t j_exc;             /* former excess of  j  */

int  empty_push;        /* number of unsuccessful attempts to push flow
                           out of  i. If it is too big - it is time for
                           global update */

n_discharge ++;
empty_push = 0;

a = i -> current;
j = a -> head;

if ( !ADMISSIBLE ( i, j, a ) ) 
  { 
    relabel ( i );
    a = i -> current;
    j = a -> head;
  }

while ( 1 )
{
  j_exc = j -> excess;

  if ( j_exc >= 0 )
    {
      b = j -> current;
      if ( ADMISSIBLE ( j, b -> head, b ) || relabel ( j ) )
	{ /* exit from j exists */

	  df = LESSEROF ( i -> excess, a -> r_cap );
	  if (j_exc == 0) n_src++;
	  INCREASE_FLOW ( i, j, a, df )
n_push ++;

	  if ( OUT_OF_EXCESS_Q ( j ) )
	    {
	      INSERT_TO_EXCESS_Q ( j );
	    }
	}
      else 
	{ 
	  /* push back */ 
	  ra = a -> sister;
	  df = LESSEROF ( j -> excess, ra -> r_cap );
	  if ( df > 0 )
	    {
	      INCREASE_FLOW ( j, i, ra, df );
	      if (j->excess == 0) n_src--;
n_push ++;
	    }

	  if ( empty_push ++ >= empty_push_bound )
	    {
	      flag_price = 1;
	      return;
	    }
	}
    }
  else /* j_exc < 0 */
    { 
      df = LESSEROF ( i -> excess, a -> r_cap );
      INCREASE_FLOW ( i, j, a, df )
n_push ++;

      if ( j -> excess >= 0 )
	{
	  if ( j -> excess > 0 )
	    {
              n_src++;
	      relabel ( j );
	      INSERT_TO_EXCESS_Q ( j );
	    }
	  total_excess += j_exc;
	}
      else
	total_excess -= df;

    }
  
  if (i -> excess <= 0)
    n_src--;
  if ( i -> excess <= 0 || flag_price ) break;

  relabel ( i );

  a = i -> current;
  j = a -> head;
}

i -> current = a;
} /* end of discharge */

/***************************************************** price_in *******/

static int price_in ()

{
node     *i,                   /* current node */
         *j;

arc      *a,                   /* current arc from i */
         *a_stop,              /* first arc from the next node */
         *b,                   /* arc to be exchanged with suspended */
         *ra,                  /* opposite to  a  */
         *rb;                  /* opposite to  b  */

double   rc;                   /* reduced cost */

int      n_in_bad,             /* number of priced_in arcs with
				  negative reduced cost */
         bad_found;            /* if 1 we are at the second scan
                                  if 0 we are at the first scan */

excess_t  i_exc,                /* excess of  i  */
          df;                   /* an amount to increase flow */


bad_found = 0;
n_in_bad = 0;

 restart:

FOR_ALL_NODES_i 
  {
    for ( a = ( i -> first ) - 1, a_stop = ( i -> suspended ) - 1; 
    a != a_stop; a -- )
      {
	rc = REDUCED_COST ( i, a -> head, a );

	    if ( (rc < 0) && ( a -> r_cap > 0) )
	      { /* bad case */
		if ( bad_found == 0 )
		  {
		    bad_found = 1;
		    UPDATE_CUT_OFF;
		    goto restart;

		  }
		df = a -> r_cap;
		INCREASE_FLOW ( i, a -> head, a, df );

                ra = a -> sister;
		j  = a -> head;

		b = -- ( i -> first );
		EXCHANGE ( a, b );

		if ( SUSPENDED ( j, ra ) )
		  {
		    rb = -- ( j -> first );
		    EXCHANGE ( ra, rb );
		  }

		    n_in_bad ++; 
	      }
	    else
	    if ( ( rc < cut_on ) && ( rc > -cut_on ) )
	      {
		b = -- ( i -> first );
		EXCHANGE ( a, b );
	      }
      }
  }

if ( n_in_bad != 0 )
  {
    n_bad_pricein ++;

    /* recalculating excess queue */

    total_excess = 0;
    n_src=0;
    RESET_EXCESS_Q;

      FOR_ALL_NODES_i 
	{
	  i -> current = i -> first;
	  i_exc = i -> excess;
	  if ( i_exc > 0 )
	    { /* i  is a source */
	      total_excess += i_exc;
	      n_src++;
	      INSERT_TO_EXCESS_Q ( i );
	    }
	}

    INSERT_TO_EXCESS_Q ( dummy_node );
  }

if (time_for_price_in == TIME_FOR_PRICE_IN2)
  time_for_price_in = TIME_FOR_PRICE_IN3;

if (time_for_price_in == TIME_FOR_PRICE_IN1)
  time_for_price_in = TIME_FOR_PRICE_IN2;

return ( n_in_bad );

} /* end of price_in */

/************************************************** refine **************/

static void refine () 

{
node     *i;      /* current node */
excess_t i_exc;   /* excess of  i  */

/* long   np, nr, ns; */  /* variables for additional print */

int    pr_in_int;   /* current number of updates between price_in */

/*
np = n_push; 
nr = n_relabel; 
ns = n_scan;
*/

n_refine ++;
n_ref ++;
n_rel = 0;
pr_in_int = 0;

/* initialize */

total_excess = 0;
n_src=0;
RESET_EXCESS_Q

time_for_price_in = TIME_FOR_PRICE_IN1;

FOR_ALL_NODES_i 
  {
    i -> current = i -> first;
    i_exc = i -> excess;
    if ( i_exc > 0 )
      { /* i  is a source */
	total_excess += i_exc;
        n_src++;
	INSERT_TO_EXCESS_Q ( i )
      }
  }


if ( total_excess <= 0 ) return;

/* main loop */

while ( 1 )
  {
    if ( EMPTY_EXCESS_Q )
      {
	if ( n_ref > PRICE_OUT_START ) 
	  {
	    price_in ();
	  }
	  
	if ( EMPTY_EXCESS_Q ) break;
      }

    REMOVE_FROM_EXCESS_Q ( i );

    /* push all excess out of i */

    if ( i -> excess > 0 )
     {
       discharge ( i );

       if ( TIME_FOR_UPDATE || flag_price )
	 {
	   if ( i -> excess > 0 )
	     {
	       INSERT_TO_EXCESS_Q ( i );
	     }

	   if ( flag_price && ( n_ref > PRICE_OUT_START ) )
	     {
	       pr_in_int = 0;
	       price_in ();
	       flag_price = 0;
	     }

	   price_update();

	   while ( flag_updt )
	     {
	       if ( n_ref == 1 )
		 {
		   err_end ( UNFEASIBLE );
		 }
	       else
		 {
		   flag_updt = 0;
		   UPDATE_CUT_OFF;
		   n_bad_relabel++;

		   pr_in_int = 0;
		   price_in ();

		   price_update ();
		 }
	     }

	   n_rel = 0;

	   if ( n_ref > PRICE_OUT_START && 
	       (pr_in_int ++ > time_for_price_in) 
	       )
	     {
	       pr_in_int = 0;
	       price_in ();
	     }

	 } /* time for update */
     }
  } /* end of main loop */

return;

} /*----- end of refine */


/*************************************************** price_refine **********/

static int price_refine ()

{

node   *i,              /* current node */
       *j,              /* opposite node */
       *ir,             /* nodes for passing over the negative cycle */
       *is;
arc    *a,              /* arc (i,j) */
       *a_stop,         /* first arc from the next node */
       *ar;

long   bmax;            /* number of farest nonempty bucket */
long   i_rank,          /* rank of node i */
       j_rank,          /* rank of node j */
       j_new_rank;      /* new rank of node j */
bucket *b,              /* current bucket */
       *b_old,          /* old and new buckets of current node */
       *b_new;
double rc,              /* reduced cost of a */
       dr,              /* ranks difference */
       dp;
int    cc;              /* return code: 1 - flow is epsilon optimal
                                        0 - refine is needed        */
long   df;              /* cycle capacity */

int    nnc,             /* number of negative cycles cancelled during
			   one iteration */
       snc;             /* total number of negative cycle cancelled */

n_prefine ++;

cc=1;
snc=0;

snc_max = ( n_ref >= START_CYCLE_CANCEL ) 
          ? MAX_CYCLES_CANCELLED
          : 0;

/* main loop */

while ( 1 )
{ /* while negative cycle is found or eps-optimal solution is constructed */

nnc=0;

FOR_ALL_NODES_i 
  {
    i -> rank    = 0;
    i -> inp     = WHITE;
    i -> current = i -> first;
  }

RESET_STACKQ

FOR_ALL_NODES_i 
  {
    if ( i -> inp == BLACK ) continue;

    i -> b_next = NULL;

    /* deapth first search */
    while ( 1 )
      {
	i -> inp = GREY;

	/* scanning arcs from node i starting from current */
	FOR_ALL_CURRENT_ARCS_a_FROM_i 
	  {
	    if ( OPEN ( a ) )
	      {
		j = a -> head;
		if ( REDUCED_COST ( i, j, a ) < 0 )
		  {
		    if ( j -> inp == WHITE )
		      { /* fresh node  - step forward */
			i -> current = a;
			j -> b_next  = i;
			i = j;
			a = j -> current;
                        a_stop = (j+1) -> suspended;
			break;
		      }

		    if ( j -> inp == GREY )
		      { /* cycle detected */
			cc = 0;
			nnc++;

			i -> current = a;
			is = ir = i;
			df = BIGGEST_FLOW;

			while ( 1 )
			  {
			    ar = ir -> current;
			    if ( ar -> r_cap <= df )
			      {
				df = ar -> r_cap;
			        is = ir;
			      }
			    if ( ir == j ) break;
			    ir = ir -> b_next;
			  } 


			ir = i;

			while ( 1 )
			  {
			    ar = ir -> current;
 			    INCREASE_FLOW( ir, ar -> head, ar, df)

			    if ( ir == j ) break;
			    ir = ir -> b_next;
			  } 


			if ( is != i )
			  {
			    for ( ir = i; ir != is; ir = ir -> b_next )
			      ir -> inp = WHITE;
			    
			    i = is;
			    a = (is -> current) + 1;
                            a_stop = (is+1) -> suspended;
			    break;
			  }

		      }                     
		  }
		/* if j-color is BLACK - continue search from i */
	      }
	  } /* all arcs from i are scanned */

	if ( a == a_stop )
	  {
	    /* step back */
	    i -> inp = BLACK;
n_prscan1++;
	    j = i -> b_next;
	    STACKQ_PUSH ( i );

	    if ( j == NULL ) break;
	    i = j;
	    i -> current ++;
	  }

      } /* end of deapth first search */
  } /* all nodes are scanned */

/* no negative cycle */
/* computing longest paths with eps-precision */


snc += nnc;

if ( snc<snc_max ) cc = 1;

if ( cc == 0 ) break;

bmax = 0;

while ( NONEMPTY_STACKQ )
  {
n_prscan2++;
    STACKQ_POP ( i );
    i_rank = i -> rank;
    FOR_ALL_ARCS_a_FROM_i 
      {
	if ( OPEN ( a ) )
	  {
	    j  = a -> head;
	    rc = REDUCED_COST ( i, j, a );


	    if ( rc < 0 ) /* admissible arc */
	      {
		dr = ( - rc - 0.5 ) / epsilon;
		if (( j_rank = dr + i_rank ) < dlinf )
		  {
		    if ( j_rank > j -> rank )
		      j -> rank = j_rank;
		  }
	      }
	  }
      } /* all arcs from i are scanned */

    if ( i_rank > 0 )
      {
	if ( i_rank > bmax ) bmax = i_rank;
	b = buckets + i_rank;
	INSERT_TO_BUCKET ( i, b )
      }
  } /* end of while-cycle: all nodes are scanned
           - longest distancess are computed */


if ( bmax == 0 ) /* preflow is eps-optimal */
  { break; }

for ( b = buckets + bmax; b != buckets; b -- )
  {
    i_rank = b - buckets;
    dp     = (double)i_rank * epsilon;

    while ( NONEMPTY_BUCKET( b ) )
      {
	GET_FROM_BUCKET ( i, b );

	n_prscan++;
	FOR_ALL_ARCS_a_FROM_i 
	  {
	    if ( OPEN ( a ) )
	      {
		j = a -> head;
        	j_rank = j -> rank;
        	if ( j_rank < i_rank )
	          {
		    rc = REDUCED_COST ( i, j, a );
 
		    if ( rc < 0 ) 
		        j_new_rank = i_rank;
		    else
		      {
			dr = rc / epsilon;
			j_new_rank = ( dr < dlinf ) ? i_rank - ( (long)dr + 1 )
			                            : 0;
		      }
		    if ( j_rank < j_new_rank )
		      {
			if ( cc == 1 )
			  {
			    j -> rank = j_new_rank;

			    if ( j_rank > 0 )
			      {
				b_old = buckets + j_rank;
				REMOVE_FROM_BUCKET ( j, b_old )
				}

			    b_new = buckets + j_new_rank;
			    INSERT_TO_BUCKET ( j, b_new )  
			  }
			else
			  {
			   df = a -> r_cap;
			    INCREASE_FLOW ( i, j, a, df ) 
			  }
		      }
		  }
	      } /* end if opened arc */
	  } /* all arcs are scanned */

	    i -> price -= dp;

      } /* end of while-cycle: the bucket is scanned */
  } /* end of for-cycle: all buckets are scanned */

if ( cc == 0 ) break;

} /* end of main loop */

/* finish: */

/* if refine needed - saturate non-epsilon-optimal arcs */

if ( cc == 0 )
{ 
FOR_ALL_NODES_i 
  {
    FOR_ALL_ARCS_a_FROM_i 
      {
	if ( REDUCED_COST ( i, a -> head, a ) < -epsilon )
	  {
	    if ( ( df = a -> r_cap ) > 0 )
	      {
		INCREASE_FLOW ( i, a -> head, a, df )
	      }
	  }

      }
  }
}


/*neg_cyc();*/

return ( cc );

} /* end of price_refine */



void compute_prices ()

{

node   *i,              /* current node */
       *j;              /* opposite node */
arc    *a,              /* arc (i,j) */
       *a_stop;         /* first arc from the next node */

long   bmax;            /* number of farest nonempty bucket */
long   i_rank,          /* rank of node i */
       j_rank,          /* rank of node j */
       j_new_rank;      /* new rank of node j */
bucket *b,              /* current bucket */
       *b_old,          /* old and new buckets of current node */
       *b_new;
double rc,              /* reduced cost of a */
       dr,              /* ranks difference */
       dp;
int    cc;              /* return code: 1 - flow is epsilon optimal
                                        0 - refine is needed        */


n_prefine ++;

cc=1;

/* main loop */

while ( 1 )
{ /* while negative cycle is found or eps-optimal solution is constructed */


FOR_ALL_NODES_i 
  {
    i -> rank    = 0;
    i -> inp     = WHITE;
    i -> current = i -> first;
  }

RESET_STACKQ

FOR_ALL_NODES_i 
  {
    if ( i -> inp == BLACK ) continue;

    i -> b_next = NULL;

    /* deapth first search */
    while ( 1 )
      {
	i -> inp = GREY;

	/* scanning arcs from node i */
	FOR_ALL_ARCS_a_FROM_i 
	  {
	    if ( OPEN ( a ) )
	      {
		j = a -> head;
		if ( REDUCED_COST ( i, j, a ) < 0 )
		  {
		    if ( j -> inp == WHITE )
		      { /* fresh node  - step forward */
			i -> current = a;
			j -> b_next  = i;
			i = j;
			a = j -> current;
                        a_stop = (j+1) -> suspended;
			break;
		      }

		    if ( j -> inp == GREY )
		      { /* cycle detected; should not happen */
			cc = 0;
		      }                     
		  }
		/* if j-color is BLACK - continue search from i */
	      }
	  } /* all arcs from i are scanned */

	if ( a == a_stop )
	  {
	    /* step back */
	    i -> inp = BLACK;
	    n_prscan1++;
	    j = i -> b_next;
	    STACKQ_PUSH ( i );

	    if ( j == NULL ) break;
	    i = j;
	    i -> current ++;
	  }

      } /* end of deapth first search */
  } /* all nodes are scanned */

/* no negative cycle */
/* computing longest paths */

if ( cc == 0 ) break;

bmax = 0;

while ( NONEMPTY_STACKQ )
  {
    n_prscan2++;
    STACKQ_POP ( i );
    i_rank = i -> rank;
    FOR_ALL_ARCS_a_FROM_i 
      {
	if ( OPEN ( a ) )
	  {
	    j  = a -> head;
	    rc = REDUCED_COST ( i, j, a );


	    if ( rc < 0 ) /* admissible arc */
	      {
		dr = - rc;
		if (( j_rank = dr + i_rank ) < dlinf )
		  {
		    if ( j_rank > j -> rank )
		      j -> rank = j_rank;
		  }
	      }
	  }
      } /* all arcs from i are scanned */

    if ( i_rank > 0 )
      {
	if ( i_rank > bmax ) bmax = i_rank;
	b = buckets + i_rank;
	INSERT_TO_BUCKET ( i, b )
      }
  } /* end of while-cycle: all nodes are scanned
           - longest distancess are computed */


if ( bmax == 0 )
  { break; }

for ( b = buckets + bmax; b != buckets; b -- )
  {
    i_rank = b - buckets;
    dp     = (double) i_rank;

    while ( NONEMPTY_BUCKET( b ) )
      {
	GET_FROM_BUCKET ( i, b )

	  n_prscan++;
	FOR_ALL_ARCS_a_FROM_i 
	  {
	    if ( OPEN ( a ) )
	      {
		j = a -> head;
        	j_rank = j -> rank;
        	if ( j_rank < i_rank )
	          {
		    rc = REDUCED_COST ( i, j, a );
 
		    if ( rc < 0 ) 
		        j_new_rank = i_rank;
		    else
		      {
			dr = rc;
			j_new_rank = ( dr < dlinf ) ? i_rank - ( (long)dr + 1 )
			                            : 0;
		      }
		    if ( j_rank < j_new_rank )
		      {
			if ( cc == 1 )
			  {
			    j -> rank = j_new_rank;

			    if ( j_rank > 0 )
			      {
				b_old = buckets + j_rank;
				REMOVE_FROM_BUCKET ( j, b_old )
				}

			    b_new = buckets + j_new_rank;
			    INSERT_TO_BUCKET ( j, b_new )  
			  }
		      }
		  }
	      } /* end if opened arc */
	  } /* all arcs are scanned */

	    i -> price -= dp;

      } /* end of while-cycle: the bucket is scanned */
  } /* end of for-cycle: all buckets are scanned */

if ( cc == 0 ) break;

} /* end of main loop */

} /* end of compute_prices */


/***************************************************** price_out ************/

static void price_out ()

{
node     *i;                /* current node */

arc      *a,                /* current arc from i */
         *a_stop,           /* first arc from the next node */
         *b;                /* arc to be exchanged with suspended */

double   n_cut_off,         /* -cut_off */
         rc;                /* reduced cost */

n_cut_off = - cut_off;

FOR_ALL_NODES_i 
  {
    FOR_ALL_ARCS_a_FROM_i 
      {
	rc = REDUCED_COST ( i, a -> head, a );

	if (((rc > cut_off) && (CLOSED(a -> sister)))
             ||
             ((rc < n_cut_off) && (CLOSED(a)))
           )
	  { /* suspend the arc */
	    b = ( i -> first ) ++ ;

	    EXCHANGE ( a, b );
	  }
      }
  }

} /* end of price_out */


/**************************************************** update_epsilon *******/
/*----- decrease epsilon after epsilon-optimal flow is constructed */

static int update_epsilon()
{

if ( epsilon <= low_bound ) return ( 1 );

epsilon = ceil ( epsilon / f_scale );

cut_off        = cut_off_factor * epsilon;
cut_on         = cut_off * CUT_OFF_GAP;

return ( 0 );
}


/*************************************************** finishup ***********/
static void finishup ( obj_ad )

double *obj_ad;       /* objective */

{
arc   *a;            /* current arc */
long  na;            /* corresponding position in capacity array */
double  obj_internal;/* objective */
double cs;           /* actual arc cost */
long   flow;         /* flow through an arc */

obj_internal = 0;

for ( a = arcs, na = 0; a != sentinel_arc ; a ++, na ++ )
    {
      /*      cs = a -> cost / dn;  */
      cs = a -> cost;

      if ( cap[na]  > 0 && ( flow = cap[na] - (a -> r_cap) ) != 0 )
	obj_internal += cs * (double) flow; 

      /*       a -> cost = cs;  */
    }

*obj_ad = obj_internal;

}


/*********************************************** init_solution ***********/
/*  static void init_solution ( ) */


/*  { */
/*  arc   *a; */   /* current arc  (i,j) */ 
/*  node  *i, */   /* tail of  a  */ 
/*        *j; */   /* head of  a  */ 
/*  long  df; */   /* ricidual capacity */ 

/*  for ( a = arcs; a != sentinel_arc ; a ++ ) */
/*      { */
/*        if ( a -> r_cap > 0 && a -> cost < 0 ) */
/*  	{ */
/*  	  df = a -> r_cap; */
/*  	  i  = ( a -> sister ) -> head; */
/*            j  = a -> head; */
/*  	  INCREASE_FLOW ( i, j, a, df ); */
/*  	} */
/*      } */
/*  } */

  /* check complimentary slackness */ 
/*  int check_cs () */

/*  { */
/*    node *i; */
/*    arc *a, *a_stop; */

/*    FOR_ALL_NODES_i */
/*      FOR_ALL_ARCS_a_FROM_i */
/*        if (OPEN(a) && (REDUCED_COST(i, a->head, a) < 0)) */
/*  	assert(0); */

/*    return(1); */
/*  } */

/************************************************* cs2 - head program ***/

static void  cs2 ( n_p, m_p, nodes_p, arcs_p, f_sc, max_c, cap_p, obj_ad)

long    n_p,        /* number of nodes */
        m_p;        /* number of arcs */
node    *nodes_p;   /* array of nodes */
arc     *arcs_p;    /* array of arcs */
long    f_sc;       /* scaling factor */
double  max_c;      /* maximal cost */
short   *cap_p;     /* capacities (changed to short by CWC) */
double  *obj_ad;    /* objective */

{

int cc;             /* for storing return code */
cs_init ( n_p, m_p, nodes_p, arcs_p, f_sc, max_c, cap_p );

/*init_solution ( );*/
cc = 0;
update_epsilon ();

do{  /* scaling loop */

    refine ();

    if ( n_ref >= PRICE_OUT_START )
      {
	price_out ( );
      }

    if ( update_epsilon () ) break;

    while ( 1 )
      {
        if ( ! price_refine () ) break;

	if ( n_ref >= PRICE_OUT_START )
	  {
	    if ( price_in () ) 
	      { 
		break; 
	      }
	  }
	if ((cc = update_epsilon ())) break;
      }
  } while ( cc == 0 );

finishup ( obj_ad );

}

/*-----------------------------------------------------------------------*/

/* SolveCS2-- formerly main() */

void SolveCS2(signed char **residue, short **mstcosts, long nrow, long ncol, 
	      long cs2scalefactor, short ***flowsptr)
{

  /*  double t; */
  arc *arp;
  node *ndp;
  long n, m, m2, nmin; 
  node *i;
  long ni;
  arc *a;
  long nNrow, nNcol;
  long to, from, num, flow, ground;
  long f_sc;

  double cost,  c_max;
  short *cap;  /* cap changed to short by CWC */

  short **rowcost, **colcost;
  short **rowflow, **colflow;
  
  /* number of rows, cols, in residue network */
  nNrow=nrow-1;
  nNcol=ncol-1;
  ground=nNrow*nNcol+1;

  /* parse input, set up the problem */
  rowcost=mstcosts;
  colcost=&(mstcosts[nrow-1]);
  f_sc=cs2scalefactor;
  cs2mcfparse( residue,rowcost,colcost,nNrow,nNcol,
	       &n,&m,&ndp,&arp,&nmin,&c_max,&cap );

  /* free memory that is no longer needed */
  Free2DArray((void **)residue,nrow-1);
  Free2DArray((void **)mstcosts,2*nrow-1);

  /* solve it! */
  fprintf(sp2,"Running cs2 MCF solver\n");
  m2 = 2 * m;
  cs2 ( n, m2, ndp, arp, f_sc, c_max, cap, &cost );


  /* parse flow solution and place into flow arrays */
  
  /* get memory for flow arrays */
  (*flowsptr)=(short **)Get2DRowColZeroMem(nrow,ncol,
					   sizeof(short *),sizeof(short));
  rowflow=(*flowsptr);
  colflow=&((*flowsptr)[nrow-1]);

  /* loop over nodes */
  for ( i = ndp; i < ndp + n; i ++ ){
    ni = N_NODE ( i );

    /* loop over arcs */
    for ( a = i -> suspended; a != (i+1)->suspended; a ++ ){

      /* if finite (non-zero) flow */
      if ( cap[ N_ARC (a) ]  > 0 &&  (cap[ N_ARC (a) ] - ( a -> r_cap ) ) ){
	
	/* get to, from nodes and flow amount */
	from=ni;
	to=N_NODE( a -> head );
	flow=cap[ N_ARC (a) ] - ( a -> r_cap );
      
	if(flow>LARGESHORT || flow<-LARGESHORT){
	  fprintf(sp0,"Flow will overflow short data type\nAbort\n");
	  exit(ABNORMAL_EXIT);
	}

	if(from==(to+1)){    
	  num=from+(int )((from-1)/nNrow);
	  colflow[(num-1) % (nNrow+1)][(int )(num-1)/(nNrow+1)]-=flow;
	}else if(from==(to-1)){
	  num=from+(int )((from-1)/nNrow)+1;
	  colflow[(num-1) % (nNrow+1)][(int )(num-1)/(nNrow+1)]+=flow;
	}else if(from==(to-nNrow)){
	  num=from+nNrow;
	  rowflow[(num-1) % nNrow][(int )((num-1)/nNrow)]+=flow;
	}else if(from==(to+nNrow)){
	  num=from;
	  rowflow[(num-1) % nNrow][(int )((num-1)/nNrow)]-=flow;
	}else if((from==ground) || (to==ground)){
	  if(to==ground){
	    num=to;
	    to=from;
	    from=num;
	    flow=-flow;
	  }
	  if(!((to-1) % nNrow)){
	    colflow[0][(int )((to-1)/nNrow)]+=flow;
	  }else if(to<=nNrow){
	    rowflow[to-1][0]+=flow;
	  }else if(to>=(ground-nNrow-1)){
	    rowflow[(to-1) % nNrow][nNcol]-=flow;
	  }else if(!(to % nNrow)){
	    colflow[nNrow][(int )((to/nNrow)-1)]-=flow;
	  }else{
	    fprintf(sp0,"Unassigned ground arc parsing cs2 solution\nAbort\n");
	    exit(ABNORMAL_EXIT);
	  }        
	}else{
	  fprintf(sp0,"Non-grid arc parsing cs2 solution\nAbort\n");
	  exit(ABNORMAL_EXIT);
	}
      } /* end if flow on arc */

    } /* end for loop over arcs of node */
  } /* end for loop over nodes */

  /* free memory */
  free(ndp-nmin);  
  free(arp);
  free(cap);
  free(buckets);

}

#endif  /* end #ifndef NO_CS2 */
