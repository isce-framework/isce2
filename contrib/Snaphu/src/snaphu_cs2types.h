/*************************************************************************

  This code is derived from cs2 v3.7
  Written by Andrew V. Goldberg and Boris Cherkassky
  Modifications for use in snaphu by Curtis W. Chen 

  Header for cs2 minimum cost flow solver.  This file is included with
  a #include from snaphu_cs2.c.

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

/* defs.h */


typedef long excess_t;

typedef  /* arc */
   struct arc_st
{
   short            r_cap;           /* residual capasity */
   short            cost;            /* cost  of the arc*/
   struct node_st   *head;           /* head node */
   struct arc_st    *sister;         /* opposite arc */
}
  arc;

typedef  /* node */
   struct node_st
{
   arc              *first;           /* first outgoing arc */
   arc              *current;         /* current outgoing arc */
   arc              *suspended;
   double           price;            /* distance from a sink */
   struct node_st   *q_next;          /* next node in push queue */
   struct node_st   *b_next;          /* next node in bucket-list */
   struct node_st   *b_prev;          /* previous node in bucket-list */
   long             rank;             /* bucket number */
   excess_t         excess;           /* excess of the node */
   signed char      inp;              /* temporary number of input arcs */
} node;

typedef /* bucket */
   struct bucket_st
{
   node             *p_first;         /* 1st node with positive excess 
                                         or simply 1st node in the buket */
} bucket;

