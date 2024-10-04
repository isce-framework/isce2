/*************************************************************************

  This code is derived from cs2 v3.7
  Written by Andrew V. Goldberg and Boris Cherkassky
  Modifications for use in snaphu by Curtis W. Chen 

  Parser for cs2 minimum cost flow solver.  Originally written to read
  DIMACS format (text) input files.  Modified to parse passed data
  from snaphu.  This file is included with a #include from
  snaphu_cs2.c.

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



int cs2mcfparse(residue, rowcost, colcost, nNrow, nNcol, n_ad, m_ad, nodes_ad, 
    arcs_ad, node_min_ad, m_c_ad, cap_ad )

/* parameters passed to set up network */
signed char    **residue;             /* 2D array of residues */
short   **rowcost;             /* 2D array of row arc costs */
short   **colcost;             /* 2D array of col arc costs */
long    nNrow;                 /* number of nodes per row */
long    nNcol;                 /* number of nodes per column */

/* these parameters are output */
long    *n_ad;                 /* address of the number of nodes */
long    *m_ad;                 /* address of the number of arcs */
node    **nodes_ad;            /* address of the array of nodes */
arc     **arcs_ad;             /* address of the array of arcs */
long    *node_min_ad;          /* address of the minimal node */
double  *m_c_ad;               /* maximal arc cost */
short    **cap_ad;             /* array of capacities (changed to short) */

{


#define ABS( x ) ( (x) >= 0 ) ? (x) : -(x)

/* variables added for unwrapping parse */
unsigned int row, col, dir;
unsigned long narcs, nnodes, nodectr, arcctr, nresidues;
long cumsupply, temp;


long inf_cap = 0;
long    n,                      /* internal number of nodes */
        node_min,               /* minimal no of node  */
        node_max,               /* maximal no of nodes */
       *arc_first,              /* internal array for holding
                                     - node degree
                                     - position of the first outgoing arc */
       *arc_tail,               /* internal array: tails of the arcs */
        /* temporary variables carrying no of nodes */
        head, tail, i;

long    m,                      /* internal number of arcs */
        /* temporary variables carrying no of arcs */
        last, arc_num, arc_new_num;

node    *nodes,                 /* pointers to the node structure */
        *head_p,
        *ndp,
        *in,
        *jn;

arc     *arcs,                  /* pointers to the arc structure */
        *arc_current,
        *arc_new,
        *arc_tmp;

long    excess,                 /* supply/demand of the node */
        low,                    /* lowest flow through the arc */
        acap;                    /* capacity */

long    cost;                   /* arc cost */


double  dcost,                  /* arc cost in double mode */
        m_c;                    /* maximal arc cost */

short    *cap;                   /* array of capacities (changed to short) */

double  total_p,                /* total supply */
        total_n,                /* total demand */
        cap_out,                /* sum of outgoing capacities */
        cap_in;                 /* sum of incoming capacities */

long    no_lines=0,             /* no of current input line */
  /*    no_plines=0, */           /* no of problem-lines */
  /*    no_nlines=0, */            /* no of node lines */
        no_alines=0,            /* no of arc-lines */
        pos_current=0;          /* 2*no_alines */

 int    /* k, */                     /* temporary */
        err_no;                 /* no of detected error */

/* -------------- error numbers & error messages ---------------- */
#define EN1   0
#define EN2   1
#define EN3   2
#define EN4   3
#define EN6   4
#define EN10  5
#define EN7   6
#define EN8   7
#define EN9   8
#define EN11  9
#define EN12 10
#define EN13 11
#define EN14 12
#define EN16 13
#define EN15 14
#define EN17 15
#define EN18 16
#define EN21 17
#define EN19 18
#define EN20 19
#define EN22 20

static char *err_message[] = 
  { 
/* 0*/    "more than one problem line",
/* 1*/    "wrong number of parameters in the problem line",
/* 2*/    "it is not a Min-cost problem line",
/* 3*/    "bad value of a parameter in the problem line",
/* 4*/    "can't obtain enough memory to solve this problem",
/* 5*/    "",
/* 6*/    "can't read problem name",
/* 7*/    "problem description must be before node description",
/* 8*/    "wrong capacity bounds",
/* 9*/    "wrong number of parameters in the node line",
/*10*/    "wrong value of parameters in the node line",
/*11*/    "unbalanced problem",
/*12*/    "node descriptions must be before arc descriptions",
/*13*/    "too many arcs in the input",
/*14*/    "wrong number of parameters in the arc line",
/*15*/    "wrong value of parameters in the arc line",
/*16*/    "unknown line type in the input",
/*17*/    "read error",
/*18*/    "not enough arcs in the input",
/*19*/    "warning: capacities too big - excess overflow possible",
/*20*/    "can't read anything from the input file",
/*21*/    "warning: infinite capacity replaced by BIGGEST_FLOW"
  };
/* --------------------------------------------------------------- */


  
/* set up */
nnodes=nNrow*nNcol+1;                           /* add one for ground node */
narcs=2*((nNrow+1)*nNcol+nNrow*(nNcol+1));  /* 2x for two directional arcs */
cumsupply=0;
nresidues=0;

/* get memory (formerly case 'p' in DIMACS file read) */
fprintf(sp2,"Setting up data structures for cs2 MCF solver\n");
n=nnodes;
m=narcs; 
if ( n <= 0  || m <= 0 )
  /*wrong value of no of arcs or nodes*/
  { err_no = EN4; goto error; }
 
/* allocating memory for  'nodes', 'arcs'  and internal arrays */
nodes    = (node*) CAlloc ( n+2, sizeof(node) );
arcs     = (arc*)  CAlloc ( 2*m+1, sizeof(arc) );
cap      = (short*) CAlloc ( 2*m,   sizeof(short) ); /* changed to short */
arc_tail = (long*) CAlloc ( 2*m,   sizeof(long) ); 
arc_first= (long*) CAlloc ( n+2, sizeof(long) );
/* arc_first [ 0 .. n+1 ] = 0 - initialized by calloc */

for ( in = nodes; in <= nodes + n; in ++ )
  in -> excess = 0;
                    
if ( nodes == NULL || arcs == NULL || 
     arc_first == NULL || arc_tail == NULL )
  /* memory is not allocated */
  { err_no = EN6; goto error; }
                     
/* setting pointer to the first arc */
arc_current = arcs;
node_max = 0;
node_min = n;
m_c      = 0;
total_p = total_n = 0;

for ( ndp = nodes; ndp < nodes + n; ndp ++ )
  ndp -> excess = 0;

/* end of former case 'p' */


/* load supply/demand info into arrays (case 'n' in former loop) */
for(col=0; col<nNcol; col++){
  for(row=0; row<nNrow; row++){
    if(residue[row][col]){
      i=(col*nNrow + row + 1);
      excess=residue[row][col];
      ( nodes + i ) -> excess = excess;
      if ( excess > 0 ) total_p += (double)excess;
      if ( excess < 0 ) total_n -= (double)excess;
      nresidues++;
      cumsupply+=residue[row][col];
    }
  }
}

/* give ground node excess of -cumsupply */
( nodes + nnodes ) -> excess = -cumsupply;
if (cumsupply < 0) total_p -= (double)cumsupply;
if (cumsupply > 0) total_n += (double)cumsupply;

/* load arc info into arrays (case 'a' in former loop) */
low=0;
acap=ARCUBOUND;

/* horizontal (row) direction arcs first */
for(arcctr=1;arcctr<=2*nNrow*nNcol+nNrow+nNcol;arcctr++){
  if(arcctr<=nNrow*(nNcol+1)){
    /* row (horizontal) arcs first */
    nodectr=arcctr;
    if(nodectr<=nNrow*nNcol){
      tail=nodectr;
    }else{
      tail=nnodes;
    }
    if(nodectr<=nNrow){
      head=nnodes;
    }else{
      head=nodectr-nNrow;
    }
    cost=rowcost[((nodectr-1) % nNrow)][(int )((nodectr-1)/nNrow)];
  }else{
    /* column (vertical) arcs */
    nodectr=arcctr-nNrow*(nNcol+1);
    if(nodectr % (nNrow+1)==0){
      tail=nnodes;
    }else{
      tail=(int )(nodectr-ceil(nodectr/(nNrow+1.0))+1);
    }
    if(nodectr % (nNrow+1)==1){
      head=nnodes;
    }else{
      head=(int )(nodectr-ceil(nodectr/(nNrow+1.0)));
    }
    cost=colcost[((nodectr-1) % (nNrow+1))][(int )((nodectr-1)/(nNrow+1))];
  }
 
  if ( tail < 0  ||  tail > n  ||
       head < 0  ||  head > n  
       )
    /* wrong value of nodes */
    { err_no = EN17; goto error; }
  
  if ( acap < 0 ) {
    acap = BIGGEST_FLOW;
    if (!inf_cap) {
      inf_cap = 1;
      fprintf ( sp0, "\ncs2 solver: %s\n", err_message[21] );
    }
  }
  
  if ( low < 0 || low > acap )
    { err_no = EN9; goto error; }

  for(dir=0;dir<=1;dir++){
    if(dir){
      /* switch head and tail and loop for two directional arcs */
      temp=tail;
      tail=head;
      head=temp;      
    }

    /* no of arcs incident to node i is placed in arc_first[i+1] */
    arc_first[tail + 1] ++; 
    arc_first[head + 1] ++;
    in    = nodes + tail;
    jn    = nodes + head;
    dcost = (double)cost;
    
    /* storing information about the arc */
    arc_tail[pos_current]        = tail;
    arc_tail[pos_current+1]      = head;
    arc_current       -> head    = jn;
    arc_current       -> r_cap   = acap - low;
    cap[pos_current]             = acap;
    arc_current       -> cost    = dcost;
    arc_current       -> sister  = arc_current + 1;
    ( arc_current + 1 ) -> head    = nodes + tail;
    ( arc_current + 1 ) -> r_cap   = 0;
    cap[pos_current+1]           = 0;
    ( arc_current + 1 ) -> cost    = -dcost;
    ( arc_current + 1 ) -> sister  = arc_current;
    
    in -> excess -= low;
    jn -> excess += low;
    
    /* searching for minimum and maximum node */
    if ( head < node_min ) node_min = head;
    if ( tail < node_min ) node_min = tail;
    if ( head > node_max ) node_max = head;
    if ( tail > node_max ) node_max = tail;

    if ( dcost < 0 ) dcost = -dcost;
    if ( dcost > m_c && acap > 0 ) m_c = dcost;
    
    no_alines   ++;
    arc_current += 2;
    pos_current += 2;

  }/* end of for loop over arc direction */
}/* end of for loop over arcss */


/* ----- all is red  or  error while reading ----- */ 

if ( ABS( total_p - total_n ) > 0.5 ) /* unbalanced problem */
  { err_no = EN13; goto error; }

/********** ordering arcs - linear time algorithm ***********/

/* first arc from the first node */
( nodes + node_min ) -> first = arcs;

/* before below loop arc_first[i+1] is the number of arcs outgoing from i;
   after this loop arc_first[i] is the position of the first 
   outgoing from node i arcs after they would be ordered;
   this value is transformed to pointer and written to node.first[i]
   */
 
for ( i = node_min + 1; i <= node_max + 1; i ++ ) 
  {
    arc_first[i]          += arc_first[i-1];
    ( nodes + i ) -> first = arcs + arc_first[i];
  }


for ( i = node_min; i < node_max; i ++ ) /* scanning all the nodes  
                                            exept the last*/
  {

    last = ( ( nodes + i + 1 ) -> first ) - arcs;
                             /* arcs outgoing from i must be cited    
                              from position arc_first[i] to the position
                              equal to initial value of arc_first[i+1]-1  */

    for ( arc_num = arc_first[i]; arc_num < last; arc_num ++ )
      { tail = arc_tail[arc_num];

        while ( tail != i )
          /* the arc no  arc_num  is not in place because arc cited here
             must go out from i;
             we'll put it to its place and continue this process
             until an arc in this position would go out from i */

          { arc_new_num  = arc_first[tail];
            arc_current  = arcs + arc_num;
            arc_new      = arcs + arc_new_num;
            
            /* arc_current must be cited in the position arc_new    
               swapping these arcs:                                 */

            head_p               = arc_new -> head;
            arc_new -> head      = arc_current -> head;
            arc_current -> head  = head_p;

            acap                 = cap[arc_new_num];
            cap[arc_new_num]     = cap[arc_num];
            cap[arc_num]         = acap;

            acap                 = arc_new -> r_cap;
            arc_new -> r_cap     = arc_current -> r_cap;
            arc_current -> r_cap = acap;

            dcost                = arc_new -> cost;
            arc_new -> cost      = arc_current -> cost;
            arc_current -> cost  = dcost;

            if ( arc_new != arc_current -> sister )
              {
                arc_tmp                = arc_new -> sister;
                arc_new  -> sister     = arc_current -> sister;
                arc_current -> sister  = arc_tmp;

                ( arc_current -> sister ) -> sister = arc_current;
                ( arc_new     -> sister ) -> sister = arc_new;
              }

            arc_tail[arc_num] = arc_tail[arc_new_num];
            arc_tail[arc_new_num] = tail;

            /* we increase arc_first[tail]  */
            arc_first[tail] ++ ;

            tail = arc_tail[arc_num];
          }
      }
    /* all arcs outgoing from  i  are in place */
  }       

/* -----------------------  arcs are ordered  ------------------------- */

/*------------ testing network for possible excess overflow ---------*/

for ( ndp = nodes + node_min; ndp <= nodes + node_max; ndp ++ )
{
   cap_in  =   ( ndp -> excess );
   cap_out = - ( ndp -> excess );
   for ( arc_current = ndp -> first; arc_current != (ndp+1) -> first; 
         arc_current ++ )
      {
        arc_num = arc_current - arcs;
        if ( cap[arc_num] > 0 ) cap_out += cap[arc_num];
        if ( cap[arc_num] == 0 ) 
          cap_in += cap[( arc_current -> sister )-arcs];
      }

   /*
   if (cap_in > BIGGEST_FLOW || cap_out > BIGGEST_FLOW)
     { 
       fprintf ( sp0, "\ncs2 solver: %s\n", err_message[EN20] );
       break;
     }
   */
}

/* ----------- assigning output values ------------*/
*m_ad = m;
*n_ad = node_max - node_min + 1;
*node_min_ad = node_min;
*nodes_ad = nodes + node_min;
*arcs_ad = arcs;
*m_c_ad  = m_c;
*cap_ad   = cap;

/* free internal memory */
free ( arc_first ); free ( arc_tail );

/* Thanks God! All is done! */
return (0);

/* ---------------------------------- */
 error:  /* error found reading input */

fprintf ( sp0, "\ncs2 solver: line %ld of input - %s\n", 
         no_lines, err_message[err_no] );

exit (ABNORMAL_EXIT);

/* this is a needless return statement so the compiler doesn't complain */
return(1);

}
/* --------------------   end of parser  -------------------*/



