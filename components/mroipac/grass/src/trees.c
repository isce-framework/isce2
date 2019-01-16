#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "defines.h"

int nearest_tree(unsigned char **, int, int, int) ;

void start_timing();					/* timing routines */
void stop_timing();

/**
 * @param flagFilename phase unwrapping flag file name
 * @param width number of samples per row
 * @param mbl maximum branch length (default=64)
 * @param start starting line (default=1)
 * @param xmin starting range pixel offset
 * @param xmax last range pixel offset
 * @param ymin starting azimuth row
 * @param ymax last azimuth row
 */
int
trees(char *flagFilename, int width, int mbl, int start, int xmin, int xmax, int ymin, int ymax)
{
    int nlines;			/* number of lines in the file */
    int xw,yh;			/* width, height of processed region */
    int offs;			/* offset number of lines to read from start of file*/
    int i;
    int r_chrg;			/* residual charge */
    unsigned char *fbf;		/* flag array */
    unsigned char **gzw;	/* set of pointers to the rows of flag array */
    FILE *flag_file;
    double Pi=4*atan2(1,1);
 
  flag_file = fopen(flagFilename,"r+"); 
  if (flag_file == NULL){
      fprintf(stderr, "flag file does not exist!\n");
      exit(-1);
  }

  if (xmax <=0) {
     xmax=width-1;				 	/* default value of xmax */
  }

  fseek(flag_file, 0L, REL_EOF);		/* determine # lines in the file */
  nlines=(int)ftell(flag_file)/width;
  fprintf(stderr,"#lines in the file: %d\n",nlines); 
  rewind(flag_file);

  if (ymax <= 0) {
     ymax=nlines-start;				/* default value of ymax */
  }

  if (ymax > nlines-start){
    ymax = nlines-start;
    fprintf(stderr,"insufficient #lines in the file, ymax: %d\n",ymax);
  }

  xw=xmax-xmin+1;	/* width of each line to process */
  yh=ymax-ymin+1;	/* height of array */
  offs=start+ymin-1;	/* first line of file to start reading/writing */
 

  fbf = (unsigned char *) malloc(sizeof(unsigned char)*width*yh);
  if(fbf == (unsigned char *) NULL) {
    fprintf(stderr,"failure to allocate space for flag array\n");
    exit(-1) ;
  }

  gzw=(unsigned char **)malloc(sizeof(unsigned char*) * yh);
  
/**************** Read in flag data *********************/
   
  fseek(flag_file, offs*width*sizeof(unsigned char), REL_BEGIN); 	/*seek start line */

  fprintf(stdout,"reading input file...\n");
  fread((char *)fbf, sizeof(unsigned char), yh*width, flag_file); 
 
  for (i=0; i< yh; i++){
    gzw[i] = (unsigned char *)(fbf + i*width + xmin);
  }
      
  fprintf(stdout,"creating trees:  width, height, mbl %d %d %d ...\n",xw, yh,mbl); 
  r_chrg=nearest_tree(gzw, yh, xw, mbl);			/* grow the trees */
  fprintf(stdout,"\nresidual charge: %d\n",r_chrg);

/************** write out flag array *************/

  fprintf(stdout,"writing output flag file...\n");
  fseek(flag_file, offs*width*sizeof(unsigned char), REL_BEGIN); 	/*seek start line */
  fwrite((char *)fbf, sizeof(unsigned char), yh*width, flag_file); 

  fclose(flag_file);
 return 0; 
}  

/* minimization of the longest connection in discharged trees */
/* 1992. Feb. 24. */
/* modified to include guiding centers and arbitrary dimensions 1993 Mar 23. */
/* by Paul Rosen */
/* search from right side to left */


/* charge_table :found in search from an unvisited charge    */
/* n_charge: # of charges in table                           */
/* charge_t: pointer to the tree table                       */
/* charge_i,j: (x,y) of the charge                           */
#define size_charge_table 8000000
int n_charge,charge_t[size_charge_table];
int charge_i[size_charge_table],charge_j[size_charge_table];

/* tree_table                                                */
/* n_tree: # of trees found                                  */
/* tree_f: pointer to the charge found first                 */
/* tree_p: pointer to the charge of previous tree            */
#define size_tree_table 3000000
int n_tree,tree_f[size_tree_table],tree_p[size_tree_table];

/* cut_table */
#define size_cut_table 8000000
int n_cut,cut_i[size_cut_table],cut_j[size_cut_table];

int dir_i[]={-1,1,0,1,1,0,-1,-1,0};
int dir_j[]={1,-1,1,1,0,0,0,-1,-1};

/* new_mode: flag decides if exisiting trees are used or not */
/*           0: unvisited charges are connected directly     */
/*           1: charges are connected thru existing trees    */
int new_mode=1; 

#define size_search_table 5000
#define VISITED 0
#define GROUNDED 1

int n_search,si[size_search_table],sj[size_search_table];

/* generate a table of points "on" the size k box around (i1,j1) */
/* only available points are stored.                             */

void make_search_table(int ,int ,int, int, int) ;
void add_charge(unsigned char **, int, int) ;
void add_cut(unsigned char **,int, int) ;
int trace_tree(unsigned char **, int ,int ,int, int, int) ;
void connect_trees(unsigned char **, int) ;
int nearest_tree(unsigned char **, int, int, int) ;
void cutmap(unsigned char **, int, int, int, int) ;

void make_search_table(int i1,int j1,int k, int data_w, int data_h)
{
  int i,j;
  n_search=0;
  if(k==0) return;
  if((j1-k)>= -1)
    for(i=i1-k;i<i1+k;i++)
      if((i>= -1)&&(i<data_w)) {si[n_search]=i; sj[n_search++]=j1-k;}
  if((i1+k)<data_w)
    for(j=j1-k;j<j1+k;j++)
      if((j>= -1)&&(j<data_h)) {si[n_search]=i1+k; sj[n_search++]=j;}
  if((j1+k)<data_h)
    for(i=i1+k;i>i1-k;i--)
      if((i>= -1)&&(i<data_w)) {si[n_search]=i; sj[n_search++]=j1+k;}
  if((i1-k)>= -1)
    for(j=j1+k;j>j1-k;j--)
      if((j>= -1)&&(j<data_h)) {si[n_search]=i1-k; sj[n_search++]=j;}
}

/* add charge (i,j) to the charge_table */
void add_charge(unsigned char **gzw, int i,int j) 
{
  if((gzw[i][j]&BRPT)!=0) return;
  gzw[i][j] |= BRPT;
  charge_i[n_charge]=i; 
  charge_j[n_charge]=j;
  charge_t[n_charge++]=n_tree;
  if(n_charge>=size_charge_table) {
    fprintf(stderr,"error: subroutine add_charge, charge table is full!\n"); exit(1);}
}

/* add cut (i,j) to the cut_table */
void add_cut(unsigned char **gzw, int i,int j) 
{
  if((gzw[i][j]&TREE)!=0) return ;
    /* {fprintf(stderr,"error: subroutine add_cut, there is a bug %d %d %d\n", gzw[i][j],i,j); return;} */
  gzw[i][j] |= TREE;
  cut_i[n_cut]=i;
  cut_j[n_cut++]=j;
  if(n_cut>=size_cut_table) {
    fprintf(stderr,"error: subroutine add_cut, cut table is full!\n"); exit(1);}
}

/* tracing an existing tree                             */
/* (i,j) is position of the found visited charge.       */
/* ip is a pointer to the charge in charge_table        */
/* from which the visited charge was found.             */

int trace_tree(unsigned char **gzw, int i,int j,int ip, int data_w, int data_h)
{
  int i1,j1,i2,j2,k,nc,b;
                                /* create a new tree in tree_table   */
  tree_f[n_tree]=n_charge;      /* pointer to the first found charge */
  tree_p[n_tree]=ip;            /* pointer to the charge in the previous tree*/

  i1=i; j1=j;                   /* (i1,j1) is position of the first charge */
  add_charge(gzw, i1,j1);       /* Add the first charge to the charge_table */

                                /* Start tracing the tree */

  b=0;                          /* b: flag indicates whether the connection */
                                /*    to the boundary is found or not.      */
                                /*    0: no connection to the boundary      */
                                /*    1: connection was found               */

  n_cut=0;                      /* first no cuts are in the table */

  /* Only 4 pixels around the first charge are inspected. */
  for(k=2; k<6; k++) {
    i2=i1+dir_i[k]; j2=j1+dir_j[k];
    if((i2== -1)||(i2==data_w)||(j2== -1)||(j2==data_h)) 
      /* The first charge was the next to the boundary.           */
      /* So the unvisited charge can be connected to the boundary */
      /* using this visited charge.                               */
      {b=1; goto clear_tree;} 
    if((gzw[i2][j2]&CUT)!=0) {
      /* cuts found around the charge are added to the cut_table. */
      add_cut(gzw,i2,j2);
      /* if visited charges is on the cut, it is added to the     */
      /* charge_table.                                            */
      if((gzw[i2][j2]&VIST)!=0) add_charge(gzw,i2,j2);
    }
  }
                             /* search from found cuts */
  nc=0;                      /* pointer to each cut in the cut_table */
  while(nc<n_cut) {          /* all of cuts in cut_table are inspected. */
    i1=cut_i[nc];            /* (i1,j1) is inspected. */
    j1=cut_j[nc++];

    /* first, 3 pixels around the cut are checked. */
    for(k=6; k<9; k++) {
      i2=i1+dir_i[k]; j2=j1+dir_j[k]; 
      if((i2== -1)||(i2==data_w)||(j2== -1)||(j2==data_h)) 
	/* The cut was the next to the boundary, which means          */
        /* the visited charge (i,j) has a connection to the boundary, */
	/* so the unvisited charge can be connected to the boundary   */
	/* thru this connection.                                      */
	{b=1; goto clear_tree;}

      /* cuts checked found in a previous tree search are marked as TREE. */
      if((gzw[i2][j2]&TREE)!=0) continue;
      
      /* cuts and visited charges found are stored in tables. */
      if((gzw[i2][j2]&CUT)!=0) 	add_cut(gzw,i2,j2);
      if((gzw[i2][j2]&VIST)!=0) add_charge(gzw,i2,j2);
    }

    /* Second, 5 pixels around the cut are checked. */
    for(k=0; k<5; k++) {
      i2=i1+dir_i[k]; j2=j1+dir_j[k]; 
      /* check connection to the boundary. */
      if((i2== -1)||(i2==data_w)||(j2== -1)||(j2==data_h)) 
	{b=1; goto clear_tree;}
      /* check if it's already checked or not. */
      if((gzw[i2][j2]&TREE)!=0) continue;
      /* check if it's marked as 'CUT' or not. */
      if((gzw[i2][j2]&CUT)!=0) {
	/* add the found cut to cut_table. */
	add_cut(gzw,i2,j2);
	/* if visited charge is on the cut, it is added to charge_table. */
	if((gzw[i2][j2]&VIST)!=0) add_charge(gzw,i2,j2);
      }
    }
  }

  /* end of search: TREE marks are all cleared. */
 clear_tree:
  for(k=0;k<n_cut;k++)
    gzw[cut_i[k]][cut_j[k]] &= 255-TREE;

  if(b==0) {
    /* if the connection to the boundary was not found,   */
    /* the newly created tree is counted and              */
    /* VISITED is returned to calling procedure, which    */
    /* means the charge was connected to a floating tree. */
    n_tree++;
    if(n_tree>=size_tree_table) {
      fprintf(stderr,"tree table is full\n"); exit(1);}
    return VISITED;
  } else 
    /* if the connection to the boundary was found,       */
    /* the newly created tree is canceled and             */
    /* GROUNDED is returned to calling procedure, which   */
    /* means the charge was connected to the boundary.    */
    return GROUNDED;
}

void connect_trees(unsigned char **gzw, int ip) 
{
  int t,i2,j2;
  /* connect trees as a chain until the connection to the */
  /* unvisited charge is established.                     */
  while(ip!=0) {
    t=charge_t[ip];  /* pointer to tree_table */
    ip=tree_f[t];
    i2=charge_i[ip]; /* (i2,j12) is the head of the tree. */
    j2=charge_j[ip];
    ip=tree_p[t];    /* ip: pointer to the charge in the previous tree */
    /* connect the head of the tree to the previous tree */
    cutmap(gzw, charge_i[ip],charge_j[ip],i2,j2);
  }
}

/* Connect a pair of unvisited charges thru existing trees  */
/* using a connection less than 'mbl'. If such a connection */
/* is impossible, leave the unvisited charges as they are.  */
/* They will be connected later using larger 'mbl'.         */

int nearest_tree(unsigned char **gzw, int data_w, int data_h, int mbl)
{
  int i,j,i2,j2,ns;
  int k,k_max;
  int n_rest;
  int ip,c;

  n_rest=0;

  for(j=0; j < data_h; j++) {
    if (j%10 == 0){
      fprintf(stdout,"\rprocessing column: %d",j);
      fflush(stdout);
    }
    for(i=0; i < data_w; i++) {
      if(((gzw[i][j]&CHG)==0)||((gzw[i][j]&VIST)!=0)) continue;
                                   /* (i,j) is an unvisited charge.*/
      gzw[i][j] |= BRPT;           /* mark as BRPT                 */
                                   /* initialize tree_table        */
      tree_f[0]=tree_p[0]=0; n_tree=1; 
                                   /* initialize charge_table      */
      charge_t[0]=0; charge_i[0]=i; charge_j[0]=j; n_charge=1;
      c=CHG-(gzw[i][j]&CHG);       /* c: opposite charge           */


      for(k_max=1; k_max <= mbl; k_max++) {

	/* search from a charge in charge_table.                      */
	/* charge_table contains the unvisited charge and charges     */
	/* found in search, and charges connected to them thru trees. */

	for(ip=0; ip < n_charge; ip++) {

	  /* for each charge, pixels inside the size k_max box around it */
	  /* are checked from near to far.                               */

	  for(k=1; k <= k_max; k++) {

	    /* First all points on size k box around the charge are      */
	    /* checked and stored in a table,                            */
	    /* then all points in the table are searched.                */

	    make_search_table(charge_i[ip],charge_j[ip],k, data_w, data_h);

	    for(ns=0; ns < n_search; ns++) {
	      i2=si[ns]; j2=sj[ns];

	      if((i2 == -1)||(j2 == -1)||(i2 == data_w-1)||(j2 == data_h-1))
		/* the charge is located at distance k from the boundary. */
		goto case_BOUNDARY;
	      else if((gzw[i2][j2]&BRPT) != 0) continue;
	      else if(((gzw[i2][j2]&VIST) == 0)&&((gzw[i2][j2]&CHG) == c))
		/* An unvisited charge with opposite charge is found. */
		goto case_OPPOSITE;
	      else if((gzw[i2][j2]&(CHG+VIST)) || (gzw[i2][j2]&GUID))

		/* When a visited charge or guiding center is found,   */
		/* if new_mode==1 then                                 */
		/* trace the existing tree from the charge and add     */
		/* charges connected to the tree to charge_table.      */
		/* When the traced tree has a connection to the        */
		/* boundary, the unvisited charge is connected to the  */
		/* charge(i2,j2).                                      */
		if(new_mode!=1) continue;
		else if(trace_tree(gzw,i2,j2,ip,data_w,data_h)==GROUNDED) goto case_GROUNDED;
	    }
	  }
	}
      }
      n_rest++; /* count unvisited charges which are not discharged. */
      goto clear_BRPT;

    case_OPPOSITE:
      /* mark the found unvisited charge. */
      gzw[i2][j2] |= VIST;
    case_BOUNDARY:
    case_GROUNDED:
      /* mark the unvisited charge.*/
      gzw[i][j] |= VIST;
      /* connect charges and trees chainly. */
      cutmap(gzw, charge_i[ip],charge_j[ip],i2,j2);
      connect_trees(gzw, ip);

    clear_BRPT:
      for(ip=0; ip<n_charge; ip++) gzw[charge_i[ip]][charge_j[ip]] &= 255-BRPT;
    }
  }
  return n_rest;
}

void cutmap(unsigned char **gzw, int i1,int j1,int i2,int j2)
{
  int m,ki,kj,kk;
  int ii,ji;
  ki=(i1<i2 ? i2-i1 : i1-i2);
  kj=(j1<j2 ? j2-j1 : j1-j2);
  kk=(ki<kj ? kj : ki);
  if(kk==0) {fprintf(stderr,"error: subroutine cutmap! kk=0\n"); return;}
  for(m=1; m<=kk; m++) {
    ii=i1+(i2-i1)*m/kk;
    ji=j1+(j2-j1)*m/kk;
    if(j1-j2==kk) ji++;
    if(i1-i2==kk) ii++;
    if(ii<0) ii=0;
    if(ji<0) ji=0;
    if((gzw[ii][ji]&CUT)==0) 
      gzw[ii][ji] |= CUT;
  }
}
