/*************************************************************************

  snaphu tile-mode source file
  Written by Curtis W. Chen
  Copyright 2002 Board of Trustees, Leland Stanford Jr. University
  Please see the supporting documentation for terms of use.
  No warranty.

*************************************************************************/

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
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "snaphu.h"



/* static (local) function prototypes */
static
long ThickenCosts(incrcostT **incrcosts, long nrow, long ncol);
static
nodeT *RegionsNeighborNode(nodeT *node1, long *arcnumptr, nodeT **nodes, 
                           long *arcrowptr, long *arccolptr, 
                           long nrow, long ncol);
static
int ClearBuckets(bucketT *bkts);
static
int MergeRegions(nodeT **nodes, nodeT *source, long *regionsizes, 
                  long closestregion, long nrow, long ncol);
static
int RenumberRegion(nodeT **nodes, nodeT *source, long newnum, 
                   long nrow, long ncol);
static
int ReadNextRegion(long tilerow, long tilecol, long nlines, long linelen,
                   outfileT *outfiles, paramT *params, 
                   short ***nextregionsptr, float ***nextunwphaseptr,
                   void ***nextcostsptr, 
                   long *nextnrowptr, long *nextncolptr);
static
int SetTileReadParams(tileparamT *tileparams, long nexttilenlines, 
                      long nexttilelinelen, long tilerow, long tilecol, 
                      long nlines, long linelen, paramT *params);
static
int ReadEdgesAboveAndBelow(long tilerow, long tilecol, long nlines, 
                           long linelen, paramT *params, outfileT *outfiles, 
                           short *regionsabove, short *regionsbelow,
                           float *unwphaseabove, float *unwphasebelow,
                           void *costsabove, void *costsbelow);
static
int TraceRegions(short **regions, short **nextregions, short **lastregions, 
                 short *regionsabove, short *regionsbelow, float **unwphase, 
                 float **nextunwphase, float **lastunwphase, 
                 float *unwphaseabove, float *unwphasebelow, void **costs, 
                 void **nextcosts, void **lastcosts, void *costsabove, 
                 void *costsbelow, long prevnrow, long prevncol, long tilerow,
                 long tilecol, long nrow, long ncol, nodeT **scndrynodes,
                 nodesuppT **nodesupp, scndryarcT **scndryarcs, 
                 long ***scndrycosts, int *nscndrynodes, 
                 int *nscndryarcs, long *totarclens, short **bulkoffsets, 
                 paramT *params);
static
long FindNumPathsOut(nodeT *from, paramT *params, long tilerow, long tilecol, 
                     long nnrow, long nncol, short **regions, 
                     short **nextregions, short **lastregions,
                     short *regionsabove, short *regionsbelow, long prevncol);
static
int RegionTraceCheckNeighbors(nodeT *from, nodeT **nextnodeptr, 
                              nodeT **primarynodes, short **regions, 
                              short **nextregions, short **lastregions, 
                              short *regionsabove, short *regionsbelow,  
                              long tilerow, long tilecol, long nnrow, 
                              long nncol, nodeT **scndrynodes, 
                              nodesuppT **nodesupp, scndryarcT **scndryarcs, 
                              long *nnewnodesptr, long *nnewarcsptr, 
                              long flowmax, long nrow, long ncol, 
                              long prevnrow, long prevncol, paramT *params, 
                              void **costs, void **rightedgecosts, 
                              void **loweredgecosts, void **leftedgecosts, 
                              void **upperedgecosts, short **flows, 
                              short **rightedgeflows, short **loweredgeflows,
                              short **leftedgeflows, short **upperedgeflows,
                              long ***scndrycosts, 
                              nodeT ***updatednontilenodesptr, 
                              long *nupdatednontilenodesptr, 
                              long *updatednontilenodesizeptr,
                              short **inontilenodeoutarcptr, 
                              long *totarclenptr);
static
int SetUpperEdge(long ncol, long tilerow, long tilecol, void **voidcosts, 
                 void *voidcostsabove, float **unwphase, 
                 float *unwphaseabove, void **voidupperedgecosts, 
                 short **upperedgeflows, paramT *params, short **bulkoffsets);
static
int SetLowerEdge(long nrow, long ncol, long tilerow, long tilecol, 
                 void **voidcosts, void *voidcostsbelow, 
                 float **unwphase, float *unwphasebelow, 
                 void **voidloweredgecosts, short **loweredgeflows, 
                 paramT *params, short **bulkoffsets);
static
int SetLeftEdge(long nrow, long prevncol, long tilerow, long tilecol, 
                void **voidcosts, void **voidlastcosts, float **unwphase, 
                float **lastunwphase, void **voidleftedgecosts, 
                short **leftedgeflows, paramT *params, short **bulkoffsets);
static
int SetRightEdge(long nrow, long ncol, long tilerow, long tilecol, 
                 void **voidcosts, void **voidnextcosts, 
                 float **unwphase, float **nextunwphase, 
                 void **voidrightedgecosts, short **rightedgeflows, 
                 paramT *params, short **bulkoffsets);
static
short AvgSigSq(short sigsq1, short sigsq2);
static
int TraceSecondaryArc(nodeT *primaryhead, nodeT **scndrynodes, 
                      nodesuppT **nodesupp, scndryarcT **scndryarcs, 
                      long ***scndrycosts, long *nnewnodesptr, 
                      long *nnewarcsptr, long tilerow, long tilecol, 
                      long flowmax, long nrow, long ncol, 
                      long prevnrow, long prevncol, paramT *params, 
                      void **tilecosts, void **rightedgecosts, 
                      void **loweredgecosts, void **leftedgecosts,
                      void **upperedgecosts, short **tileflows, 
                      short **rightedgeflows, short **loweredgeflows, 
                      short **leftedgeflows, short **upperedgeflows,
                      nodeT ***updatednontilenodesptr, 
                      long *nupdatednontilenodesptr, 
                      long *updatednontilenodesizeptr,
                      short **inontilenodeoutarcptr, long *totarclenptr);
static
nodeT *FindScndryNode(nodeT **scndrynodes, nodesuppT **nodesupp, 
                      long tilenum, long primaryrow, long primarycol);
static
int IntegrateSecondaryFlows(long linelen, long nlines, nodeT **scndrynodes, 
                            nodesuppT **nodesupp, scndryarcT **scndryarcs, 
                            int *nscndryarcs, short **scndryflows, 
                            short **bulkoffsets, outfileT *outfiles, 
                            paramT *params);
static
int ParseSecondaryFlows(long tilenum, int *nscndryarcs, short **tileflows, 
                        short **regions, short **scndryflows, 
                        nodesuppT **nodesupp, scndryarcT **scndryarcs, 
                        long nrow, long ncol, long ntilerow, long ntilecol,
                        paramT *params);
static
int AssembleTileConnComps(long linelen, long nlines,
                          outfileT *outfiles, paramT *params);
static
int ConnCompSizeNPixCompare(const void *ptr1, const void *ptr2);



/* function: SetupTile()
 * ---------------------
 * Sets up tile parameters and output file names for the current tile.
 */
int SetupTile(long nlines, long linelen, paramT *params, 
              tileparamT *tileparams, outfileT *outfiles, 
              outfileT *tileoutfiles, long tilerow, long tilecol){

  long ni, nj;
  char tempstring[MAXTMPSTRLEN], path[MAXSTRLEN], basename[MAXSTRLEN];
  char *tiledir;


  /* set parameters for current tile */
  ni=ceil((nlines+(params->ntilerow-1)*params->rowovrlp)
          /(double )params->ntilerow);
  nj=ceil((linelen+(params->ntilecol-1)*params->colovrlp)
          /(double )params->ntilecol);
  tileparams->firstrow=tilerow*(ni-params->rowovrlp);
  tileparams->firstcol=tilecol*(nj-params->colovrlp);
  if(tilerow==params->ntilerow-1){
    tileparams->nrow=nlines-(params->ntilerow-1)*(ni-params->rowovrlp);
  }else{
    tileparams->nrow=ni;
  }
  if(tilecol==params->ntilecol-1){
    tileparams->ncol=linelen-(params->ntilecol-1)*(nj-params->colovrlp);
  }else{
    tileparams->ncol=nj;
  }

  /* error checking on tile size */
  if(params->minregionsize > (tileparams->nrow)*(tileparams->ncol)){
    fflush(NULL);
    fprintf(sp0,"Minimum region size cannot exceed tile size\nAbort\n");
    exit(ABNORMAL_EXIT);
  }

  /* set output files */
  tiledir=params->tiledir;
  ParseFilename(outfiles->outfile,path,basename);
  sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
          tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
  StrNCopy(tileoutfiles->outfile,tempstring,MAXSTRLEN);
  if(strlen(outfiles->initfile)){
    ParseFilename(outfiles->initfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->initfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->initfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->flowfile)){
    ParseFilename(outfiles->flowfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->flowfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->flowfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->eifile)){
    ParseFilename(outfiles->eifile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->eifile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->eifile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->rowcostfile)){
    ParseFilename(outfiles->rowcostfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->rowcostfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->rowcostfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->colcostfile)){
    ParseFilename(outfiles->colcostfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->colcostfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->colcostfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->mstrowcostfile)){
    ParseFilename(outfiles->mstrowcostfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->mstrowcostfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->mstrowcostfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->mstcolcostfile)){
    ParseFilename(outfiles->mstcolcostfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->mstcolcostfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->mstcolcostfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->mstcostsfile)){
    ParseFilename(outfiles->mstcostsfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->mstcostsfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->mstcostsfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->corrdumpfile)){
    ParseFilename(outfiles->corrdumpfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->corrdumpfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->corrdumpfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->rawcorrdumpfile)){
    ParseFilename(outfiles->rawcorrdumpfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->rawcorrdumpfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->rawcorrdumpfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->conncompfile)){
    ParseFilename(outfiles->conncompfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->conncompfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->conncompfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->costoutfile)){
    ParseFilename(outfiles->costoutfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->costoutfile,tempstring,MAXSTRLEN);
  }else{
    sprintf(tempstring,"%s/%s%s%ld_%ld.%ld",
            tiledir,TMPTILEROOT,TMPTILECOSTSUFFIX,tilerow,tilecol,
            tileparams->ncol);
    StrNCopy(tileoutfiles->costoutfile,tempstring,MAXSTRLEN);
  }
  if(strlen(outfiles->logfile)){
    ParseFilename(outfiles->logfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld",
            tiledir,TMPTILEROOT,basename,tilerow,tilecol,tileparams->ncol);
    StrNCopy(tileoutfiles->logfile,tempstring,MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->logfile,"",MAXSTRLEN);
  }
  tileoutfiles->outfileformat=TMPTILEOUTFORMAT;

  /* done */
  return(0);

}


/* function: SetUpDoTileMask()
 * ---------------------------
 * Read the tile mask if a file name is specified in the infiles structure,
 * otherwise return an array of all ones.
 */
signed char **SetUpDoTileMask(infileT *infiles, long ntilerow, long ntilecol){

  long row, col;
  signed char **dotilemask;
  tileparamT readparams[1];


  /* initialize stack structures to zero for good measure */
  memset(readparams,0,sizeof(tileparamT));
  
  /* get memory */
  dotilemask=(signed char **)Get2DMem(ntilerow,ntilecol,sizeof(signed char *),
                                      sizeof(signed char));

  /* see if a file name was passed */
  if(strlen(infiles->dotilemaskfile)){

    /* read the input file */
    readparams->nrow=ntilerow;
    readparams->ncol=ntilecol;
    readparams->firstrow=0;
    readparams->firstcol=0;
    Read2DArray((void ***)&dotilemask,infiles->dotilemaskfile,ntilecol,
                ntilerow,readparams,sizeof(signed char *),sizeof(signed char));

  }else{

    /* set array to be all ones */
    for(row=0;row<ntilerow;row++){
      for(col=0;col<ntilecol;col++){
        dotilemask[row][col]=1;
      }
    }
  }

  /* return the array pointer */
  return(dotilemask);

}


/* function: GrowRegions()
 * -----------------------
 * Grows contiguous regions demarcated by arcs whose residual costs are
 * less than some threshold.  Numbers the regions sequentially from 0.
 */
int GrowRegions(void **costs, short **flows, long nrow, long ncol, 
                incrcostT **incrcosts, outfileT *outfiles, 
                tileparamT *tileparams, paramT *params){

  long i, row, col, maxcol;
  long arcrow, arccol, arcnum, fromdist, arcdist;
  long regioncounter, *regionsizes, regionsizeslen, *thisregionsize;
  long closestregiondist, closestregion, lastfromdist;
  long costthresh, minsize, maxcost;
  long costtypesize;
  short **regions;
  nodeT **nodes;
  nodeT *source, *from, *to, *ground;
  char regionfile[MAXSTRLEN];
  bucketT bkts[1];
  void **growregionscosts;
  tileparamT temptileparams[1];
  void (*tempcalccostfnptr)();      
  long (*tempevalcostfnptr)();      

  
  /* initialize stack structures to zero for good measure */
  memset(regionfile,0,MAXSTRLEN);
  memset(bkts,0,sizeof(bucketT));
  memset(temptileparams,0,sizeof(tileparamT));
  
  /* set up */
  fprintf(sp1,"Growing reliable regions\n");
  minsize=params->minregionsize;
  costthresh=params->tilecostthresh;
  closestregion=0;

  /* store current values of CalcCost and EvalCost function pointers */
  tempcalccostfnptr=CalcCost;
  tempevalcostfnptr=EvalCost;

  /* reread statistical costs from stored file if costs array is for Lp mode */
  if(params->p >= 0){
    if(params->costmode==TOPO){
      costtypesize=sizeof(costT);
      CalcCost=CalcCostTopo;
      EvalCost=EvalCostTopo;
    }else if(params->costmode==DEFO){
      costtypesize=sizeof(costT);
      CalcCost=CalcCostDefo;
      EvalCost=EvalCostDefo;
    }else if(params->costmode==SMOOTH){
      costtypesize=sizeof(smoothcostT);
      CalcCost=CalcCostSmooth;
      EvalCost=EvalCostSmooth;
    }else{
      fflush(NULL);
      fprintf(sp0,"Illegal cost mode in GrowRegions().  This is a bug.\n");
      exit(ABNORMAL_EXIT);
    }
    temptileparams->firstrow=0;
    temptileparams->firstcol=0;
    temptileparams->nrow=nrow;
    temptileparams->ncol=ncol;
    growregionscosts=NULL;
    Read2DRowColFile((void ***)&growregionscosts,outfiles->costoutfile,
                     ncol,nrow,temptileparams,costtypesize);
  }else{
    growregionscosts=costs;
  }

  /* loop over all arcs */
  for(arcrow=0;arcrow<2*nrow-1;arcrow++){
    if(arcrow<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(arccol=0;arccol<maxcol;arccol++){

      /* compute incremental costs of unit flows in either direction */
      ReCalcCost(growregionscosts,incrcosts,flows[arcrow][arccol],
                 arcrow,arccol,1,nrow,params);

      /* store lesser of incremental costs in first field */
      if(incrcosts[arcrow][arccol].negcost<incrcosts[arcrow][arccol].poscost){
        incrcosts[arcrow][arccol].poscost=incrcosts[arcrow][arccol].negcost;
      }

      /* subtract costthresh and take negative of costs, then clip to zero */
      incrcosts[arcrow][arccol].poscost
        =-(incrcosts[arcrow][arccol].poscost-costthresh);
      if(incrcosts[arcrow][arccol].poscost<0){
        incrcosts[arcrow][arccol].poscost=0;
      } 
    }
  }

  /* thicken the costs arrays; results stored in negcost field */
  maxcost=ThickenCosts(incrcosts,nrow,ncol);
 
  /* initialize nodes and buckets for region growing */
  ground=NULL;
  nodes=(nodeT **)Get2DMem(nrow,ncol,sizeof(nodeT *),sizeof(nodeT));
  InitNodeNums(nrow,ncol,nodes,ground);
  InitNodes(nrow,ncol,nodes,ground);
  bkts->size=maxcost+2;
  bkts->minind=0;
  bkts->maxind=bkts->size-1;
  bkts->curr=0;
  bkts->wrapped=FALSE;
  bkts->bucketbase=(nodeT **)MAlloc(bkts->size*sizeof(nodeT *));
  bkts->bucket=bkts->bucketbase;
  for(i=0;i<bkts->size;i++){
    bkts->bucket[i]=NULL;
  }

  /* initialize region variables */
  regioncounter=-1;
  regionsizeslen=INITARRSIZE;
  regionsizes=(long *)MAlloc(regionsizeslen*sizeof(long));
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      nodes[row][col].incost=-1;
    }
  }

  /* loop over all nodes (pixels) to make sure each is in a group */
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){

      /* see if node is not in a group */
      if(nodes[row][col].incost<0){

        /* clear the buckets */
        ClearBuckets(bkts);

        /* make node source and put it in the first bucket */
        source=&nodes[row][col];
        source->next=NULL;
        source->prev=NULL;
        source->group=INBUCKET;
        source->outcost=0;
        bkts->bucket[0]=source;
        bkts->curr=0;
        lastfromdist=0;

        /* increment the region counter */
        if(++regioncounter>=regionsizeslen){
          regionsizeslen+=INITARRSIZE;
          regionsizes=(long *)ReAlloc(regionsizes,
                                       regionsizeslen*sizeof(long));
        }
        thisregionsize=&regionsizes[regioncounter];

        /* set up */
        (*thisregionsize)=0;
        closestregiondist=VERYFAR;

        /* loop to grow region */
        while(TRUE){

          /* set from node to closest node in circular bucket structure */
          from=ClosestNode(bkts);
          
          /* break if we can't grow any more and the region is big enough */
          if(from==NULL){
            if(*thisregionsize>=minsize){

              /* no more nonregion nodes, and current region is big enough */
              break;

            }else{

              /* no more nonregion nodes, but current region still too small */
              /* merge with another region */
              MergeRegions(nodes,source,regionsizes,closestregion,nrow,ncol);
              regioncounter--;
              break;

            }
          }else{
            fromdist=from->outcost;
            if(fromdist>lastfromdist){
              if(regionsizes[regioncounter]>=minsize){

                /* region grown to all nodes within mincost, is big enough */
                break;

              }
              if(fromdist>closestregiondist){

                /* another region closer than new node, so merge regions */
                MergeRegions(nodes,source,regionsizes,closestregion,nrow,ncol);
                regioncounter--;
                break;
              }
            }
          }

          /* make from node a part of the current region */
          from->incost=regioncounter;
          (*thisregionsize)++;
          lastfromdist=fromdist;

          /* scan from's neighbors */
          arcnum=0;
          while((to=RegionsNeighborNode(from,&arcnum,nodes,
                                        &arcrow,&arccol,nrow,ncol))!=NULL){
           
            /* get cost of arc to the to node */
            arcdist=incrcosts[arcrow][arccol].negcost;

            /* see if to node is already in another region */
            if(to->incost>=0){

              /* keep track of which neighboring region is closest */
              if(to->incost!=regioncounter && arcdist<closestregiondist){
                closestregiondist=arcdist;
                closestregion=to->incost;
              }

            }else{

              /* to node is not in another region */
              /* compare distance of new nodes to temp labels */
              if(arcdist<(to->outcost)){

                /* if to node is already in a (circular) bucket, remove it */
                if(to->group==INBUCKET){
                  BucketRemove(to,to->outcost,bkts);
                }
                
                /* update to node */
                to->outcost=arcdist;
                to->pred=from;

                /* insert to node into appropriate (circular) bucket */
                BucketInsert(to,arcdist,bkts);
                if(arcdist<bkts->curr){
                  bkts->curr=arcdist;
                }
              }
            }
          }
        }
      }
    }
  }
  fprintf(sp2,"Tile partitioned into %ld regions\n",regioncounter+1);

  /* write regions array */
  /* write as shorts if multiple tiles */
  if(params->ntilerow > 1 || params->ntilecol>1){
    regions=(short **)Get2DMem(nrow,ncol,sizeof(short *),sizeof(short));
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        if(nodes[row][col].incost>LARGESHORT){
          fflush(NULL);
          fprintf(sp0,
                  "Number of regions in tile exceeds max allowed\nAbort\n");
          exit(ABNORMAL_EXIT);
        }
        regions[row][col]=nodes[row][col].incost;
      }
    }
    sprintf(regionfile,"%s%s",outfiles->outfile,REGIONSUFFIX);
    fprintf(sp2,"Writing region data to file %s\n",regionfile);
    Write2DArray((void **)regions,regionfile,nrow,ncol,sizeof(short));
  }else{
    regions=NULL;
  }

  /* return CalcCost and EvalCost function pointers to initial values */
  CalcCost=tempcalccostfnptr;
  EvalCost=tempevalcostfnptr;

  /* free memory */
  if(params->p >= 0){
    Free2DArray((void **)growregionscosts,2*nrow-1);
  }
  Free2DArray((void **)nodes,nrow);
  if(regions!=NULL){
    Free2DArray((void **)regions,nrow);
  }
  free(bkts->bucketbase);

  /* done */
  return(0);

}


/* function: GrowConnCompsMask()
 * -----------------------------
 * Grows contiguous regions demarcated by arcs whose residual costs are
 * less than some threshold.  Numbers the regions sequentially from 1.
 * Writes out byte file of connected component mask, with 0 for any pixels
 * not assigned to a component.
 */
int GrowConnCompsMask(void **costs, short **flows, long nrow, long ncol, 
                      incrcostT **incrcosts, outfileT *outfiles, 
                      paramT *params){

  long i, row, col, maxcol;
  long arcrow, arccol, arcnum;
  long regioncounter, *regionsizes, regionsizeslen, *thisregionsize;
  long *sortedregionsizes;
  long costthresh, minsize, maxncomps, ntied, newnum;
  unsigned long outtypemax, outtypesize;
  nodeT **nodes;
  nodeT *source, *from, *to, *ground;
  unsigned char *ucharbuf;
  unsigned int *uintbuf;
  void *outbufptr;
  bucketT bkts[1];
  char realoutfile[MAXSTRLEN];
  FILE *conncompfp;


  /* initialize stack structures to zero for good measure */
  memset(bkts,0,sizeof(bucketT));
  
  /* error checking */
  fprintf(sp1,"Growing connected component mask\n");
  minsize=params->minconncompfrac*nrow*ncol;
  maxncomps=params->maxncomps;
  costthresh=params->conncompthresh;
  if(minsize>nrow*ncol){
    fflush(NULL);
    fprintf(sp0,"Minimum region size cannot exceed tile size\nAbort\n");
    exit(ABNORMAL_EXIT);
  }

  /* loop over all arcs */
  for(arcrow=0;arcrow<2*nrow-1;arcrow++){
    if(arcrow<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(arccol=0;arccol<maxcol;arccol++){

      /* compute incremental costs of unit flows in either direction */
      ReCalcCost(costs,incrcosts,flows[arcrow][arccol],
                 arcrow,arccol,1,nrow,params);

      /* store lesser of incremental costs in first field */
      if(incrcosts[arcrow][arccol].negcost<incrcosts[arcrow][arccol].poscost){
        incrcosts[arcrow][arccol].poscost=incrcosts[arcrow][arccol].negcost;
      }

      /* subtract costthresh and take negative of costs, then clip to zero */
      incrcosts[arcrow][arccol].poscost
        =-(incrcosts[arcrow][arccol].poscost-costthresh);
      if(incrcosts[arcrow][arccol].poscost<0){
        incrcosts[arcrow][arccol].poscost=0;
      } 
    }
  }

  /* thicken the costs arrays; results stored in negcost field */
  ThickenCosts(incrcosts,nrow,ncol);
 
  /* initialize nodes and buckets for region growing */
  ground=NULL;
  nodes=(nodeT **)Get2DMem(nrow,ncol,sizeof(nodeT *),sizeof(nodeT));
  InitNodeNums(nrow,ncol,nodes,ground);
  InitNodes(nrow,ncol,nodes,ground);
  bkts->size=1;
  bkts->minind=0;
  bkts->maxind=0;
  bkts->wrapped=FALSE;
  bkts->bucketbase=(nodeT **)MAlloc(sizeof(nodeT *));
  bkts->bucket=bkts->bucketbase;
  bkts->bucket[0]=NULL;
  
  /* initialize region variables */
  regioncounter=0;
  regionsizeslen=INITARRSIZE;
  regionsizes=(long *)MAlloc(regionsizeslen*sizeof(long));
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      nodes[row][col].incost=-1;
    }
  }

  /* loop over all nodes (pixels) to make sure each is in a group */
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){

      /* see if node is not in a group */
      if(nodes[row][col].incost<0){

        /* clear the buckets */
        ClearBuckets(bkts);

        /* make node source and put it in the first bucket */
        source=&nodes[row][col];
        source->next=NULL;
        source->prev=NULL;
        source->group=INBUCKET;
        source->outcost=0;
        bkts->bucket[0]=source;
        bkts->curr=0;

        /* increment the region counter */
        if(++regioncounter>=regionsizeslen){
          regionsizeslen+=INITARRSIZE;
          regionsizes=(long *)ReAlloc(regionsizes,
                                       regionsizeslen*sizeof(long));
        }
        thisregionsize=&regionsizes[regioncounter];

        /* set up */
        (*thisregionsize)=0;

        /* loop to grow region */
        while(TRUE){

          /* set from node to closest node in circular bucket structure */
          from=ClosestNode(bkts);
          
          /* break if we can't grow any more and the region is big enough */
          if(from==NULL){
            if(regionsizes[regioncounter]>=minsize){

              /* no more nonregion nodes, and current region is big enough */
              break;

            }else{

              /* no more nonregion nodes, but current region still too small */
              /* zero out the region */
              RenumberRegion(nodes,source,0,nrow,ncol);
              regioncounter--;
              break;

            }
          }

          /* make from node a part of the current region */
          from->incost=regioncounter;
          (*thisregionsize)++;

          /* scan from's neighbors */
          arcnum=0;
          while((to=RegionsNeighborNode(from,&arcnum,nodes,
                                        &arcrow,&arccol,nrow,ncol))!=NULL){
           
            /* see if to can be reached */
            if(to->incost<0 && incrcosts[arcrow][arccol].negcost==0 
               && to->group!=INBUCKET){

              /* update to node */
              to->pred=from;
              BucketInsert(to,0,bkts);

            }
          }
        }
      }
    }
  }
  fprintf(sp2,"%ld connected components formed\n",regioncounter);

  /* make sure we don't have too many components */
  if(regioncounter>maxncomps){

    /* copy regionsizes array and sort to find new minimum region size */
    fprintf(sp2,"Keeping only %ld connected components\n",maxncomps);
    sortedregionsizes=(long *)MAlloc(regioncounter*sizeof(long));
    for(i=0;i<regioncounter;i++){
      sortedregionsizes[i]=regionsizes[i+1];
    }
    qsort((void *)sortedregionsizes,regioncounter,sizeof(long),LongCompare);
    minsize=sortedregionsizes[regioncounter-maxncomps];

    /* see how many regions of size minsize still need zeroing */
    ntied=0;
    i=regioncounter-maxncomps-1;
    while(i>=0 && sortedregionsizes[i]==minsize){
      ntied++;
      i--;
    }

    /* zero out regions that are too small */
    newnum=-1;
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        i=nodes[row][col].incost;
        if(i>0){
          if(regionsizes[i]<minsize 
             || (regionsizes[i]==minsize && (ntied--)>0)){

            /* region too small, so zero it out */
            RenumberRegion(nodes,&(nodes[row][col]),0,nrow,ncol);

          }else{

            /* keep region, assign it new region number */
            /* temporarily assign negative of new number to avoid collisions */
            RenumberRegion(nodes,&(nodes[row][col]),newnum--,nrow,ncol);

          }
        }
      }
    }

    /* flip temporary negative region numbers so they are positive */
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        nodes[row][col].incost=-nodes[row][col].incost;
      }
    }
  }

  /* write connected components as appropriate data type */
  ucharbuf=(unsigned char *)MAlloc(ncol*sizeof(unsigned char));
  uintbuf=(unsigned int *)MAlloc(ncol*sizeof(unsigned int));
  if(params->conncompouttype==CONNCOMPOUTTYPEUCHAR){
    outtypemax=UCHAR_MAX;
    outtypesize=(int )sizeof(unsigned char);
    outbufptr=(void *)ucharbuf;
  }else if(params->conncompouttype==CONNCOMPOUTTYPEUINT){
    outtypemax=UINT_MAX;
    outtypesize=(int )sizeof(unsigned int);
    outbufptr=(void *)uintbuf;
  }else{
    fflush(NULL);
    fprintf(sp0,"Bad conncompouttype in GrowConnCompMask()\n");
    exit(ABNORMAL_EXIT);
  }
  fprintf(sp1,"Writing connected components to file %s"
          " as %d-byte unsigned ints\n",
          outfiles->conncompfile,((int )outtypesize));
  conncompfp=OpenOutputFile(outfiles->conncompfile,realoutfile);
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      if(nodes[row][col].incost>outtypemax){
        fflush(NULL);
        fprintf(sp0,"Number of connected components too large for output type\n"
                "Abort\n");
        exit(ABNORMAL_EXIT);
      }
      uintbuf[col]=(unsigned int)(nodes[row][col].incost);
    }
    if(params->conncompouttype==CONNCOMPOUTTYPEUCHAR){
      for(col=0;col<ncol;col++){
        ucharbuf[col]=(unsigned char )uintbuf[col];
      }
    }
    if(fwrite(outbufptr,outtypesize,ncol,conncompfp)!=ncol){
      fflush(NULL);
      fprintf(sp0,"Error while writing to file %s (device full?)\nAbort\n",
              realoutfile);
      exit(ABNORMAL_EXIT);
    }
  }
  if(fclose(conncompfp)){
    fflush(NULL);
    fprintf(sp0,"WARNING: problem closing file %s (disk full?)\n",
            outfiles->conncompfile);
  }

  /* free memory */
  Free2DArray((void **)nodes,nrow);
  free(bkts->bucketbase);
  free(uintbuf);
  free(ucharbuf);
  return(0);
  
}


/* function: ThickenCosts()
 * ------------------------
 */
static
long ThickenCosts(incrcostT **incrcosts, long nrow, long ncol){

  long row, col, templong, maxcost;
  double n;
  

  /* initialize variable storing maximum cost */
  maxcost=-LARGEINT;

  /* loop over row arcs and convolve */
  for(row=0;row<nrow-1;row++){
    for(col=0;col<ncol;col++){
      templong=2*incrcosts[row][col].poscost;
      n=2.0;
      if(col!=0){
        templong+=incrcosts[row][col-1].poscost;
        n+=1.0;
      }
      if(col!=ncol-1){
        templong+=incrcosts[row][col+1].poscost;
        n+=1.0;
      }
      templong=LRound(templong/n);
      if(templong>LARGESHORT){
        fflush(NULL);
        fprintf(sp0,"WARNING: COSTS CLIPPED IN ThickenCosts()\n");
        incrcosts[row][col].negcost=LARGESHORT;
      }else{
        incrcosts[row][col].negcost=templong;
      }
      if(incrcosts[row][col].negcost>maxcost){
        maxcost=incrcosts[row][col].negcost;
      }
    }
  }

  /* loop over column arcs and convolve */
  for(row=nrow-1;row<2*nrow-1;row++){
    for(col=0;col<ncol-1;col++){
      templong=2*incrcosts[row][col].poscost;
      n=2.0;
      if(row!=nrow-1){
        templong+=incrcosts[row-1][col].poscost;
        n+=1.0;
      }
      if(row!=2*nrow-2){
        templong+=incrcosts[row+1][col].poscost;
        n+=1.0;
      }
      templong=LRound(templong/n);
      if(templong>LARGESHORT){
        fflush(NULL);
        fprintf(sp0,"WARNING: COSTS CLIPPED IN ThickenCosts()\n");
        incrcosts[row][col].negcost=LARGESHORT;
      }else{      
        incrcosts[row][col].negcost=templong;
      }
      if(incrcosts[row][col].negcost>maxcost){
        maxcost=incrcosts[row][col].negcost;
      }
    }
  }

  /* return maximum cost */
  return(maxcost);

}


/* function: RegionsNeighborNode()
 * -------------------------------
 * Return the neighboring node of the given node corresponding to the
 * given arc number.
 */
static
nodeT *RegionsNeighborNode(nodeT *node1, long *arcnumptr, nodeT **nodes, 
                           long *arcrowptr, long *arccolptr, 
                           long nrow, long ncol){
  
  long row, col;

  row=node1->row;
  col=node1->col;

  while(TRUE){
    switch((*arcnumptr)++){
    case 0:
      if(col!=ncol-1){
        *arcrowptr=nrow-1+row;
        *arccolptr=col;
        return(&nodes[row][col+1]);
      }
      break;
    case 1:
      if(row!=nrow-1){
      *arcrowptr=row;
      *arccolptr=col;
        return(&nodes[row+1][col]);
      }
      break;
    case 2:
      if(col!=0){
        *arcrowptr=nrow-1+row;
        *arccolptr=col-1;
        return(&nodes[row][col-1]);
      }
      break;
    case 3:
      if(row!=0){
        *arcrowptr=row-1;
        *arccolptr=col;
        return(&nodes[row-1][col]);
      }
      break;
    default:
      return(NULL);
    }
  }
}


/* function: ClearBuckets()
 * ------------------------
 * Removes any nodes in the bucket data structure passed, and resets
 * their distances to VERYFAR.  Assumes bukets indexed from 0.
 */
static
int ClearBuckets(bucketT *bkts){

  nodeT *currentnode, *nextnode;
  long i;

  /* loop over all buckets */
  for(i=0;i<bkts->size;i++){

    /* clear the bucket */
    nextnode=bkts->bucketbase[i];
    while(nextnode!=NULL){
      currentnode=nextnode;
      nextnode=currentnode->next;
      currentnode->group=NOTINBUCKET;
      currentnode->outcost=VERYFAR;
      currentnode->pred=NULL;
    }
    bkts->bucketbase[i]=NULL;
  }

  /* reset bucket parameters */
  bkts->minind=0;
  bkts->maxind=bkts->size-1;
  bkts->wrapped=FALSE;

  /* done */
  return(0);
}


/* function: MergeRegions()
 * ------------------------
 * 
 */
static
int MergeRegions(nodeT **nodes, nodeT *source, long *regionsizes, 
                 long closestregion, long nrow, long ncol){

  long nextnodelistlen, nextnodelistnext, arcnum, arcrow, arccol, regionnum;
  nodeT *from, *to, **nextnodelist;

  
  /* initialize */
  nextnodelistlen=INITARRSIZE;
  nextnodelist=(nodeT **)MAlloc(nextnodelistlen*sizeof(nodeT **));
  nextnodelist[0]=source;
  nextnodelistnext=1;
  regionnum=source->incost;


  /* find all nodes in current region and switch their regions */
  while(nextnodelistnext){
    from=nextnodelist[--nextnodelistnext];
    from->incost=closestregion;
    arcnum=0;
    while((to=RegionsNeighborNode(from,&arcnum,nodes,
                                  &arcrow,&arccol,nrow,ncol))!=NULL){
      if(to->incost==regionnum){
        if(nextnodelistnext>=nextnodelistlen){
          nextnodelistlen+=INITARRSIZE;
          nextnodelist=(nodeT **)ReAlloc(nextnodelist,
                                         nextnodelistlen*sizeof(nodeT *));
        }
        nextnodelist[nextnodelistnext++]=to;
      }
    }
  }

  /* update size of region to which we are merging */
  regionsizes[closestregion]+=regionsizes[regionnum];

  /* free memory */
  free(nextnodelist);
  return(0);
  
}


/* function: RenumberRegion()
 * --------------------------
 * 
 */
static
int RenumberRegion(nodeT **nodes, nodeT *source, long newnum, 
                   long nrow, long ncol){

  long nextnodelistlen, nextnodelistnext, arcnum, arcrow, arccol, regionnum;
  nodeT *from, *to, **nextnodelist;

  
  /* initialize */
  nextnodelistlen=INITARRSIZE;
  nextnodelist=(nodeT **)MAlloc(nextnodelistlen*sizeof(nodeT **));
  nextnodelist[0]=source;
  nextnodelistnext=1;
  regionnum=source->incost;


  /* find all nodes in current region and switch their regions */
  while(nextnodelistnext){
    from=nextnodelist[--nextnodelistnext];
    from->incost=newnum;
    arcnum=0;
    while((to=RegionsNeighborNode(from,&arcnum,nodes,
                                  &arcrow,&arccol,nrow,ncol))!=NULL){
      if(to->incost==regionnum){
        if(nextnodelistnext>=nextnodelistlen){
          nextnodelistlen+=INITARRSIZE;
          nextnodelist=(nodeT **)ReAlloc(nextnodelist,
                                         nextnodelistlen*sizeof(nodeT *));
        }
        nextnodelist[nextnodelistnext++]=to;
      }
    }
  }

  /* free memory */
  free(nextnodelist);
  return(0);
  
}


/* function: AssembleTiles()
 * -------------------------
 */
int AssembleTiles(outfileT *outfiles, paramT *params, 
                  long nlines, long linelen){

  long tilerow, tilecol, ntilerow, ntilecol, ntiles, rowovrlp, colovrlp;
  long i, j, k, ni, nj, dummylong, costtypesize;
  long nrow, ncol, prevnrow, prevncol, nextnrow, nextncol;
  long n, ncycle, nflowdone, nflow, candidatelistsize, candidatebagsize;
  long nnodes, maxnflowcycles, arclen, narcs, sourcetilenum, flowmax;
  long nnondecreasedcostiter;
  long *totarclens;
  long ***scndrycosts;
  double avgarclen;
  float **unwphase, **nextunwphase, **lastunwphase, **tempunwphase;
  float *unwphaseabove, *unwphasebelow;
  void **costs, **nextcosts, **lastcosts, **tempcosts;
  void *costsabove, *costsbelow;
  short **scndryflows, **bulkoffsets, **regions, **nextregions, **lastregions;
  short **tempregions, *regionsbelow, *regionsabove;
  int *nscndrynodes, *nscndryarcs;
  incrcostT **incrcosts;
  totalcostT totalcost, oldtotalcost, mintotalcost;
  nodeT *source;
  nodeT **scndrynodes, ***scndryapexes;
  signed char **iscandidate;
  signed char notfirstloop;
  candidateT *candidatebag, *candidatelist;
  nodesuppT **nodesupp;
  scndryarcT **scndryarcs;
  bucketT *bkts;
  char filename[MAXSTRLEN];

  
  /* set up */
  fprintf(sp1,"Assembling tiles\n"); 
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  ntiles=ntilerow*ntilecol;
  rowovrlp=params->rowovrlp;
  colovrlp=params->colovrlp;
  ni=ceil((nlines+(ntilerow-1)*rowovrlp)/(double )ntilerow);
  nj=ceil((linelen+(ntilecol-1)*colovrlp)/(double )ntilecol);
  nrow=0;
  ncol=0;
  flowmax=params->scndryarcflowmax;
  prevnrow=0;

  /* we are reading statistical costs from file even if (params->p >= 0) */
  /* need to set size of cost type and CalcCost function pointer to */
  /*   be consistent with data stored in file written by BuildCostArrays() */
  if(params->costmode==TOPO){
    costtypesize=sizeof(costT);
    CalcCost=CalcCostTopo;
    EvalCost=EvalCostTopo;
  }else if(params->costmode==DEFO){
    costtypesize=sizeof(costT);
    CalcCost=CalcCostDefo;
    EvalCost=EvalCostDefo;
  }else if(params->costmode==SMOOTH){
    costtypesize=sizeof(smoothcostT);
    CalcCost=CalcCostSmooth;
    EvalCost=EvalCostSmooth;
  }else{
    fflush(NULL);
    fprintf(sp0,"Illegal cost mode in AssembleTiles().  Abort\n");
    exit(ABNORMAL_EXIT);
  }
  /*
  if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
    costtypesize=sizeof(costT);
  }else if(CalcCost==CalcCostSmooth){
    costtypesize=sizeof(smoothcostT);
  }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
           || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
    costtypesize=sizeof(short);
  }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
           || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
    costtypesize=sizeof(bidircostT);
  }
  */

  /* get memory */
  regions=(short **)Get2DMem(ni,nj,sizeof(short *),sizeof(short));
  nextregions=(short **)Get2DMem(ni,nj,sizeof(short *),sizeof(short));
  lastregions=(short **)Get2DMem(ni,nj,sizeof(short *),sizeof(short));
  regionsbelow=(short *)MAlloc(nj*sizeof(short));
  regionsabove=(short *)MAlloc(nj*sizeof(short));
  unwphase=(float **)Get2DMem(ni,nj,sizeof(float *),sizeof(float));
  nextunwphase=(float **)Get2DMem(ni,nj,sizeof(float *),sizeof(float));
  lastunwphase=(float **)Get2DMem(ni,nj,sizeof(float *),sizeof(float));
  unwphaseabove=(float *)MAlloc(nj*sizeof(float));
  unwphasebelow=(float *)MAlloc(nj*sizeof(float));
  scndrynodes=(nodeT **)MAlloc(ntiles*sizeof(nodeT *));
  nodesupp=(nodesuppT **)MAlloc(ntiles*sizeof(nodesuppT *));
  scndryarcs=(scndryarcT **)MAlloc(ntiles*sizeof(scndryarcT *));
  scndrycosts=(long ***)MAlloc(ntiles*sizeof(long **));
  nscndrynodes=(int *)MAlloc(ntiles*sizeof(int));
  nscndryarcs=(int *)MAlloc(ntiles*sizeof(int));
  totarclens=(long *)MAlloc(ntiles*sizeof(long));
  bulkoffsets=(short **)Get2DMem(ntilerow,ntilecol,sizeof(short *),
                                 sizeof(short));
  costs=(void **)Get2DRowColMem(ni+2,nj+2,sizeof(void *),costtypesize);
  nextcosts=(void **)Get2DRowColMem(ni+2,nj+2,sizeof(void *),costtypesize);
  lastcosts=(void **)Get2DRowColMem(ni+2,nj+2,sizeof(void *),costtypesize);
  costsabove=(void *)MAlloc(nj*costtypesize);
  costsbelow=(void *)MAlloc(nj*costtypesize);


  /* trace regions and parse secondary nodes and arcs for each tile */
  bulkoffsets[0][0]=0;
  for(tilerow=0;tilerow<ntilerow;tilerow++){
    for(tilecol=0;tilecol<ntilecol;tilecol++){

      /* read region, unwrapped phase, and flow data */
      if(tilecol==0){
        ReadNextRegion(tilerow,0,nlines,linelen,outfiles,params,
                       &nextregions,&nextunwphase,&nextcosts,
                       &nextnrow,&nextncol);
        prevnrow=nrow;
        nrow=nextnrow;
      }
      prevncol=ncol;
      ncol=nextncol;
      tempregions=lastregions;
      lastregions=regions;
      regions=nextregions;
      nextregions=tempregions;
      tempunwphase=lastunwphase;
      lastunwphase=unwphase;
      unwphase=nextunwphase;
      nextunwphase=tempunwphase;
      tempcosts=lastcosts;
      lastcosts=costs;
      costs=nextcosts;
      nextcosts=tempcosts;
      if(tilecol!=ntilecol-1){
        ReadNextRegion(tilerow,tilecol+1,nlines,linelen,outfiles,params,
                       &nextregions,&nextunwphase,&nextcosts,
                       &nextnrow,&nextncol);
      }
      ReadEdgesAboveAndBelow(tilerow,tilecol,nlines,linelen,params,
                             outfiles,regionsabove,regionsbelow,
                             unwphaseabove,unwphasebelow,
                             costsabove,costsbelow);

      /* trace region edges to form nodes and arcs */
      TraceRegions(regions,nextregions,lastregions,regionsabove,regionsbelow,
                   unwphase,nextunwphase,lastunwphase,unwphaseabove,
                   unwphasebelow,costs,nextcosts,lastcosts,costsabove,
                   costsbelow,prevnrow,prevncol,tilerow,tilecol,
                   nrow,ncol,scndrynodes,nodesupp,scndryarcs,
                   scndrycosts,nscndrynodes,nscndryarcs,totarclens,
                   bulkoffsets,params);

    }
  }

  /* scale costs based on average number of primary arcs per secondary arc */
  arclen=0;
  narcs=0;
  for(i=0;i<ntiles;i++){
    arclen+=totarclens[i];
    narcs+=nscndryarcs[i];
  }
  avgarclen=arclen/narcs;

  /* may need to adjust scaling so fewer costs clipped */
  for(i=0;i<ntiles;i++){
    for(j=0;j<nscndryarcs[i];j++){
      if(scndrycosts[i][j][2*flowmax+1]!=ZEROCOSTARC){
        for(k=1;k<=2*flowmax;k++){
          scndrycosts[i][j][k]=(long )ceil(scndrycosts[i][j][k]/avgarclen);
        }
        scndrycosts[i][j][2*flowmax+1]=LRound(scndrycosts[i][j][2*flowmax+1]
                                              /avgarclen);
        if(scndrycosts[i][j][2*flowmax+1]<0){
          scndrycosts[i][j][2*flowmax+1]=0;
        }
      }
    }
  }

  /* free some intermediate memory */
  Free2DArray((void **)regions,ni);
  Free2DArray((void **)nextregions,ni);
  Free2DArray((void **)lastregions,ni);
  Free2DArray((void **)unwphase,ni);
  Free2DArray((void **)nextunwphase,ni);
  Free2DArray((void **)lastunwphase,ni);
  Free2DArray((void **)costs,ni);
  Free2DArray((void **)nextcosts,ni);
  Free2DArray((void **)lastcosts,ni);
  free(costsabove);
  free(costsbelow);
  free(unwphaseabove);
  free(unwphasebelow);
  free(regionsabove);
  free(regionsbelow);


  /* get memory for nongrid arrays of secondary network problem */
  scndryflows=(short **)MAlloc(ntiles*sizeof(short *));
  iscandidate=(signed char **)MAlloc(ntiles*sizeof(signed char*));
  scndryapexes=(nodeT ***)MAlloc(ntiles*sizeof(nodeT **));
  incrcosts=(incrcostT **)MAlloc(ntiles*sizeof(incrcostT *));
  nnodes=0;
  for(i=0;i<ntiles;i++){
    scndryflows[i]=(short *)CAlloc(nscndryarcs[i],sizeof(short));
    iscandidate[i]=(signed char *)MAlloc(nscndryarcs[i]*sizeof(signed char));
    scndryapexes[i]=(nodeT **)MAlloc(nscndryarcs[i]*sizeof(nodeT *));
    incrcosts[i]=(incrcostT *)MAlloc(nscndryarcs[i]*sizeof(incrcostT));
    nnodes+=nscndrynodes[i];
  }
  
  /* set up network for secondary solver */
  InitNetwork(scndryflows,&dummylong,&ncycle,&nflowdone,&dummylong,&nflow,
              &candidatebagsize,&candidatebag,&candidatelistsize,
              &candidatelist,NULL,NULL,&bkts,&dummylong,NULL,NULL,NULL,
              NULL,NULL,NULL,NULL,ntiles,0,&notfirstloop,&totalcost,params);
  oldtotalcost=totalcost;
  mintotalcost=totalcost;
  nnondecreasedcostiter=0;

  /* set pointers to functions for nongrid secondary network */
  CalcCost=CalcCostNonGrid;
  EvalCost=EvalCostNonGrid;
  SetNonGridNetworkFunctionPointers();

  /* solve the secondary network problem */
  /* main loop: loop over flow increments and sources */
  fprintf(sp1,"Running optimizer for secondary network\n");
  fprintf(sp1,"Number of nodes in secondary network: %ld\n",nnodes);
  maxnflowcycles=LRound(nnodes*params->maxcyclefraction);
  while(TRUE){ 
 
    fprintf(sp1,"Flow increment: %ld  (Total improvements: %ld)\n",
            nflow,ncycle);

    /* set up the incremental (residual) cost arrays */
    SetupIncrFlowCosts((void **)scndrycosts,incrcosts,scndryflows,nflow,ntiles,
                       ntiles,nscndryarcs,params); 

    /* set the tree root (equivalent to source of shortest path problem) */
    sourcetilenum=(long )ntilecol*floor(ntilerow/2.0)+floor(ntilecol/2.0);
    source=&scndrynodes[sourcetilenum][0];

    /* set up network variables for tree solver */
    SetupTreeSolveNetwork(scndrynodes,NULL,scndryapexes,iscandidate,
                          ntiles,nscndrynodes,ntiles,nscndryarcs,
                          ntiles,0);
      
    /* run the solver, and increment nflowdone if no cycles are found */
    n=TreeSolve(scndrynodes,nodesupp,NULL,source,&candidatelist,&candidatebag,
                &candidatelistsize,&candidatebagsize,bkts,scndryflows,
                (void **)scndrycosts,incrcosts,scndryapexes,iscandidate,0,
                nflow,NULL,NULL,NULL,ntiles,nscndrynodes,ntiles,nscndryarcs,
                ntiles,0,NULL,nnodes,params);
    
    /* evaluate and save the total cost (skip if first loop through nflow) */
    if(notfirstloop){
      oldtotalcost=totalcost;
      totalcost=EvaluateTotalCost((void **)scndrycosts,scndryflows,ntiles,0,
                                  nscndryarcs,params);
      if(totalcost<mintotalcost){
        mintotalcost=totalcost;
      }
      if(totalcost>oldtotalcost || (n>0 && totalcost==oldtotalcost)){
        fflush(NULL);
        fprintf(sp1,"Caution: Unexpected increase in total cost\n");
      }
      if(totalcost>=mintotalcost){
        nnondecreasedcostiter++;
      }else{
        nnondecreasedcostiter=0;
      }
    }

    /* consider this flow increment done if not too many neg cycles found */
    ncycle+=n;
    if(n<=maxnflowcycles){
      nflowdone++;
    }else{
      nflowdone=1;
    }

    /* break if lack of total cost reduction is sustained */
    if(nnondecreasedcostiter>=2*params->maxflow){
      fflush(NULL);
      fprintf(sp0,"WARNING: No overall cost reduction for too many iterations."
              "  Breaking loop\n");
      break;
    }


    /* break if we're done with all flow increments or problem is convex */
    if(nflowdone>=params->maxflow){
      break;
    }

    /* update flow increment */
    nflow++;
    if(nflow>params->maxflow){
      nflow=1;
      notfirstloop=TRUE;
    }

  } /* end loop until no more neg cycles */

  /* free some memory */
  for(i=0;i<ntiles;i++){
    for(j=0;j<nscndryarcs[i];j++){
      free(scndrycosts[i][j]);
    }
  }
  Free2DArray((void **)scndrycosts,ntiles);
  Free2DArray((void **)scndryapexes,ntiles);
  Free2DArray((void **)iscandidate,ntiles);
  Free2DArray((void **)incrcosts,ntiles);
  free(candidatebag);
  free(candidatelist);  
  free(bkts->bucketbase);

  /* assemble connected component files if needed */
  if(strlen(outfiles->conncompfile)){
    AssembleTileConnComps(linelen,nlines,outfiles,params);
  }
  
  /* integrate phase from secondary network problem */
  IntegrateSecondaryFlows(linelen,nlines,scndrynodes,nodesupp,scndryarcs,
                          nscndryarcs,scndryflows,bulkoffsets,outfiles,params);

  /* free remaining memory */
  for(i=0;i<ntiles;i++){
    for(j=0;j<nscndrynodes[i];j++){
      free(nodesupp[i][j].neighbornodes);
      free(nodesupp[i][j].outarcs);
    }
  }
  Free2DArray((void **)nodesupp,ntiles);
  Free2DArray((void **)scndrynodes,ntiles);
  Free2DArray((void **)scndryarcs,ntiles);
  Free2DArray((void **)scndryflows,ntiles);
  free(nscndrynodes);
  free(nscndryarcs);
  Free2DArray((void **)bulkoffsets,ntilerow);

  /* remove temporary tile log files and tile directory */
  if(params->rmtmptile){
    fflush(NULL);
    fprintf(sp1,"Removing temporary directory %s\n",params->tiledir);
    for(tilerow=0;tilerow<ntilerow;tilerow++){
      for(tilecol=0;tilecol<ntilecol;tilecol++){
        sprintf(filename,"%s/%s%ld_%ld",
                params->tiledir,LOGFILEROOT,tilerow,tilecol);
        unlink(filename);
      }
    }
    rmdir(params->tiledir);
  }

  /* Give notice about increasing overlap if there are edge artifacts */
  if(params->rowovrlp<ni || params->colovrlp<nj){
    fflush(NULL);
    fprintf(sp1,
            "SUGGESTION: Try increasing tile overlap and/or size"
            " if solution has edge artifacts\n");
  }
  
  /* done */
  return(0);
}


/* function: ReadNextRegion()
 * --------------------------
 */
static
int ReadNextRegion(long tilerow, long tilecol, long nlines, long linelen,
                   outfileT *outfiles, paramT *params, 
                   short ***nextregionsptr, float ***nextunwphaseptr,
                   void ***nextcostsptr, 
                   long *nextnrowptr, long *nextncolptr){
  
  long nexttilelinelen, nexttilenlines, costtypesize;
  tileparamT nexttileparams[1];
  outfileT nexttileoutfiles[1];
  char nextfile[MAXSTRLEN], tempstring[MAXTMPSTRLEN];
  char path[MAXSTRLEN], basename[MAXSTRLEN];
  

  /* initialize stack structures and strings to zero for good measure */
  memset(nexttileparams,0,sizeof(tileparamT));
  memset(nexttileoutfiles,0,sizeof(outfileT));
  memset(nextfile,0,MAXSTRLEN);
  memset(tempstring,0,MAXSTRLEN);  
  memset(path,0,MAXSTRLEN);  
  memset(basename,0,MAXSTRLEN);  
  
  /* size of the data type for holding cost data depends on cost mode */
  if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
    costtypesize=sizeof(costT);
  }else if(CalcCost==CalcCostSmooth){
    costtypesize=sizeof(smoothcostT);
  }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
           || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
    costtypesize=sizeof(short);
  }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
           || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
    costtypesize=sizeof(bidircostT);
  }else{
    fflush(NULL);
    fprintf(sp0,"ERROR: Bad CalcCost func ptr in ReadNextRegion()\n");
    exit(ABNORMAL_EXIT);
  }

  /* use SetupTile() to set filenames only; tile params overwritten below */
  SetupTile(nlines,linelen,params,nexttileparams,outfiles,nexttileoutfiles,
            tilerow,tilecol);
  nexttilenlines=nexttileparams->nrow;
  nexttilelinelen=nexttileparams->ncol;

  /* set tile parameters, overwriting values set by SetupTile() above */
  SetTileReadParams(nexttileparams,nexttilenlines,nexttilelinelen,
                    tilerow,tilecol,nlines,linelen,params);

  /* read region data */
  ParseFilename(outfiles->outfile,path,basename);
  sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld%s",
          params->tiledir,TMPTILEROOT,basename,tilerow,tilecol,
          nexttilelinelen,REGIONSUFFIX);
  StrNCopy(nextfile,tempstring,MAXSTRLEN);
  Read2DArray((void ***)nextregionsptr,nextfile,
              nexttilelinelen,nexttilenlines,
              nexttileparams,sizeof(short *),sizeof(short));

  /* read unwrapped phase data */
  if(TMPTILEOUTFORMAT==ALT_LINE_DATA){
    ReadAltLineFilePhase(nextunwphaseptr,nexttileoutfiles->outfile,
                         nexttilelinelen,nexttilenlines,nexttileparams);
  }else if(TMPTILEOUTFORMAT==FLOAT_DATA){
    Read2DArray((void ***)nextunwphaseptr,nexttileoutfiles->outfile,
                nexttilelinelen,nexttilenlines,nexttileparams,
                sizeof(float *),sizeof(float));
  }else{
    fflush(NULL);
    fprintf(sp0,"Cannot read format of unwrapped phase tile data\nAbort\n");
    exit(ABNORMAL_EXIT);
  }

  /* read cost data */
  Read2DRowColFile((void ***)nextcostsptr,nexttileoutfiles->costoutfile,
                   nexttilelinelen,nexttilenlines,nexttileparams,
                   costtypesize);

  /* flip sign of wrapped phase if flip flag is set */
  FlipPhaseArraySign(*nextunwphaseptr,params,
                     nexttileparams->nrow,nexttileparams->ncol);

  /* set outputs */
  (*nextnrowptr)=nexttileparams->nrow;
  (*nextncolptr)=nexttileparams->ncol;
  return(0);

}

/* function: SetTileReadParams()
 * -----------------------------
 * Set parameters for reading the nonoverlapping piece of each tile.  
 * ni and nj are the numbers of rows and columns in this particular tile.
 * The meanings of these variables are different for the last row 
 * and column.
 */
static
int SetTileReadParams(tileparamT *tileparams, long nexttilenlines, 
                      long nexttilelinelen, long tilerow, long tilecol, 
                      long nlines, long linelen, paramT *params){

  long rowovrlp, colovrlp;

  /* set temporary variables */
  rowovrlp=params->rowovrlp;
  colovrlp=params->colovrlp;

  /* row parameters */
  if(tilerow==0){
    tileparams->firstrow=0;
  }else{
    tileparams->firstrow=ceil(rowovrlp/2.0);
  }
  if(tilerow!=params->ntilerow-1){
    tileparams->nrow=nexttilenlines-floor(rowovrlp/2.0)-tileparams->firstrow;
  }else{
    tileparams->nrow=nexttilenlines-tileparams->firstrow;
  }

  /* column parameters */
  if(tilecol==0){
    tileparams->firstcol=0;
  }else{
    tileparams->firstcol=ceil(colovrlp/2.0);
  }
  if(tilecol!=params->ntilecol-1){
    tileparams->ncol=nexttilelinelen-floor(colovrlp/2.0)-tileparams->firstcol;
  }else{
    tileparams->ncol=nexttilelinelen-tileparams->firstcol;
  }
  return(0);
}


/* function: ReadEdgesAboveAndBelow()
 * ----------------------------------
 */
static
int ReadEdgesAboveAndBelow(long tilerow, long tilecol, long nlines, 
                           long linelen, paramT *params, outfileT *outfiles, 
                           short *regionsabove, short *regionsbelow,
                           float *unwphaseabove, float *unwphasebelow,
                           void *costsabove, void *costsbelow){

  long ni, nj, readtilelinelen, readtilenlines, costtypesize;
  long ntilerow, ntilecol, rowovrlp, colovrlp;
  tileparamT tileparams[1];
  outfileT outfilesabove[1], outfilesbelow[1];
  float **unwphaseaboveptr, **unwphasebelowptr;
  void **costsaboveptr, **costsbelowptr;
  short **regionsaboveptr, **regionsbelowptr;
  char tempstring[MAXTMPSTRLEN], readregionfile[MAXSTRLEN];
  char path[MAXSTRLEN], basename[MAXSTRLEN];


  /* initialize stack structures to zero for good measure */
  memset(tileparams,0,sizeof(tileparamT));
  memset(outfilesabove,0,sizeof(outfileT));
  memset(outfilesbelow,0,sizeof(outfileT));
  memset(tempstring,0,MAXSTRLEN);
  memset(readregionfile,0,MAXSTRLEN);
  memset(path,0,MAXSTRLEN);
  memset(basename,0,MAXSTRLEN);
  
  /* set temporary variables */
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  rowovrlp=params->rowovrlp;
  colovrlp=params->colovrlp;
  ni=ceil((nlines+(ntilerow-1)*rowovrlp)/(double )ntilerow);
  nj=ceil((linelen+(ntilecol-1)*colovrlp)/(double )ntilecol);

  /* size of the data type for holding cost data depends on cost mode */
  if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
    costtypesize=sizeof(costT);
  }else if(CalcCost==CalcCostSmooth){
    costtypesize=sizeof(smoothcostT);
  }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
           || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
    costtypesize=sizeof(short);
  }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
           || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
    costtypesize=sizeof(bidircostT);
  }else{
    fflush(NULL);
    fprintf(sp0,"ERROR: Bad CalcCost func ptr in ReadEdgesAboveAndBelow()\n");
    exit(ABNORMAL_EXIT);
  }

  /* set names of files with SetupTile() */
  /* tile parameters set by SetupTile() will be overwritten below */
  if(tilerow!=0){
    SetupTile(nlines,linelen,params,tileparams,outfiles,outfilesabove,
              tilerow-1,tilecol);
  }
  if(tilerow!=ntilerow-1){
    SetupTile(nlines,linelen,params,tileparams,outfiles,outfilesbelow,
              tilerow+1,tilecol);
  }

  /* temporary pointers, so we can use Read2DArray() with 1D output array */
  unwphaseaboveptr=&unwphaseabove;
  unwphasebelowptr=&unwphasebelow;
  costsaboveptr=&costsabove;
  costsbelowptr=&costsbelow;
  regionsaboveptr=&regionsabove;
  regionsbelowptr=&regionsbelow;
  
  /* set some reading parameters */
  if(tilecol==0){
    tileparams->firstcol=0;
  }else{
    tileparams->firstcol=ceil(colovrlp/2.0);
  }
  if(tilecol!=params->ntilecol-1){
    readtilelinelen=nj;
    tileparams->ncol=readtilelinelen-floor(colovrlp/2.0)-tileparams->firstcol;
  }else{
    readtilelinelen=linelen-(ntilecol-1)*(nj-colovrlp);
    tileparams->ncol=readtilelinelen-tileparams->firstcol;
  }
  tileparams->nrow=1;

  /* read last line of tile above */
  readtilenlines=ni;
  if(tilerow!=0){
    tileparams->firstrow=readtilenlines-floor(rowovrlp/2.0)-1;

    /* read region data */
    ParseFilename(outfiles->outfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld%s",
            params->tiledir,TMPTILEROOT,basename,tilerow-1,tilecol,
            readtilelinelen,REGIONSUFFIX);
    StrNCopy(readregionfile,tempstring,MAXSTRLEN);
    Read2DArray((void ***)&regionsaboveptr,readregionfile,
                readtilelinelen,readtilenlines,
                tileparams,sizeof(short *),sizeof(short));

    /* read unwrapped phase data */
    if(TMPTILEOUTFORMAT==ALT_LINE_DATA){
      ReadAltLineFilePhase(&unwphaseaboveptr,outfilesabove->outfile,
                           readtilelinelen,readtilenlines,tileparams);
    }else if(TMPTILEOUTFORMAT==FLOAT_DATA){
      Read2DArray((void ***)&unwphaseaboveptr,outfilesabove->outfile,
                  readtilelinelen,readtilenlines,tileparams,
                  sizeof(float *),sizeof(float));
    }

    /* flip sign of wrapped phase if flip flag is set */
    FlipPhaseArraySign(unwphaseaboveptr,params,
                       tileparams->nrow,tileparams->ncol);

    /* read costs data */
    tileparams->firstrow--;
    Read2DRowColFileRows((void ***)&costsaboveptr,outfilesabove->costoutfile,
                         readtilelinelen,readtilenlines,tileparams,
                         costtypesize);

    /* remove temporary tile cost file unless told to save it */
    if(params->rmtmptile && !strlen(outfiles->costoutfile)){
      unlink(outfilesabove->costoutfile);
    }
  }

  /* read first line of tile below */
  if(tilerow!=ntilerow-1){
    if(tilerow==params->ntilerow-2){
      readtilenlines=nlines-(ntilerow-1)*(ni-rowovrlp);
    }
    tileparams->firstrow=ceil(rowovrlp/2.0);

    /* read region data */
    ParseFilename(outfiles->outfile,path,basename);
    sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld%s",
            params->tiledir,TMPTILEROOT,basename,tilerow+1,tilecol,
            readtilelinelen,REGIONSUFFIX);
    StrNCopy(readregionfile,tempstring,MAXSTRLEN);
    Read2DArray((void ***)&regionsbelowptr,readregionfile,
                readtilelinelen,readtilenlines,
                tileparams,sizeof(short *),sizeof(short));

    /* read unwrapped phase data */
    if(TMPTILEOUTFORMAT==ALT_LINE_DATA){
      ReadAltLineFilePhase(&unwphasebelowptr,outfilesbelow->outfile,
                           readtilelinelen,readtilenlines,tileparams);
    }else if(TMPTILEOUTFORMAT==FLOAT_DATA){
      Read2DArray((void ***)&unwphasebelowptr,outfilesbelow->outfile,
                  readtilelinelen,readtilenlines,tileparams,
                  sizeof(float *),sizeof(float));
    }

    /* flip the sign of the wrapped phase if flip flag is set */
    FlipPhaseArraySign(unwphasebelowptr,params,
                       tileparams->nrow,tileparams->ncol);

    /* read costs data */
    Read2DRowColFileRows((void ***)&costsbelowptr,outfilesbelow->costoutfile,
                         readtilelinelen,readtilenlines,tileparams,
                         costtypesize);

  }else{

    /* remove temporoary tile cost file for last row unless told to save it */
    if(params->rmtmptile && !strlen(outfiles->costoutfile)){
      SetupTile(nlines,linelen,params,tileparams,outfiles,outfilesbelow,
                tilerow,tilecol);
      unlink(outfilesbelow->costoutfile);
    }
  }

  /* done */
  return(0);
  
}


/* function: TraceRegions()
 * ------------------------
 * Trace edges of region data to form nodes and arcs of secondary
 * (ie, region-level) network problem.  Primary nodes and arcs are
 * those of the original, pixel-level network problem.  Flows along
 * edges are computed knowing the unwrapped phase values of edges
 * of adjacent tiles.  Costs along edges are approximated in that they
 * are calculated from combining adjacent cost parameters, not from 
 * using the exact method in BuildCostArrays().
 */
static
int TraceRegions(short **regions, short **nextregions, short **lastregions, 
                 short *regionsabove, short *regionsbelow, float **unwphase, 
                 float **nextunwphase, float **lastunwphase, 
                 float *unwphaseabove, float *unwphasebelow, void **costs, 
                 void **nextcosts, void **lastcosts, void *costsabove, 
                 void *costsbelow, long prevnrow, long prevncol, long tilerow,
                 long tilecol, long nrow, long ncol, nodeT **scndrynodes,
                 nodesuppT **nodesupp, scndryarcT **scndryarcs, 
                 long ***scndrycosts, int *nscndrynodes, 
                 int *nscndryarcs, long *totarclens, short **bulkoffsets, 
                 paramT *params){

  long i, j, row, col, nnrow, nncol, tilenum, costtypesize;
  long nnewnodes, nnewarcs, npathsout, flowmax, totarclen;
  long nupdatednontilenodes, updatednontilenodesize, ntilecol;
  short **flows;
  short **rightedgeflows, **loweredgeflows, **leftedgeflows, **upperedgeflows;
  short *inontilenodeoutarc;
  void **rightedgecosts, **loweredgecosts, **leftedgecosts, **upperedgecosts;
  nodeT **primarynodes, **updatednontilenodes;
  nodeT *from, *to, *nextnode, *tempnode;
  nodesuppT *fromsupp, *tosupp;


  /* initialize */
  ntilecol=params->ntilecol;
  nnrow=nrow+1;
  nncol=ncol+1;
  primarynodes=(nodeT **)Get2DMem(nnrow,nncol,sizeof(nodeT *),sizeof(nodeT));
  for(row=0;row<nnrow;row++){
    for(col=0;col<nncol;col++){
      primarynodes[row][col].row=row;
      primarynodes[row][col].col=col;
      primarynodes[row][col].group=NOTINBUCKET;
      primarynodes[row][col].pred=NULL;
      primarynodes[row][col].next=NULL;
    }
  }
  nextnode=&primarynodes[0][0];
  tilenum=tilerow*ntilecol+tilecol;
  scndrynodes[tilenum]=NULL;
  nodesupp[tilenum]=NULL;
  scndryarcs[tilenum]=NULL;
  scndrycosts[tilenum]=NULL;
  nnewnodes=0;
  nnewarcs=0;
  totarclen=0;
  flowmax=params->scndryarcflowmax;
  updatednontilenodesize=INITARRSIZE;
  nupdatednontilenodes=0;

  /* size of the data type for holding cost data depends on cost mode */
  if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
    costtypesize=sizeof(costT);
  }else if(CalcCost==CalcCostSmooth){
    costtypesize=sizeof(smoothcostT);
  }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
           || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
    costtypesize=sizeof(short);
  }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
           || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
    costtypesize=sizeof(bidircostT);
  }else{
    fflush(NULL);
    fprintf(sp0,"ERROR: Bad CalcCost func ptr in TraceRegions()\n");
    exit(ABNORMAL_EXIT);
  }

  /* get memory */
  updatednontilenodes=(nodeT **)MAlloc(updatednontilenodesize*sizeof(nodeT *));
  inontilenodeoutarc=(short *)MAlloc(updatednontilenodesize*sizeof(short));
  flows=(short **)Get2DRowColMem(nrow+1,ncol+1,sizeof(short *),sizeof(short));
  rightedgeflows=(short **)Get2DMem(nrow,1,sizeof(short *),sizeof(short));
  leftedgeflows=(short **)Get2DMem(nrow,1,sizeof(short *),sizeof(short));
  upperedgeflows=(short **)Get2DMem(1,ncol,sizeof(short *),sizeof(short));
  loweredgeflows=(short **)Get2DMem(1,ncol,sizeof(short *),sizeof(short));
  rightedgecosts=(void **)Get2DMem(nrow,1,sizeof(void *),costtypesize);
  leftedgecosts=(void **)Get2DMem(nrow,1,sizeof(void *),costtypesize);
  upperedgecosts=(void **)Get2DMem(1,ncol,sizeof(void *),costtypesize);
  loweredgecosts=(void **)Get2DMem(1,ncol,sizeof(void *),costtypesize);

  /* parse flows for this tile */
  CalcFlow(unwphase,&flows,nrow,ncol);

  /* set up cost and flow arrays for boundaries */
  SetUpperEdge(ncol,tilerow,tilecol,costs,costsabove,unwphase,unwphaseabove,
               upperedgecosts,upperedgeflows,params, bulkoffsets);
  SetLowerEdge(nrow,ncol,tilerow,tilecol,costs,costsbelow,unwphase,
               unwphasebelow,loweredgecosts,loweredgeflows,
               params,bulkoffsets);
  SetLeftEdge(nrow,prevncol,tilerow,tilecol,costs,lastcosts,unwphase,
              lastunwphase,leftedgecosts,leftedgeflows,params, bulkoffsets);
  SetRightEdge(nrow,ncol,tilerow,tilecol,costs,nextcosts,unwphase, 
               nextunwphase,rightedgecosts,rightedgeflows,
               params,bulkoffsets);

  /* trace edges between regions */
  while(nextnode!=NULL){
 
    /* get next primary node from stack */
    from=nextnode;
    nextnode=nextnode->next;
    from->group=NOTINBUCKET;

    /* find number of paths out of from node */
    npathsout=FindNumPathsOut(from,params,tilerow,tilecol,nnrow,nncol,regions,
                              nextregions,lastregions,regionsabove,
                              regionsbelow,prevncol);

    /* secondary node exists if region edges fork */
    if(npathsout>2){

      /* mark primary node to indicate that secondary node exists for it */
      from->group=ONTREE;
      
      /* create secondary node if not already created in another tile */
      if((from->row!=0 || tilerow==0) && (from->col!=0 || tilecol==0)){

        /* create the secondary node */
        nnewnodes++;
        if(nnewnodes > SHRT_MAX){
          fflush(NULL);
          fprintf(sp0,"Exceeded maximum number of secondary nodes\n"
                  "Decrease TILECOSTTHRESH and/or increase MINREGIONSIZE\n");
          exit(ABNORMAL_EXIT);
        }
        scndrynodes[tilenum]=(nodeT *)ReAlloc(scndrynodes[tilenum],
                                              nnewnodes*sizeof(nodeT));
        nodesupp[tilenum]=(nodesuppT *)ReAlloc(nodesupp[tilenum],
                                               nnewnodes*sizeof(nodesuppT));
        scndrynodes[tilenum][nnewnodes-1].row=tilenum;
        scndrynodes[tilenum][nnewnodes-1].col=nnewnodes-1;
        nodesupp[tilenum][nnewnodes-1].row=from->row;
        nodesupp[tilenum][nnewnodes-1].col=from->col;
        nodesupp[tilenum][nnewnodes-1].noutarcs=0;
        nodesupp[tilenum][nnewnodes-1].neighbornodes=NULL;
        nodesupp[tilenum][nnewnodes-1].outarcs=NULL;
      }      

      /* create the secondary arc to this node if it doesn't already exist */
      if(from->pred!=NULL
         && ((from->row==from->pred->row && (from->row!=0 || tilerow==0))
             || (from->col==from->pred->col && (from->col!=0 || tilecol==0)))){

        TraceSecondaryArc(from,scndrynodes,nodesupp,scndryarcs,scndrycosts,
                          &nnewnodes,&nnewarcs,tilerow,tilecol,flowmax,
                          nrow,ncol,prevnrow,prevncol,params,costs,
                          rightedgecosts,loweredgecosts,leftedgecosts,
                          upperedgecosts,flows,rightedgeflows,loweredgeflows, 
                          leftedgeflows,upperedgeflows,&updatednontilenodes,
                          &nupdatednontilenodes,&updatednontilenodesize,
                          &inontilenodeoutarc,&totarclen);
      }
    }

    /* scan neighboring primary nodes and place path candidates into stack */
    RegionTraceCheckNeighbors(from,&nextnode,primarynodes,regions,
                              nextregions,lastregions,regionsabove,
                              regionsbelow,tilerow,tilecol,nnrow,nncol,
                              scndrynodes,nodesupp,scndryarcs,&nnewnodes,
                              &nnewarcs,flowmax,nrow,ncol,prevnrow,prevncol,
                              params,costs,rightedgecosts,loweredgecosts,
                              leftedgecosts,upperedgecosts,flows,
                              rightedgeflows,loweredgeflows,leftedgeflows,
                              upperedgeflows,scndrycosts,&updatednontilenodes,
                              &nupdatednontilenodes,&updatednontilenodesize,
                              &inontilenodeoutarc,&totarclen);
  }


  /* reset temporary secondary node and arc pointers in data structures */
  /* secondary node row, col stored level, incost of primary node pointed to */

  /* update nodes in this tile */
  for(i=0;i<nnewnodes;i++){
    for(j=0;j<nodesupp[tilenum][i].noutarcs;j++){
      tempnode=nodesupp[tilenum][i].neighbornodes[j];
      nodesupp[tilenum][i].neighbornodes[j]
        =&scndrynodes[tempnode->level][tempnode->incost];
    }
  }

  /* update nodes not in this tile that were affected (that have new arcs) */
  for(i=0;i<nupdatednontilenodes;i++){
    row=updatednontilenodes[i]->row;
    col=updatednontilenodes[i]->col;
    j=inontilenodeoutarc[i];
    tempnode=nodesupp[row][col].neighbornodes[j];
    nodesupp[row][col].neighbornodes[j]
      =&scndrynodes[tempnode->level][tempnode->incost];
  }

  /* update secondary arcs */
  for(i=0;i<nnewarcs;i++){

    /* update node pointers in secondary arc structure */
    tempnode=scndryarcs[tilenum][i].from;
    scndryarcs[tilenum][i].from
      =&scndrynodes[tempnode->level][tempnode->incost];
    from=scndryarcs[tilenum][i].from;
    tempnode=scndryarcs[tilenum][i].to;
    scndryarcs[tilenum][i].to
      =&scndrynodes[tempnode->level][tempnode->incost];
    to=scndryarcs[tilenum][i].to;

    /* update secondary arc pointers in nodesupp strcutres */
    fromsupp=&nodesupp[from->row][from->col];
    j=0;
    while(fromsupp->neighbornodes[j]!=to){
      j++;
    }
    fromsupp->outarcs[j]=&scndryarcs[tilenum][i];
    tosupp=&nodesupp[to->row][to->col];
    j=0;
    while(tosupp->neighbornodes[j]!=from){
      j++;
    }
    tosupp->outarcs[j]=&scndryarcs[tilenum][i];
  }

  /* set outputs */
  nscndrynodes[tilenum]=nnewnodes;
  nscndryarcs[tilenum]=nnewarcs;
  totarclens[tilenum]=totarclen;

  /* free memory */
  Free2DArray((void **)primarynodes,nnrow);
  Free2DArray((void **)flows,2*nrow-1);
  Free2DArray((void **)rightedgeflows,nrow);
  Free2DArray((void **)leftedgeflows,nrow);
  Free2DArray((void **)upperedgeflows,1);
  Free2DArray((void **)loweredgeflows,1);
  Free2DArray((void **)rightedgecosts,nrow);
  Free2DArray((void **)leftedgecosts,nrow);
  Free2DArray((void **)upperedgecosts,1);
  Free2DArray((void **)loweredgecosts,1);
  return(0);
}


/* function: FindNumPathsOut()
 * ---------------------------
 * Check all outgoing arcs to see how many paths out there are. 
 */
static
long FindNumPathsOut(nodeT *from, paramT *params, long tilerow, long tilecol, 
                     long nnrow, long nncol, short **regions, 
                     short **nextregions, short **lastregions,
                     short *regionsabove, short *regionsbelow, long prevncol){

  long npathsout, ntilerow, ntilecol, fromrow, fromcol;

  /* initialize */
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  fromrow=from->row;
  fromcol=from->col;
  npathsout=0;

  /* rightward arc */
  if(fromcol!=nncol-1){
    if(fromrow==0 || fromrow==nnrow-1 
       || regions[fromrow-1][fromcol]!=regions[fromrow][fromcol]){
      npathsout++;
    }
  }else{
    if(fromrow==0 || fromrow==nnrow-1 || 
       (tilecol!=ntilecol-1
        && nextregions[fromrow-1][0]!=nextregions[fromrow][0])){
      npathsout++;
    }
  }

  /* downward arc */
  if(fromrow!=nnrow-1){
    if(fromcol==0 || fromcol==nncol-1
       || regions[fromrow][fromcol]!=regions[fromrow][fromcol-1]){
      npathsout++;
    }
  }else{
    if(fromcol==0 || fromcol==nncol-1 ||
       (tilerow!=ntilerow-1 
        && regionsbelow[fromcol]!=regionsbelow[fromcol-1])){
      npathsout++;
    }
  }

  /* leftward arc */
  if(fromcol!=0){
    if(fromrow==0 || fromrow==nnrow-1 
       || regions[fromrow][fromcol-1]!=regions[fromrow-1][fromcol-1]){
      npathsout++;
    }
  }else{
    if(fromrow==0 || fromrow==nnrow-1 || 
       (tilecol!=0
        && (lastregions[fromrow][prevncol-1]
            !=lastregions[fromrow-1][prevncol-1]))){
      npathsout++;
    }
  }

  /* upward arc */
  if(fromrow!=0){
    if(fromcol==0 || fromcol==nncol-1
       || regions[fromrow-1][fromcol-1]!=regions[fromrow-1][fromcol]){
      npathsout++;
    }
  }else{
    if(fromcol==0 || fromcol==nncol-1 ||
       (tilerow!=0
        && regionsabove[fromcol-1]!=regionsabove[fromcol])){
      npathsout++;
    }
  }

  /* return number of paths out of node */
  return(npathsout);

}


/* function: RegionTraceCheckNeighbors()
 * -------------------------------------
 */
static
int RegionTraceCheckNeighbors(nodeT *from, nodeT **nextnodeptr, 
                              nodeT **primarynodes, short **regions, 
                              short **nextregions, short **lastregions, 
                              short *regionsabove, short *regionsbelow,  
                              long tilerow, long tilecol, long nnrow, 
                              long nncol, nodeT **scndrynodes, 
                              nodesuppT **nodesupp, scndryarcT **scndryarcs, 
                              long *nnewnodesptr, long *nnewarcsptr, 
                              long flowmax, long nrow, long ncol, 
                              long prevnrow, long prevncol, paramT *params, 
                              void **costs, void **rightedgecosts, 
                              void **loweredgecosts, void **leftedgecosts, 
                              void **upperedgecosts, short **flows, 
                              short **rightedgeflows, short **loweredgeflows,
                              short **leftedgeflows, short **upperedgeflows,
                              long ***scndrycosts, 
                              nodeT ***updatednontilenodesptr, 
                              long *nupdatednontilenodesptr, 
                              long *updatednontilenodesizeptr,
                              short **inontilenodeoutarcptr, 
                              long *totarclenptr){

  long fromrow, fromcol;
  nodeT *to, *nextnode;


  /* initialize */
  fromrow=from->row;
  fromcol=from->col;
  nextnode=(*nextnodeptr);


  /* check rightward arc */
  if(fromcol!=nncol-1){
    to=&primarynodes[fromrow][fromcol+1];
    if(fromrow==0 || fromrow==nnrow-1 
       || regions[fromrow-1][fromcol]!=regions[fromrow][fromcol]){
      if(to!=from->pred){
        to->pred=from;
        if(to->group==NOTINBUCKET){
          to->group=INBUCKET;
          to->next=nextnode;
          nextnode=to;
        }else if(to->group==ONTREE && (fromrow!=0 || tilerow==0)){
          TraceSecondaryArc(to,scndrynodes,nodesupp,scndryarcs,scndrycosts,
                            nnewnodesptr,nnewarcsptr,tilerow,tilecol,flowmax,
                            nrow,ncol,prevnrow,prevncol,params,costs,
                            rightedgecosts,loweredgecosts,leftedgecosts,
                            upperedgecosts,flows,rightedgeflows,
                            loweredgeflows,leftedgeflows,upperedgeflows,
                            updatednontilenodesptr,nupdatednontilenodesptr,
                            updatednontilenodesizeptr,inontilenodeoutarcptr,
                            totarclenptr);
        }
      }
    }
  }


  /* check downward arc */
  if(fromrow!=nnrow-1){
    to=&primarynodes[fromrow+1][fromcol];
    if(fromcol==0 || fromcol==nncol-1
       || regions[fromrow][fromcol]!=regions[fromrow][fromcol-1]){
      if(to!=from->pred){
        to->pred=from;
        if(to->group==NOTINBUCKET){
          to->group=INBUCKET;
          to->next=nextnode;
          nextnode=to;
        }else if(to->group==ONTREE && (fromcol!=0 || tilecol==0)){
          TraceSecondaryArc(to,scndrynodes,nodesupp,scndryarcs,scndrycosts,
                            nnewnodesptr,nnewarcsptr,tilerow,tilecol,flowmax,
                            nrow,ncol,prevnrow,prevncol,params,costs,
                            rightedgecosts,loweredgecosts,leftedgecosts,
                            upperedgecosts,flows,rightedgeflows,
                            loweredgeflows,leftedgeflows,upperedgeflows,
                            updatednontilenodesptr,nupdatednontilenodesptr,
                            updatednontilenodesizeptr,inontilenodeoutarcptr,
                            totarclenptr);
        }
      }
    }
  }

      
  /* check leftward arc */
  if(fromcol!=0){
    to=&primarynodes[fromrow][fromcol-1];
    if(fromrow==0 || fromrow==nnrow-1 
       || regions[fromrow][fromcol-1]!=regions[fromrow-1][fromcol-1]){
      if(to!=from->pred){
        to->pred=from;
        if(to->group==NOTINBUCKET){
          to->group=INBUCKET;
          to->next=nextnode;
          nextnode=to;
        }else if(to->group==ONTREE && (fromrow!=0 || tilerow==0)){
          TraceSecondaryArc(to,scndrynodes,nodesupp,scndryarcs,scndrycosts,
                            nnewnodesptr,nnewarcsptr,tilerow,tilecol,flowmax,
                            nrow,ncol,prevnrow,prevncol,params,costs,
                            rightedgecosts,loweredgecosts,leftedgecosts,
                            upperedgecosts,flows,rightedgeflows,
                            loweredgeflows,leftedgeflows,upperedgeflows,
                            updatednontilenodesptr,nupdatednontilenodesptr,
                            updatednontilenodesizeptr,inontilenodeoutarcptr,
                            totarclenptr);
        }
      }
    }
  }


  /* check upward arc */
  if(fromrow!=0){
    to=&primarynodes[fromrow-1][fromcol];
    if(fromcol==0 || fromcol==nncol-1
       || regions[fromrow-1][fromcol-1]!=regions[fromrow-1][fromcol]){
      if(to!=from->pred){
        to->pred=from;
        if(to->group==NOTINBUCKET){
          to->group=INBUCKET;
          to->next=nextnode;
          nextnode=to;
        }else if(to->group==ONTREE && (fromcol!=0 || tilecol==0)){
          TraceSecondaryArc(to,scndrynodes,nodesupp,scndryarcs,scndrycosts,
                            nnewnodesptr,nnewarcsptr,tilerow,tilecol,flowmax,
                            nrow,ncol,prevnrow,prevncol,params,costs,
                            rightedgecosts,loweredgecosts,leftedgecosts,
                            upperedgecosts,flows,rightedgeflows,
                            loweredgeflows,leftedgeflows,upperedgeflows,
                            updatednontilenodesptr,nupdatednontilenodesptr,
                            updatednontilenodesizeptr,inontilenodeoutarcptr,
                            totarclenptr);
        }
      }
    }
  }


  /* set return values */
  *nextnodeptr=nextnode;
  return(0);
}


/* function: SetUpperEdge()
 * ------------------------
 */
static
int SetUpperEdge(long ncol, long tilerow, long tilecol, void **voidcosts, 
                 void *voidcostsabove, float **unwphase, 
                 float *unwphaseabove, void **voidupperedgecosts, 
                 short **upperedgeflows, paramT *params, short **bulkoffsets){

  long col, reloffset;
  double dphi, dpsi;
  costT **upperedgecosts, **costs, *costsabove;
  smoothcostT **upperedgesmoothcosts, **smoothcosts, *smoothcostsabove;
  long nshortcycle;


  /* typecast generic pointers to costT pointers */
  upperedgecosts=(costT **)voidupperedgecosts;
  costs=(costT **)voidcosts;
  costsabove=(costT *)voidcostsabove;
  upperedgesmoothcosts=(smoothcostT **)voidupperedgecosts;
  smoothcosts=(smoothcostT **)voidcosts;
  smoothcostsabove=(smoothcostT *)voidcostsabove;

  /* see if tile is in top row */
  if(tilerow!=0){

    /* set up */
    nshortcycle=params->nshortcycle;
    reloffset=bulkoffsets[tilerow-1][tilecol]-bulkoffsets[tilerow][tilecol];

    /* loop over all arcs on the boundary */
    for(col=0;col<ncol;col++){
      dphi=(unwphaseabove[col]-unwphase[0][col])/TWOPI;
      upperedgeflows[0][col]=(short )LRound(dphi)-reloffset;
      dpsi=dphi-floor(dphi);
      if(dpsi>0.5){
        dpsi-=1.0;
      }
      if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
        upperedgecosts[0][col].offset=(short )LRound(nshortcycle*dpsi);
        upperedgecosts[0][col].sigsq=AvgSigSq(costs[0][col].sigsq,
                                              costsabove[col].sigsq);
        if(costs[0][col].dzmax>costsabove[col].dzmax){
          upperedgecosts[0][col].dzmax=costs[0][col].dzmax;
        }else{
          upperedgecosts[0][col].dzmax=costsabove[col].dzmax;
        }
        if(costs[0][col].laycost<costsabove[col].laycost){
          upperedgecosts[0][col].laycost=costs[0][col].laycost;
        }else{
          upperedgecosts[0][col].laycost=costsabove[col].laycost;
        }
      }else if(CalcCost==CalcCostSmooth){
        upperedgesmoothcosts[0][col].offset=(short )LRound(nshortcycle*dpsi);
        upperedgesmoothcosts[0][col].sigsq
          =AvgSigSq(smoothcosts[0][col].sigsq,smoothcostsabove[col].sigsq);
      }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
               || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
        ((short **)voidupperedgecosts)[0][col]=
          (((short **)voidcosts)[0][col]+((short *)voidcostsabove)[col])/2;
      }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
               || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
        ((bidircostT **)voidupperedgecosts)[0][col].posweight=
          (((bidircostT **)voidcosts)[0][col].posweight
           +((bidircostT *)voidcostsabove)[col].posweight)/2;
        ((bidircostT **)voidupperedgecosts)[0][col].negweight=
          (((bidircostT **)voidcosts)[0][col].negweight
           +((bidircostT *)voidcostsabove)[col].negweight)/2;
      }else{
        fflush(NULL);
        fprintf(sp0,"Illegal cost mode in SetUpperEdge().  This is a bug.\n");
        exit(ABNORMAL_EXIT);
      }
    }
  }else{
    if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
      for(col=0;col<ncol;col++){
        upperedgecosts[0][col].offset=LARGESHORT/2;
        upperedgecosts[0][col].sigsq=LARGESHORT;
        upperedgecosts[0][col].dzmax=LARGESHORT;
        upperedgecosts[0][col].laycost=0;
      }
    }else if(CalcCost==CalcCostSmooth){
      for(col=0;col<ncol;col++){
        upperedgesmoothcosts[0][col].offset=0;
        upperedgesmoothcosts[0][col].sigsq=LARGESHORT;
      }
    }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
             || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
      for(col=0;col<ncol;col++){
        ((short **)voidupperedgecosts)[0][col]=0;
      }
    }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
             || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
      for(col=0;col<ncol;col++){
        ((bidircostT **)voidupperedgecosts)[0][col].posweight=0;
        ((bidircostT **)voidupperedgecosts)[0][col].negweight=0;
      }
    }else{
      fflush(NULL);
      fprintf(sp0,"Illegal cost mode in SetUpperEdge().  This is a bug.\n");
      exit(ABNORMAL_EXIT);
    }
  }

  /* done */
  return(0);
  
}


/* function: SetLowerEdge()
 * ------------------------
 */
static
int SetLowerEdge(long nrow, long ncol, long tilerow, long tilecol, 
                 void **voidcosts, void *voidcostsbelow, 
                 float **unwphase, float *unwphasebelow, 
                 void **voidloweredgecosts, short **loweredgeflows, 
                 paramT *params, short **bulkoffsets){

  long *flowhistogram;
  long col, iflow, reloffset, nmax;
  long flowlimhi, flowlimlo, maxflow, minflow, tempflow;
  double dphi, dpsi;
  costT **loweredgecosts, **costs, *costsbelow;
  smoothcostT **loweredgesmoothcosts, **smoothcosts, *smoothcostsbelow;
  long nshortcycle;

  /* typecast generic pointers to costT pointers */
  loweredgecosts=(costT **)voidloweredgecosts;
  costs=(costT **)voidcosts;
  costsbelow=(costT *)voidcostsbelow;
  loweredgesmoothcosts=(smoothcostT **)voidloweredgecosts;
  smoothcosts=(smoothcostT **)voidcosts;
  smoothcostsbelow=(smoothcostT *)voidcostsbelow;

  /* see if tile is in bottom row */
  if(tilerow!=params->ntilerow-1){
  
    /* set up */
    nshortcycle=params->nshortcycle;
    flowlimhi=LARGESHORT;
    flowlimlo=-LARGESHORT;
    flowhistogram=(long *)CAlloc(flowlimhi-flowlimlo+1,sizeof(long));
    minflow=flowlimhi;
    maxflow=flowlimlo;

    /* loop over all arcs on the boundary */
    for(col=0;col<ncol;col++){
      dphi=(unwphase[nrow-1][col]-unwphasebelow[col])/TWOPI;
      tempflow=(short )LRound(dphi);
      loweredgeflows[0][col]=tempflow;
      if(tempflow<minflow){
        if(tempflow<flowlimlo){
          fflush(NULL);
          fprintf(sp0,"Overflow in tile offset\nAbort\n");
          exit(ABNORMAL_EXIT);
        }
        minflow=tempflow;
      }
      if(tempflow>maxflow){
        if(tempflow>flowlimhi){
          fflush(NULL);
          fprintf(sp0,"Overflow in tile offset\nAbort\n");
          exit(ABNORMAL_EXIT);
        }
        maxflow=tempflow;
      }
      flowhistogram[tempflow-flowlimlo]++;
      dpsi=dphi-floor(dphi);
      if(dpsi>0.5){
        dpsi-=1.0;
      }
      if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
        loweredgecosts[0][col].offset=(short )LRound(nshortcycle*dpsi);
        loweredgecosts[0][col].sigsq=AvgSigSq(costs[nrow-2][col].sigsq,
                                              costsbelow[col].sigsq);
        if(costs[nrow-2][col].dzmax>costsbelow[col].dzmax){
          loweredgecosts[0][col].dzmax=costs[nrow-2][col].dzmax;
        }else{
          loweredgecosts[0][col].dzmax=costsbelow[col].dzmax;
        }
        if(costs[nrow-2][col].laycost<costsbelow[col].laycost){
          loweredgecosts[0][col].laycost=costs[nrow-2][col].laycost;
        }else{
          loweredgecosts[0][col].laycost=costsbelow[col].laycost;
        }
      }else if(CalcCost==CalcCostSmooth){
        loweredgesmoothcosts[0][col].offset=(short )LRound(nshortcycle*dpsi);
        loweredgesmoothcosts[0][col].sigsq
          =AvgSigSq(smoothcosts[nrow-2][col].sigsq,smoothcostsbelow[col].sigsq);
      }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
               || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
        ((short **)voidloweredgecosts)[0][col]=
          (((short **)voidcosts)[nrow-2][col]
           +((short *)voidcostsbelow)[col])/2;
      }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
               || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
        ((bidircostT **)voidloweredgecosts)[0][col].posweight=
          (((bidircostT **)voidcosts)[nrow-2][col].posweight
           +((bidircostT *)voidcostsbelow)[col].posweight)/2;
        ((bidircostT **)voidloweredgecosts)[0][col].negweight=
          (((bidircostT **)voidcosts)[nrow-2][col].negweight
           +((bidircostT *)voidcostsbelow)[col].negweight)/2;
      }else{
        fflush(NULL);
        fprintf(sp0,"Illegal cost mode in SetLowerEdge().  This is a bug.\n");
        exit(ABNORMAL_EXIT);
      }
    }

    /* set bulk tile offset equal to mode of flow histogram */
    nmax=0;
    reloffset=0;
    for(iflow=minflow;iflow<=maxflow;iflow++){
      if(flowhistogram[iflow-flowlimlo]>nmax){
        nmax=flowhistogram[iflow-flowlimlo];
        reloffset=iflow;
      }
    }
    bulkoffsets[tilerow+1][tilecol]=bulkoffsets[tilerow][tilecol]-reloffset;

    /* subtract relative tile offset from edge flows */
    for(col=0;col<ncol;col++){
      loweredgeflows[0][col]-=reloffset;      
    }

    /* finish up */
    free(flowhistogram);

  }else{
    if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
      for(col=0;col<ncol;col++){
        loweredgecosts[0][col].offset=LARGESHORT/2;
        loweredgecosts[0][col].sigsq=LARGESHORT;
        loweredgecosts[0][col].dzmax=LARGESHORT;
        loweredgecosts[0][col].laycost=0;
      }
    }else if(CalcCost==CalcCostSmooth){
      for(col=0;col<ncol;col++){
        loweredgesmoothcosts[0][col].offset=0;
        loweredgesmoothcosts[0][col].sigsq=LARGESHORT;
      }
    }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
             || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
      for(col=0;col<ncol;col++){
        ((short **)voidloweredgecosts)[0][col]=0;
      }
    }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
             || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
      for(col=0;col<ncol;col++){
        ((bidircostT **)voidloweredgecosts)[0][col].posweight=0;
        ((bidircostT **)voidloweredgecosts)[0][col].negweight=0;
      }
    }else{
      fflush(NULL);
      fprintf(sp0,"Illegal cost mode in SetLowerEdge().  This is a bug.\n");
      exit(ABNORMAL_EXIT);
    }
  }

  /* done */
  return(0);

}


/* function: SetLeftEdge()
 * -----------------------
 */
static
int SetLeftEdge(long nrow, long prevncol, long tilerow, long tilecol, 
                void **voidcosts, void **voidlastcosts, float **unwphase, 
                float **lastunwphase, void **voidleftedgecosts, 
                short **leftedgeflows, paramT *params, short **bulkoffsets){

  long row, reloffset;
  double dphi, dpsi;
  costT  **leftedgecosts, **costs, **lastcosts;
  smoothcostT  **leftedgesmoothcosts, **smoothcosts, **lastsmoothcosts;
  long nshortcycle;

  /* typecast generic pointers to costT pointers */
  leftedgecosts=(costT **)voidleftedgecosts;
  costs=(costT **)voidcosts;
  lastcosts=(costT **)voidlastcosts;
  leftedgesmoothcosts=(smoothcostT **)voidleftedgecosts;
  smoothcosts=(smoothcostT **)voidcosts;
  lastsmoothcosts=(smoothcostT **)voidlastcosts;

  /* see if tile is in left column */
  if(tilecol!=0){

    /* set up */
    nshortcycle=params->nshortcycle;
    reloffset=bulkoffsets[tilerow][tilecol]-bulkoffsets[tilerow][tilecol-1];

    /* loop over all arcs on the boundary */
    for(row=0;row<nrow;row++){
      dphi=(unwphase[row][0]
            -lastunwphase[row][prevncol-1])/TWOPI;
      leftedgeflows[row][0]=(short )LRound(dphi)-reloffset;
      dpsi=dphi-floor(dphi);
      if(dpsi>0.5){
        dpsi-=1.0;
      }
      if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
        leftedgecosts[row][0].offset=(short )LRound(TILEDPSICOLFACTOR
                                                    *nshortcycle*dpsi);
        leftedgecosts[row][0].sigsq
          =AvgSigSq(costs[row+nrow-1][0].sigsq,
                    lastcosts[row+nrow-1][prevncol-2].sigsq);
        if(costs[row+nrow-1][0].dzmax>lastcosts[row+nrow-1][prevncol-2].dzmax){
          leftedgecosts[row][0].dzmax=costs[row+nrow-1][0].dzmax;
        }else{
          leftedgecosts[row][0].dzmax=lastcosts[row+nrow-1][prevncol-2].dzmax;
        }
        if(costs[row+nrow-1][0].laycost
           >lastcosts[row+nrow-1][prevncol-2].laycost){
          leftedgecosts[row][0].laycost=costs[row+nrow-1][0].laycost;
        }else{
          leftedgecosts[row][0].laycost
            =lastcosts[row+nrow-1][prevncol-2].laycost;
        }
      }else if(CalcCost==CalcCostSmooth){
        leftedgesmoothcosts[row][0].offset
          =(short )LRound(TILEDPSICOLFACTOR*nshortcycle*dpsi);
        leftedgesmoothcosts[row][0].sigsq
          =AvgSigSq(smoothcosts[row+nrow-1][0].sigsq,
                    lastsmoothcosts[row+nrow-1][prevncol-2].sigsq);
      }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
               || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
        ((short **)voidleftedgecosts)[row][0]=
          (((short **)voidcosts)[row+nrow-1][0]
           +((short **)voidlastcosts)[row+nrow-1][prevncol-2])/2;
      }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
               || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
        ((bidircostT **)voidleftedgecosts)[row][0].posweight=
          (((bidircostT **)voidcosts)[row+nrow-1][0].posweight
           +((bidircostT **)voidlastcosts)[row+nrow-1][prevncol-2].posweight)
          /2;
        ((bidircostT **)voidleftedgecosts)[row][0].negweight=
          (((bidircostT **)voidcosts)[row+nrow-1][0].negweight
           +((bidircostT **)voidlastcosts)[row+nrow-1][prevncol-2].negweight)
          /2;
      }else{
        fflush(NULL);
        fprintf(sp0,"Illegal cost mode in SetLeftEdge().  This is a bug.\n");
        exit(ABNORMAL_EXIT);
      }
    }
  }else{
    if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
      for(row=0;row<nrow;row++){
        leftedgecosts[row][0].offset=LARGESHORT/2;
        leftedgecosts[row][0].sigsq=LARGESHORT;
        leftedgecosts[row][0].dzmax=LARGESHORT;
        leftedgecosts[row][0].laycost=0;
      }
    }else if(CalcCost==CalcCostSmooth){
      for(row=0;row<nrow;row++){
        leftedgesmoothcosts[row][0].offset=0;
        leftedgesmoothcosts[row][0].sigsq=LARGESHORT;
      }
    }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
             || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
      for(row=0;row<nrow;row++){
        ((short **)voidleftedgecosts)[row][0]=0;
      }
    }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
             || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
      for(row=0;row<nrow;row++){
        ((bidircostT **)voidleftedgecosts)[row][0].posweight=0;
        ((bidircostT **)voidleftedgecosts)[row][0].negweight=0;
      }
    }else{
      fflush(NULL);
      fprintf(sp0,"Illegal cost mode in SetLeftEdge().  This is a bug.\n");
      exit(ABNORMAL_EXIT);
    }
  }

  /* done */
  return(0);

}


/* function: SetRightEdge()
 * ------------------------
 */
static
int SetRightEdge(long nrow, long ncol, long tilerow, long tilecol, 
                 void **voidcosts, void **voidnextcosts, 
                 float **unwphase, float **nextunwphase, 
                 void **voidrightedgecosts, short **rightedgeflows, 
                 paramT *params, short **bulkoffsets){

  long *flowhistogram;
  long row, iflow, reloffset, nmax;
  long flowlimhi, flowlimlo, maxflow, minflow, tempflow;
  double dphi, dpsi;
  costT  **rightedgecosts, **costs, **nextcosts;
  smoothcostT  **rightedgesmoothcosts, **smoothcosts, **nextsmoothcosts;
  long nshortcycle;

  /* typecast generic pointers to costT pointers */
  rightedgecosts=(costT **)voidrightedgecosts;
  costs=(costT **)voidcosts;
  nextcosts=(costT **)voidnextcosts;
  rightedgesmoothcosts=(smoothcostT **)voidrightedgecosts;
  smoothcosts=(smoothcostT **)voidcosts;
  nextsmoothcosts=(smoothcostT **)voidnextcosts;

  /* see if tile in right column */  
  if(tilecol!=params->ntilecol-1){

    /* set up */
    nshortcycle=params->nshortcycle;
    flowlimhi=LARGESHORT;
    flowlimlo=-LARGESHORT;
    flowhistogram=(long *)CAlloc(flowlimhi-flowlimlo+1,sizeof(long));
    minflow=flowlimhi;
    maxflow=flowlimlo;

    /* loop over all arcs on the boundary */
    for(row=0;row<nrow;row++){
      dphi=(nextunwphase[row][0]
            -unwphase[row][ncol-1])/TWOPI;
      tempflow=(short )LRound(dphi);
      rightedgeflows[row][0]=tempflow;
      if(tempflow<minflow){
        if(tempflow<flowlimlo){
          fflush(NULL);
          fprintf(sp0,"Overflow in tile offset\nAbort\n");
          exit(ABNORMAL_EXIT);
        }
        minflow=tempflow;
      }
      if(tempflow>maxflow){
        if(tempflow>flowlimhi){
          fflush(NULL);
          fprintf(sp0,"Overflow in tile offset\nAbort\n");
          exit(ABNORMAL_EXIT);
        }
        maxflow=tempflow;
      }
      flowhistogram[tempflow-flowlimlo]++;    
      dpsi=dphi-floor(dphi);
      if(dpsi>0.5){
        dpsi-=1.0;
      }
      if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
        rightedgecosts[row][0].offset=(short )LRound(TILEDPSICOLFACTOR
                                                     *nshortcycle*dpsi);
        rightedgecosts[row][0].sigsq
          =AvgSigSq(costs[row+nrow-1][ncol-2].sigsq,
                    nextcosts[row+nrow-1][0].sigsq);
        if(costs[row+nrow-1][ncol-2].dzmax>nextcosts[row+nrow-1][0].dzmax){
          rightedgecosts[row][0].dzmax=costs[row+nrow-1][ncol-2].dzmax;
        }else{
          rightedgecosts[row][0].dzmax=nextcosts[row+nrow-1][0].dzmax;
        }
        if(costs[row+nrow-1][ncol-2].laycost>nextcosts[row+nrow-1][0].laycost){
          rightedgecosts[row][0].laycost=costs[row+nrow-1][ncol-2].laycost;
        }else{
          rightedgecosts[row][0].laycost=nextcosts[row+nrow-1][0].laycost;
        }
      }else if(CalcCost==CalcCostSmooth){
        rightedgesmoothcosts[row][0].offset
          =(short )LRound(TILEDPSICOLFACTOR*nshortcycle*dpsi);
        rightedgesmoothcosts[row][0].sigsq
          =AvgSigSq(smoothcosts[row+nrow-1][ncol-2].sigsq,
                    nextsmoothcosts[row+nrow-1][0].sigsq);
      }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
               || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
        ((short **)voidrightedgecosts)[row][0]=
          (((short **)voidcosts)[row+nrow-1][ncol-2]
           +((short **)voidnextcosts)[row+nrow-1][0])/2;
      }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
               || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
        ((bidircostT **)voidrightedgecosts)[row][0].posweight=
          (((bidircostT **)voidcosts)[row+nrow-1][ncol-2].posweight
           +((bidircostT **)voidnextcosts)[row+nrow-1][ncol-2].posweight)/2;
        ((bidircostT **)voidrightedgecosts)[row][0].negweight=
          (((bidircostT **)voidcosts)[row+nrow-1][ncol-2].negweight
           +((bidircostT **)voidnextcosts)[row+nrow-1][ncol-2].negweight)/2;
      }else{
        fflush(NULL);
        fprintf(sp0,"Illegal cost mode in SetRightEdge().  This is a bug.\n");
        exit(ABNORMAL_EXIT);
      }
    }

    /* set bulk tile offset equal to mode of flow histogram */
    if(tilerow==0){
      nmax=0;
      reloffset=0;
      for(iflow=minflow;iflow<=maxflow;iflow++){
        if(flowhistogram[iflow-flowlimlo]>nmax){
          nmax=flowhistogram[iflow-flowlimlo];
          reloffset=iflow;
        }
      }
      bulkoffsets[tilerow][tilecol+1]=bulkoffsets[tilerow][tilecol]+reloffset;
    }else{
      reloffset=bulkoffsets[tilerow][tilecol+1]-bulkoffsets[tilerow][tilecol];
    }

    /* subtract relative tile offset from edge flows */
    for(row=0;row<nrow;row++){
      rightedgeflows[row][0]-=reloffset;
    }


    /* finish up */
    free(flowhistogram);

  }else{
    if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
      for(row=0;row<nrow;row++){
        rightedgecosts[row][0].offset=LARGESHORT/2;
        rightedgecosts[row][0].sigsq=LARGESHORT;
        rightedgecosts[row][0].dzmax=LARGESHORT;
        rightedgecosts[row][0].laycost=0;
      }
    }else if(CalcCost==CalcCostSmooth){
      for(row=0;row<nrow;row++){
        rightedgesmoothcosts[row][0].offset=0;
        rightedgesmoothcosts[row][0].sigsq=LARGESHORT;
      }
    }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1
             || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
      for(row=0;row<nrow;row++){
        ((short **)voidrightedgecosts)[row][0]=0;
      }
    }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
             || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
      for(row=0;row<nrow;row++){
        ((bidircostT **)voidrightedgecosts)[row][0].posweight=0;
        ((bidircostT **)voidrightedgecosts)[row][0].negweight=0;
      }
    }else{
      fflush(NULL);
      fprintf(sp0,"Illegal cost mode in SetRightEdge().  This is a bug.\n");
      exit(ABNORMAL_EXIT);
    }
  }

  /* done */
  return(0);

}


/* function: AvgSigSq()
 * --------------------
 * Return average of sigsq values after chcking for special value and
 * clipping to short.
 */
static
short AvgSigSq(short sigsq1, short sigsq2){

  int sigsqavg;

  
  /* if either value is special LARGESHORT value, use that */
  if(sigsq1==LARGESHORT || sigsq2==LARGESHORT){
    return(LARGESHORT);
  }

  /* compute average */
  sigsqavg=(int )ceil(0.5*(((int )sigsq1)+((int )sigsq2)));

  /* clip */
  sigsqavg=LClip(sigsqavg,-LARGESHORT,LARGESHORT);

  /* return */
  return((short )sigsqavg);
  
}


/* function: TraceSecondaryArc()
 * -----------------------------
 */
static
int TraceSecondaryArc(nodeT *primaryhead, nodeT **scndrynodes, 
                      nodesuppT **nodesupp, scndryarcT **scndryarcs, 
                      long ***scndrycosts, long *nnewnodesptr, 
                      long *nnewarcsptr, long tilerow, long tilecol, 
                      long flowmax, long nrow, long ncol, 
                      long prevnrow, long prevncol, paramT *params, 
                      void **tilecosts, void **rightedgecosts, 
                      void **loweredgecosts, void **leftedgecosts,
                      void **upperedgecosts, short **tileflows, 
                      short **rightedgeflows, short **loweredgeflows, 
                      short **leftedgeflows, short **upperedgeflows,
                      nodeT ***updatednontilenodesptr, 
                      long *nupdatednontilenodesptr, 
                      long *updatednontilenodesizeptr,
                      short **inontilenodeoutarcptr, long *totarclenptr){

  long i, row, col, nnewnodes, arclen, ntilerow, ntilecol, arcnum;
  long tilenum, nflow, primaryarcrow, primaryarccol, poscost, negcost, nomcost;
  long nnrow, nncol, calccostnrow, nnewarcs, arroffset, nshortcycle;
  long mincost, mincostflow, minweight, maxcost;
  long *scndrycostarr;
  double sigsq, sumsigsqinv, tempdouble, tileedgearcweight;
  short **flows;
  void **costs;
  nodeT *tempnode, *primarytail, *scndrytail, *scndryhead;
  nodeT *primarydummy, *scndrydummy;
  nodesuppT *supptail, *supphead, *suppdummy;
  scndryarcT *newarc;
  signed char primaryarcdir, zerocost;


  /* do nothing if source is passed or if arc already done in previous tile */
  if(primaryhead->pred==NULL
     || (tilerow!=0 && primaryhead->row==0 && primaryhead->pred->row==0)
     || (tilecol!=0 && primaryhead->col==0 && primaryhead->pred->col==0)){
    return(0);
  }
  
  /* set up */
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  nnrow=nrow+1;
  nncol=ncol+1;
  tilenum=tilerow*ntilecol+tilecol;
  scndrycostarr=(long *)MAlloc((2*flowmax+2)*sizeof(long));
  tileedgearcweight=params->tileedgeweight;
  nshortcycle=params->nshortcycle;
  zerocost=FALSE;
  arroffset=0;
  sigsq=0;

  /* loop to determine appropriate value for arroffset */
  while(TRUE){

    /* initialize variables */
    arclen=0;
    sumsigsqinv=0;
    for(nflow=1;nflow<=2*flowmax;nflow++){
      scndrycostarr[nflow]=0;
    }

    /* loop over primary arcs on secondary arc again to get costs */
    primarytail=primaryhead->pred;
    tempnode=primaryhead;
    while(TRUE){

      /* get primary arc just traversed */
      arclen++;
      if(tempnode->col==primarytail->col+1){              /* rightward arc */
        primaryarcdir=1;
        primaryarccol=primarytail->col;
        if(primarytail->row==0){                               /* top edge */
          if(tilerow==0){
            zerocost=TRUE;
          }else{
            primaryarcrow=0;
            costs=upperedgecosts;
            flows=upperedgeflows;
            calccostnrow=2;
          }
        }else if(primarytail->row==nnrow-1){                /* bottom edge */
          if(tilerow==ntilerow-1){
            zerocost=TRUE;
          }else{
            primaryarcrow=0;
            costs=loweredgecosts;
            flows=loweredgeflows;
            calccostnrow=2;
          }
        }else{                                               /* normal arc */
          primaryarcrow=primarytail->row-1;
          costs=tilecosts;
          flows=tileflows;
          calccostnrow=nrow;
        }
      }else if(tempnode->row==primarytail->row+1){         /* downward arc */
        primaryarcdir=1;
        if(primarytail->col==0){                              /* left edge */
          if(tilecol==0){
            zerocost=TRUE;
          }else{
            primaryarcrow=primarytail->row;
            primaryarccol=0;
            costs=leftedgecosts;
            flows=leftedgeflows; 
            calccostnrow=0;
          }
        }else if(primarytail->col==nncol-1){                 /* right edge */
          if(tilecol==ntilecol-1){
            zerocost=TRUE;
          }else{
            primaryarcrow=primarytail->row;
            primaryarccol=0;
            costs=rightedgecosts;
            flows=rightedgeflows;
            calccostnrow=0;
          }
        }else{                                               /* normal arc */
          primaryarcrow=primarytail->row+nrow-1;
          primaryarccol=primarytail->col-1;
          costs=tilecosts;
          flows=tileflows;
          calccostnrow=nrow;
        }
      }else if(tempnode->col==primarytail->col-1){         /* leftward arc */
        primaryarcdir=-1;
        primaryarccol=primarytail->col-1;
        if(primarytail->row==0){                               /* top edge */
          if(tilerow==0){
            zerocost=TRUE;
          }else{
            primaryarcrow=0;
            costs=upperedgecosts;
            flows=upperedgeflows;
            calccostnrow=2;
          }
        }else if(primarytail->row==nnrow-1){                /* bottom edge */
          if(tilerow==ntilerow-1){
            zerocost=TRUE;
          }else{
            primaryarcrow=0;
            costs=loweredgecosts;
            flows=loweredgeflows;
            calccostnrow=2;
          }
        }else{                                               /* normal arc */
          primaryarcrow=primarytail->row-1;
          costs=tilecosts;
          flows=tileflows;
          calccostnrow=nrow;
        }
      }else{                                                 /* upward arc */
        primaryarcdir=-1;
        if(primarytail->col==0){                              /* left edge */
          if(tilecol==0){
            zerocost=TRUE;
          }else{
            primaryarcrow=primarytail->row-1; 
            primaryarccol=0;
            costs=leftedgecosts;
            flows=leftedgeflows;        
            calccostnrow=0;
          }
        }else if(primarytail->col==nncol-1){                 /* right edge */
          if(tilecol==ntilecol-1){
            zerocost=TRUE;
          }else{            
            primaryarcrow=primarytail->row-1; 
            primaryarccol=0;
            costs=rightedgecosts;
            flows=rightedgeflows;
            calccostnrow=0;
          }
        }else{                                               /* normal arc */
          primaryarcrow=primarytail->row+nrow-2; 
          primaryarccol=primarytail->col-1;
          costs=tilecosts;
          flows=tileflows;
          calccostnrow=nrow;
        }
      }

      /* keep absolute cost of arc to the previous node */
      if(!zerocost){

        /* accumulate incremental cost in table for each nflow increment */
        /* offset flow in flow array temporarily by arroffset then undo below */
        flows[primaryarcrow][primaryarccol]-=primaryarcdir*arroffset;
        nomcost=EvalCost(costs,flows,primaryarcrow,primaryarccol,calccostnrow,
                         params);
        for(nflow=1;nflow<=flowmax;nflow++){
          flows[primaryarcrow][primaryarccol]+=(primaryarcdir*nflow);
          poscost=EvalCost(costs,flows,primaryarcrow,primaryarccol,
                           calccostnrow,params);
          flows[primaryarcrow][primaryarccol]-=(2*primaryarcdir*nflow);
          negcost=EvalCost(costs,flows,primaryarcrow,primaryarccol,
                           calccostnrow,params);
          flows[primaryarcrow][primaryarccol]+=(primaryarcdir*nflow);
          tempdouble=(scndrycostarr[nflow]+(poscost-nomcost));
          if(tempdouble>LARGEINT){
            scndrycostarr[nflow]=LARGEINT;
          }else if(tempdouble<-LARGEINT){
            scndrycostarr[nflow]=-LARGEINT;
          }else{
            scndrycostarr[nflow]+=(poscost-nomcost);
          }
          tempdouble=(scndrycostarr[nflow+flowmax]+(negcost-nomcost));
          if(tempdouble>LARGEINT){
            scndrycostarr[nflow+flowmax]=LARGEINT;
          }else if(tempdouble<-LARGEINT){
            scndrycostarr[nflow+flowmax]=-LARGEINT;
          }else{
            scndrycostarr[nflow+flowmax]+=(negcost-nomcost);
          }
        }
        flows[primaryarcrow][primaryarccol]+=primaryarcdir*arroffset;

        /* accumulate term to be used for cost growth beyond table bounds */
        if(CalcCost==CalcCostTopo || CalcCost==CalcCostDefo){
          sigsq=((costT **)costs)[primaryarcrow][primaryarccol].sigsq;
        }else if(CalcCost==CalcCostSmooth){
          sigsq=((smoothcostT **)costs)[primaryarcrow][primaryarccol].sigsq;
        }else if(CalcCost==CalcCostL0 || CalcCost==CalcCostL1 
                 || CalcCost==CalcCostL2 || CalcCost==CalcCostLP){
          minweight=((short **)costs)[primaryarcrow][primaryarccol];
          if(minweight<1){
            sigsq=LARGESHORT;
          }else{
            sigsq=1.0/(double )minweight;
          }
        }else if(CalcCost==CalcCostL0BiDir || CalcCost==CalcCostL1BiDir
                 || CalcCost==CalcCostL2BiDir || CalcCost==CalcCostLPBiDir){
          minweight=LMin(((bidircostT **)costs)[primaryarcrow][primaryarccol]
                         .posweight,
                         ((bidircostT **)costs)[primaryarcrow][primaryarccol]
                         .negweight);
          if(minweight<1){
            sigsq=LARGESHORT;
          }else{
            sigsq=1.0/(double )minweight;
          }
        }
        if(sigsq<LARGESHORT){    /* zero cost arc if sigsq == LARGESHORT */
          sumsigsqinv+=(1.0/sigsq);
        }
      }

      /* break if found the secondary arc tail */
      if(primarytail->group==ONTREE){
        break;
      }  
    
      /* move up the tree */
      tempnode=primarytail;
      primarytail=primarytail->pred;

    } /* end while loop for tracing secondary arc for costs */

    /* break if we have a zero-cost arc on the edge of the full array */
    if(zerocost){
      break;
    }

    /* find flow index with minimum cost */
    mincost=0;
    maxcost=0;
    mincostflow=0;
    for(nflow=1;nflow<=flowmax;nflow++){
      if(scndrycostarr[nflow]<mincost){
        mincost=scndrycostarr[nflow];
        mincostflow=nflow;
      }
      if(scndrycostarr[flowmax+nflow]<mincost){
        mincost=scndrycostarr[flowmax+nflow];
        mincostflow=-nflow;
      }
      if(scndrycostarr[nflow]>maxcost){
        maxcost=scndrycostarr[nflow];
      }
      if(scndrycostarr[flowmax+nflow]>maxcost){
        maxcost=scndrycostarr[flowmax+nflow];
      }
    }

    /* if cost was all zero, treat as zero cost arc */
    if(maxcost==mincost){
      zerocost=TRUE;
      sumsigsqinv=0;
    }

    /* break if cost array adequately centered on minimum cost flow */
    if(mincostflow==0){
      break;
    }

    /* correct value of arroffset for next loop */
    if(mincostflow==flowmax){
      arroffset-=((long )floor(1.5*flowmax));
    }else if(mincostflow==-flowmax){
      arroffset+=((long )floor(1.5*flowmax));      
    }else{
      arroffset-=mincostflow;
    }

  } /* end while loop for determining arroffset */


  /* ignore this arc if primary head is same as tail (ie, if arc loops) */
  /* only way this can happen is if region is connected at one corner only */
  /* so any possible improvements should have been made by primary solver */
  if(primaryhead==primarytail){
    free(scndrycostarr);
    return(0);
  }

  /* see if we have a secondary arc on the edge of the full-sized array */
  /* these arcs have zero cost since the edge is treated as a single node */
  /* secondary arcs whose primary arcs all have zero cost are also zeroed */
  if(zerocost){

    /* set sum of standard deviations to indicate zero-cost secondary arc */
    scndrycostarr[0]=0;
    for(nflow=1;nflow<=2*flowmax;nflow++){
      scndrycostarr[nflow]=0;
    }
    scndrycostarr[2*flowmax+1]=ZEROCOSTARC;

  }else{

    /* give extra weight to arcs on tile edges */
    if((primaryhead->row==primarytail->row 
        && (primaryhead->row==0 || primaryhead->row==nnrow-1))
       || (primaryhead->col==primarytail->col
           && (primaryhead->col==0 || primaryhead->col==nncol-1))){
      for(nflow=1;nflow<=2*flowmax;nflow++){
        tempdouble=scndrycostarr[nflow]*tileedgearcweight;
        if(tempdouble>LARGEINT){
          scndrycostarr[nflow]=LARGEINT;
        }else if(tempdouble<-LARGEINT){
          scndrycostarr[nflow]=-LARGEINT;
        }else{
          scndrycostarr[nflow]=LRound(tempdouble);
        }
      }
      sumsigsqinv*=tileedgearcweight; 

    }

    /* store sum of primary cost variances at end of secondary cost array */
    tempdouble=sumsigsqinv*nshortcycle*nshortcycle;
    if(tempdouble<LARGEINT){
      scndrycostarr[2*flowmax+1]=LRound(tempdouble);
    }else{
      scndrycostarr[2*flowmax+1]=LARGEINT;
    }
    scndrycostarr[0]=arroffset;

  }


  /* find secondary nodes corresponding to primary head, tail */
  if(primarytail->row==0 && tilerow!=0){
    scndrytail=FindScndryNode(scndrynodes,nodesupp,
                              (tilerow-1)*ntilecol+tilecol,
                              prevnrow,primarytail->col);
  }else if(primarytail->col==0 && tilecol!=0){
    scndrytail=FindScndryNode(scndrynodes,nodesupp,
                              tilerow*ntilecol+(tilecol-1),
                              primarytail->row,prevncol);
  }else{
    scndrytail=FindScndryNode(scndrynodes,nodesupp,tilenum,
                              primarytail->row,primarytail->col);
  }
  if(primaryhead->row==0 && tilerow!=0){
    scndryhead=FindScndryNode(scndrynodes,nodesupp,
                              (tilerow-1)*ntilecol+tilecol,
                              prevnrow,primaryhead->col);
  }else if(primaryhead->col==0 && tilecol!=0){
    scndryhead=FindScndryNode(scndrynodes,nodesupp,
                              tilerow*ntilecol+(tilecol-1),
                              primaryhead->row,prevncol);
  }else{
    scndryhead=FindScndryNode(scndrynodes,nodesupp,tilenum,
                              primaryhead->row,primaryhead->col);
  }

  /* see if there is already arc between secondary head, tail */
  row=scndrytail->row;
  col=scndrytail->col;
  for(i=0;i<nodesupp[row][col].noutarcs;i++){
    tempnode=nodesupp[row][col].neighbornodes[i];
    if((nodesupp[row][col].outarcs[i]==NULL
        && tempnode->row==primaryhead->row 
        && tempnode->col==primaryhead->col)
       || (nodesupp[row][col].outarcs[i]!=NULL
           && tempnode->row==scndryhead->row 
           && tempnode->col==scndryhead->col)){

      /* see if secondary arc traverses only one primary arc */
      primarydummy=primaryhead->pred;
      if(primarydummy->group!=ONTREE){
      
        /* arc already exists, free memory for cost array (will trace again) */
        free(scndrycostarr);

        /* set up dummy node */
        primarydummy->group=ONTREE;
        nnewnodes=++(*nnewnodesptr);
        scndrynodes[tilenum]=(nodeT *)ReAlloc(scndrynodes[tilenum],
                                              nnewnodes*sizeof(nodeT));
        scndrydummy=&scndrynodes[tilenum][nnewnodes-1];
        nodesupp[tilenum]=(nodesuppT *)ReAlloc(nodesupp[tilenum],
                                               nnewnodes*sizeof(nodesuppT));
        suppdummy=&nodesupp[tilenum][nnewnodes-1];
        scndrydummy->row=tilenum;
        scndrydummy->col=nnewnodes-1;
        suppdummy->row=primarydummy->row;
        suppdummy->col=primarydummy->col;
        suppdummy->noutarcs=0;
        suppdummy->neighbornodes=NULL;
        suppdummy->outarcs=NULL;

        /* recursively call TraceSecondaryArc() to set up arcs */
        TraceSecondaryArc(primarydummy,scndrynodes,nodesupp,scndryarcs,
                          scndrycosts,nnewnodesptr,nnewarcsptr,tilerow,tilecol,
                          flowmax,nrow,ncol,prevnrow,prevncol,params,tilecosts,
                          rightedgecosts,loweredgecosts,leftedgecosts,
                          upperedgecosts,tileflows,rightedgeflows,
                          loweredgeflows,leftedgeflows,upperedgeflows,
                          updatednontilenodesptr,nupdatednontilenodesptr,
                          updatednontilenodesizeptr,inontilenodeoutarcptr,
                          totarclenptr);
        TraceSecondaryArc(primaryhead,scndrynodes,nodesupp,scndryarcs,
                          scndrycosts,nnewnodesptr,nnewarcsptr,tilerow,tilecol,
                          flowmax,nrow,ncol,prevnrow,prevncol,params,tilecosts,
                          rightedgecosts,loweredgecosts,leftedgecosts,
                          upperedgecosts,tileflows,rightedgeflows,
                          loweredgeflows,leftedgeflows,upperedgeflows,
                          updatednontilenodesptr,nupdatednontilenodesptr,
                          updatednontilenodesizeptr,inontilenodeoutarcptr,
                          totarclenptr);
      }else{

        /* only one primary arc; just delete other secondary arc */
        /* find existing secondary arc (must be in this tile) */
        /* swap direction of existing secondary arc if necessary */
        arcnum=0;
        while(TRUE){
          if(scndryarcs[tilenum][arcnum].from==primarytail 
             && scndryarcs[tilenum][arcnum].to==primaryhead){
            break;
          }else if(scndryarcs[tilenum][arcnum].from==primaryhead
                   && scndryarcs[tilenum][arcnum].to==primarytail){
            scndryarcs[tilenum][arcnum].from=primarytail;
            scndryarcs[tilenum][arcnum].to=primaryhead;
            break;
          }
          arcnum++;
        }

        /* assign cost of this secondary arc to existing secondary arc */
        free(scndrycosts[tilenum][arcnum]);
        scndrycosts[tilenum][arcnum]=scndrycostarr;

        /* update direction data in secondary arc structure */
        if(primarytail->col==primaryhead->col+1){
          scndryarcs[tilenum][arcnum].fromdir=RIGHT;
        }else if(primarytail->row==primaryhead->row+1){
          scndryarcs[tilenum][arcnum].fromdir=DOWN;
        }else if(primarytail->col==primaryhead->col-1){
          scndryarcs[tilenum][arcnum].fromdir=LEFT;
        }else{
          scndryarcs[tilenum][arcnum].fromdir=UP;
        }
      }

      /* we're done */
      return(0);
    }
  }

  /* set up secondary arc datastructures */
  nnewarcs=++(*nnewarcsptr);
  if(nnewarcs > SHRT_MAX){
    fflush(NULL);
    fprintf(sp0,"Exceeded maximum number of secondary arcs\n"
            "Decrease TILECOSTTHRESH and/or increase MINREGIONSIZE\n");
    exit(ABNORMAL_EXIT);  
  }
  scndryarcs[tilenum]=(scndryarcT *)ReAlloc(scndryarcs[tilenum],
                                            nnewarcs*sizeof(scndryarcT));
  newarc=&scndryarcs[tilenum][nnewarcs-1];
  newarc->arcrow=tilenum;
  newarc->arccol=nnewarcs-1;
  scndrycosts[tilenum]=(long **)ReAlloc(scndrycosts[tilenum],
                                        nnewarcs*sizeof(long *));
  scndrycosts[tilenum][nnewarcs-1]=scndrycostarr;

  /* update secondary node data */
  /* store primary nodes in nodesuppT neighbornodes[] arrays since */
  /* secondary node addresses change in ReAlloc() calls in TraceRegions() */
  supptail=&nodesupp[scndrytail->row][scndrytail->col];
  supphead=&nodesupp[scndryhead->row][scndryhead->col];
  supptail->noutarcs++;
  supptail->neighbornodes=(nodeT **)ReAlloc(supptail->neighbornodes,
                                            supptail->noutarcs
                                            *sizeof(nodeT *));
  supptail->neighbornodes[supptail->noutarcs-1]=primaryhead;
  primarytail->level=scndrytail->row;
  primarytail->incost=scndrytail->col;
  supptail->outarcs=(scndryarcT **)ReAlloc(supptail->outarcs,
                                           supptail->noutarcs
                                           *sizeof(scndryarcT *));
  supptail->outarcs[supptail->noutarcs-1]=NULL;
  supphead->noutarcs++;
  supphead->neighbornodes=(nodeT **)ReAlloc(supphead->neighbornodes,
                                            supphead->noutarcs
                                            *sizeof(nodeT *));
  supphead->neighbornodes[supphead->noutarcs-1]=primarytail;
  primaryhead->level=scndryhead->row;
  primaryhead->incost=scndryhead->col;
  supphead->outarcs=(scndryarcT **)ReAlloc(supphead->outarcs,
                                           supphead->noutarcs
                                           *sizeof(scndryarcT *));
  supphead->outarcs[supphead->noutarcs-1]=NULL;

  /* keep track of updated secondary nodes that were not in this tile */
  if(scndrytail->row!=tilenum){
    if(++(*nupdatednontilenodesptr)==(*updatednontilenodesizeptr)){
      (*updatednontilenodesizeptr)+=INITARRSIZE;
      (*updatednontilenodesptr)=(nodeT **)ReAlloc((*updatednontilenodesptr),
                                                  (*updatednontilenodesizeptr)
                                                  *sizeof(nodeT *));
      (*inontilenodeoutarcptr)=(short *)ReAlloc((*inontilenodeoutarcptr),
                                                (*updatednontilenodesizeptr)
                                                *sizeof(short));
    }    
    (*updatednontilenodesptr)[*nupdatednontilenodesptr-1]=scndrytail;
    (*inontilenodeoutarcptr)[*nupdatednontilenodesptr-1]=supptail->noutarcs-1;
  }
  if(scndryhead->row!=tilenum){
    if(++(*nupdatednontilenodesptr)==(*updatednontilenodesizeptr)){
      (*updatednontilenodesizeptr)+=INITARRSIZE;
      (*updatednontilenodesptr)=(nodeT **)ReAlloc((*updatednontilenodesptr),
                                                  (*updatednontilenodesizeptr)
                                                  *sizeof(nodeT *));
      (*inontilenodeoutarcptr)=(short *)ReAlloc((*inontilenodeoutarcptr),
                                                (*updatednontilenodesizeptr)
                                                *sizeof(short));
    }    
    (*updatednontilenodesptr)[*nupdatednontilenodesptr-1]=scndryhead;
    (*inontilenodeoutarcptr)[*nupdatednontilenodesptr-1]=supphead->noutarcs-1;
  }

  /* set up node data in secondary arc structure */
  newarc->from=primarytail;
  newarc->to=primaryhead;
  
  /* set up direction data in secondary arc structure */
  tempnode=primaryhead->pred;
  if(tempnode->col==primaryhead->col+1){
    newarc->fromdir=RIGHT;
  }else if(tempnode->row==primaryhead->row+1){
    newarc->fromdir=DOWN;
  }else if(tempnode->col==primaryhead->col-1){
    newarc->fromdir=LEFT;
  }else{
    newarc->fromdir=UP;
  }

  /* add number of primary arcs in secondary arc to counter */
  (*totarclenptr)+=arclen;

  /* done */
  return(0);

}


/* function: FindScndryNode()
 * --------------------------
 */
static
nodeT *FindScndryNode(nodeT **scndrynodes, nodesuppT **nodesupp, 
                      long tilenum, long primaryrow, long primarycol){

  long nodenum;
  nodesuppT *nodesuppptr;

  /* set temporary variables */
  nodesuppptr=nodesupp[tilenum];

  /* loop over all nodes in the tile until we find a match */
  nodenum=0;
  while(nodesuppptr[nodenum].row!=primaryrow
        || nodesuppptr[nodenum].col!=primarycol){
    nodenum++;
  }
  return(&scndrynodes[tilenum][nodenum]);
}


/* function: IntegrateSecondaryFlows()
 * -----------------------------------
 */
static
int IntegrateSecondaryFlows(long linelen, long nlines, nodeT **scndrynodes, 
                            nodesuppT **nodesupp, scndryarcT **scndryarcs, 
                            int *nscndryarcs, short **scndryflows, 
                            short **bulkoffsets, outfileT *outfiles, 
                            paramT *params){
  
  FILE *outfp;
  float **unwphase, **tileunwphase, **mag, **tilemag;
  float *outline;
  long row, col, colstart, nrow, ncol, nnrow, nncol, maxcol;
  long readtilelinelen, readtilenlines, nextcoloffset, nextrowoffset;
  long tilerow, tilecol, ntilerow, ntilecol, rowovrlp, colovrlp;
  long ni, nj, tilenum;
  double tileoffset;
  short **regions, **tileflows;
  char realoutfile[MAXSTRLEN], readfile[MAXSTRLEN], tempstring[MAXTMPSTRLEN];
  char path[MAXSTRLEN], basename[MAXSTRLEN];
  signed char writeerror;
  tileparamT readtileparams[1];
  outfileT readtileoutfiles[1];


  /* initialize stack structures to zero for good measure */
  memset(readtileparams,0,sizeof(tileparamT));
  memset(readtileoutfiles,0,sizeof(outfileT));
  memset(realoutfile,0,MAXSTRLEN);
  memset(readfile,0,MAXSTRLEN);
  memset(tempstring,0,MAXSTRLEN);
  memset(path,0,MAXSTRLEN);
  memset(basename,0,MAXSTRLEN);

  /* set up */
  fprintf(sp1,"Integrating secondary flows\n");
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  rowovrlp=params->rowovrlp;
  colovrlp=params->colovrlp;
  ni=ceil((nlines+(ntilerow-1)*rowovrlp)/(double )ntilerow);
  nj=ceil((linelen+(ntilecol-1)*colovrlp)/(double )ntilecol);
  nextcoloffset=0;
  writeerror=FALSE;
  nrow=0;

  /* get memory */
  regions=(short **)Get2DMem(ni,nj,sizeof(short *),sizeof(short));
  tileflows=(short **)Get2DRowColMem(ni+2,nj+2,sizeof(short *),sizeof(short));
  tileunwphase=(float **)Get2DMem(ni,nj,sizeof(float *),sizeof(float));
  tilemag=(float **)Get2DMem(ni,nj,sizeof(float *),sizeof(float));
  unwphase=(float **)Get2DMem(ni,linelen,sizeof(float *),sizeof(float));
  mag=(float **)Get2DMem(ni,linelen,sizeof(float *),sizeof(float));
  outline=(float *)MAlloc(2*linelen*sizeof(float));

  /* flip sign of bulk offsets if flip flag is set */
  /* do this and flip flow signs instead of flipping phase signs */
  if(params->flipphasesign){
    for(row=0;row<ntilerow;row++){
      for(col=0;col<ntilecol;col++){
        bulkoffsets[row][col]*=-1;
      }
    }
  }

  /* open output file */
  outfp=OpenOutputFile(outfiles->outfile,realoutfile);

  /* process each tile row */
  for(tilerow=0;tilerow<ntilerow;tilerow++){
    
    /* process each tile column, place into unwrapped tile row array */
    nextrowoffset=0;
    for(tilecol=0;tilecol<ntilecol;tilecol++){

      /* use SetupTile() to set filenames; tile params overwritten below */
      SetupTile(nlines,linelen,params,readtileparams,outfiles,
                readtileoutfiles,tilerow,tilecol);
      colstart=readtileparams->firstcol;
      readtilenlines=readtileparams->nrow;
      readtilelinelen=readtileparams->ncol;

      /* set tile read parameters */
      SetTileReadParams(readtileparams,readtilenlines,readtilelinelen,
                        tilerow,tilecol,nlines,linelen,params);
      colstart+=readtileparams->firstcol;
      nrow=readtileparams->nrow;
      ncol=readtileparams->ncol;
      nnrow=nrow+1;
      nncol=ncol+1;

      /* read unwrapped phase */
      /* phase sign not flipped for positive baseline */
      /* since flow will be flipped if necessary */
      if(TMPTILEOUTFORMAT==ALT_LINE_DATA){
        ReadAltLineFile(&tilemag,&tileunwphase,readtileoutfiles->outfile,
                        readtilelinelen,readtilenlines,readtileparams);
      }else if(TMPTILEOUTFORMAT==FLOAT_DATA){
        Read2DArray((void ***)&tileunwphase,readtileoutfiles->outfile,
                    readtilelinelen,readtilenlines,readtileparams,
                    sizeof(float *),sizeof(float));
      }
        
      /* read regions */
      ParseFilename(outfiles->outfile,path,basename);
      sprintf(tempstring,"%s/%s%s_%ld_%ld.%ld%s",
              params->tiledir,TMPTILEROOT,basename,tilerow,tilecol,
              readtilelinelen,REGIONSUFFIX);
      StrNCopy(readfile,tempstring,MAXSTRLEN);
      Read2DArray((void ***)&regions,readfile,readtilelinelen,readtilenlines,
                  readtileparams,sizeof(short *),sizeof(short));

      /* remove temporary files unless told so save them */
      if(params->rmtmptile){
        unlink(readtileoutfiles->outfile);
        unlink(readfile);
      }

      /* zero out primary flow array */
      for(row=0;row<2*nrow+1;row++){
        if(row<nrow){
          maxcol=ncol;
        }else{
          maxcol=ncol+1;
        }
        for(col=0;col<maxcol;col++){
          tileflows[row][col]=0;
        }
      }

      /* loop over each secondary arc in this tile and parse flows */
      /* if flip flag set, flow derived from flipped phase array */
      /* flip flow for integration in ParseSecondaryFlows() */
      tilenum=tilerow*ntilecol+tilecol;
      ParseSecondaryFlows(tilenum,nscndryarcs,tileflows,regions,scndryflows,
                          nodesupp,scndryarcs,nrow,ncol,ntilerow,ntilecol,
                          params);

      /* place tile mag, adjusted unwrapped phase into output arrays */
      mag[0][colstart]=tilemag[0][0];
      if(tilecol==0){
        tileoffset=TWOPI*nextcoloffset;
      }else{
        tileoffset=TWOPI*nextrowoffset;
      }
      unwphase[0][colstart]=tileunwphase[0][0]+tileoffset;
      for(col=1;col<ncol;col++){
        mag[0][colstart+col]=tilemag[0][col];
        unwphase[0][colstart+col]
          =(float )((double )unwphase[0][colstart+col-1]
                    +(double )tileunwphase[0][col]
                    -(double )tileunwphase[0][col-1]
                    +tileflows[nnrow][col]*TWOPI);
      }
      if(tilecol!=ntilecol-1){
        nextrowoffset=(LRound((unwphase[0][colstart+ncol-1]
                               -tileunwphase[0][ncol-1])/TWOPI)
                       +tileflows[nnrow][nncol-1]
                       +bulkoffsets[tilerow][tilecol]
                       -bulkoffsets[tilerow][tilecol+1]); 
      }
      for(row=1;row<nrow;row++){
        for(col=0;col<ncol;col++){
          mag[row][colstart+col]=tilemag[row][col];
          unwphase[row][colstart+col]
            =(float )((double )unwphase[row-1][colstart+col]
                      +(double )tileunwphase[row][col]
                      -(double )tileunwphase[row-1][col]
                      -tileflows[row][col]*TWOPI);
        }
      }
      if(tilecol==0 && tilerow!=ntilerow-1){
        nextcoloffset=(LRound((unwphase[nrow-1][colstart]
                              -tileunwphase[nrow-1][0])/TWOPI)
                       -tileflows[nnrow-1][0]
                       +bulkoffsets[tilerow][tilecol]
                       -bulkoffsets[tilerow+1][tilecol]);
      }

    } /* end loop over tile columns */

    /* write out tile row */
    for(row=0;row<nrow;row++){
      if(outfiles->outfileformat==ALT_LINE_DATA){
        if(fwrite(mag[row],sizeof(float),linelen,outfp)!=linelen
           || fwrite(unwphase[row],sizeof(float),linelen,outfp)!=linelen){
          writeerror=TRUE;
          break;
        }
      }else if(outfiles->outfileformat==ALT_SAMPLE_DATA){
        for(col=0;col<linelen;col++){
          outline[2*col]=mag[row][col];
          outline[2*col+1]=unwphase[row][col];
        }
        if(fwrite(outline,sizeof(float),2*linelen,outfp)!=2*linelen){
          writeerror=TRUE;
          break;
        }
      }else{
        if(fwrite(unwphase[row],sizeof(float),linelen,outfp)!=linelen){
          writeerror=TRUE;
          break;
        }
      }
    }
    if(writeerror){
      fflush(NULL);
      fprintf(sp0,"Error while writing to file %s (device full?)\nAbort\n",
              realoutfile);
      exit(ABNORMAL_EXIT);
    }

  } /* end loop over tile rows */


  /* close output file, free memory */
  fprintf(sp1,"Output written to file %s\n",realoutfile);
  if(fclose(outfp)){
    fflush(NULL);
    fprintf(sp0,"WARNING: problem closing file %s (disk full?)\n",realoutfile);
  }
  Free2DArray((void **)regions,ni);
  Free2DArray((void **)tileflows,ni);
  Free2DArray((void **)tileunwphase,ni);
  Free2DArray((void **)tilemag,ni);
  Free2DArray((void **)unwphase,ni);
  Free2DArray((void **)mag,ni);
  free(outline);
  return(0);
  
}


/* function: ParseSecondaryFlows()
 * -------------------------------
 */
static
int ParseSecondaryFlows(long tilenum, int *nscndryarcs, short **tileflows, 
                         short **regions, short **scndryflows, 
                         nodesuppT **nodesupp, scndryarcT **scndryarcs, 
                         long nrow, long ncol, long ntilerow, long ntilecol,
                         paramT *params){

  nodeT *scndryfrom, *scndryto;
  long arcnum, nnrow, nncol, nflow, primaryfromrow, primaryfromcol;
  long prevrow, prevcol, thisrow, thiscol, nextrow, nextcol;
  signed char phaseflipsign;


  /* see if we need to flip sign of flow because of positive topo baseline */
  if(params->flipphasesign){
    phaseflipsign=-1;
  }else{
    phaseflipsign=1;
  }

  /* loop over all arcs in tile */
  for(arcnum=0;arcnum<nscndryarcs[tilenum];arcnum++){

    /* do nothing if prev arc has no secondary flow */
    nflow=phaseflipsign*scndryflows[tilenum][arcnum];
    if(nflow){

      /* get arc info */
      nnrow=nrow+1;
      nncol=ncol+1;
      scndryfrom=scndryarcs[tilenum][arcnum].from;
      scndryto=scndryarcs[tilenum][arcnum].to;
      if(scndryfrom->row==tilenum){
        primaryfromrow=nodesupp[scndryfrom->row][scndryfrom->col].row;
        primaryfromcol=nodesupp[scndryfrom->row][scndryfrom->col].col;
      }else if(scndryfrom->row==tilenum-ntilecol){
        primaryfromrow=0;
        primaryfromcol=nodesupp[scndryfrom->row][scndryfrom->col].col;
      }else if(scndryfrom->row==tilenum-1){
        primaryfromrow=nodesupp[scndryfrom->row][scndryfrom->col].row;
        primaryfromcol=0;
      }else{
        primaryfromrow=0;
        primaryfromcol=0;
      }
      if(scndryto->row==tilenum){
        thisrow=nodesupp[scndryto->row][scndryto->col].row;
        thiscol=nodesupp[scndryto->row][scndryto->col].col;
      }else if(scndryto->row==tilenum-ntilecol){
        thisrow=0;
        thiscol=nodesupp[scndryto->row][scndryto->col].col;
      }else if(scndryto->row==tilenum-1){
        thisrow=nodesupp[scndryto->row][scndryto->col].row;
        thiscol=0;
      }else{
        thisrow=0;
        thiscol=0;
      }

      /* set initial direction out of secondary arc head */
      switch(scndryarcs[tilenum][arcnum].fromdir){
      case RIGHT:
        nextrow=thisrow;
        nextcol=thiscol+1;
        tileflows[thisrow][thiscol]-=nflow;
        break;
      case DOWN:
        nextrow=thisrow+1;
        nextcol=thiscol;
        tileflows[nnrow+thisrow][thiscol]-=nflow;
        break;
      case LEFT:
        nextrow=thisrow;
        nextcol=thiscol-1;
        tileflows[thisrow][thiscol-1]+=nflow;
        break;
      default:
        nextrow=thisrow-1;
        nextcol=thiscol;
        tileflows[nnrow+thisrow-1][thiscol]+=nflow;
        break;
      }

      /* use region data to trace path between secondary from, to */
      while(!(nextrow==primaryfromrow && nextcol==primaryfromcol)){

        /* move to next node */
        prevrow=thisrow;
        prevcol=thiscol;
        thisrow=nextrow;
        thiscol=nextcol;
    
        /* check rightward arc */
        if(thiscol!=nncol-1){
          if(thisrow==0 || thisrow==nnrow-1
             || regions[thisrow-1][thiscol]!=regions[thisrow][thiscol]){
            if(!(thisrow==prevrow && thiscol+1==prevcol)){
              tileflows[thisrow][thiscol]-=nflow;
              nextcol++;
            }
          }
        }

        /* check downward arc */
        if(thisrow!=nnrow-1){
          if(thiscol==0 || thiscol==nncol-1
             || regions[thisrow][thiscol]!=regions[thisrow][thiscol-1]){
            if(!(thisrow+1==prevrow && thiscol==prevcol)){
              tileflows[nnrow+thisrow][thiscol]-=nflow;
              nextrow++;
            }
          }
        }
    
        /* check leftward arc */
        if(thiscol!=0){
          if(thisrow==0 || thisrow==nnrow-1
             || regions[thisrow][thiscol-1]!=regions[thisrow-1][thiscol-1]){
            if(!(thisrow==prevrow && thiscol-1==prevcol)){
              tileflows[thisrow][thiscol-1]+=nflow;
              nextcol--;
            }
          }
        }

        /* check upward arc */
        if(thisrow!=0){
          if(thiscol==0 || thiscol==nncol-1
             || regions[thisrow-1][thiscol-1]!=regions[thisrow-1][thiscol]){
            if(!(thisrow-1==prevrow && thiscol==prevcol)){
              tileflows[nnrow+thisrow-1][thiscol]+=nflow;
              nextrow--;
            }
          }
        }
      }   
    }
  }
  return(0);
}


/* function: AssembleTileConnComps()
 * ---------------------------------
 * Assemble conntected components per tile.
 */
static
int AssembleTileConnComps(long linelen, long nlines,
                          outfileT *outfiles, paramT *params){

  int ipass;
  long k;
  long row, col, colstart, nrow, ncol;
  long readtilelinelen, readtilenlines;
  long tilerow, tilecol, ntilerow, ntilecol, rowovrlp, colovrlp;
  long ni, nj, tilenum;
  unsigned int iconncomp, iconncompmax;
  long ntileconncomp, nconncomp;
  long ntileconncompmem, nconncompmem, nmemold;
  char realoutfile[MAXSTRLEN], readfile[MAXSTRLEN], tempstring[MAXTMPSTRLEN];
  char path[MAXSTRLEN], basename[MAXSTRLEN];
  signed char writeerror;
  tileparamT readtileparams[1];
  outfileT readtileoutfiles[1];
  unsigned int *tilemapping;
  unsigned int **tileconncomps, **tilerowconncomps;
  unsigned char **ucharbuf;
  unsigned char *ucharoutbuf;
  conncompsizeT *tileconncompsizes, *conncompsizes;
  FILE *outfp;

  
  /* initialize stack structures to zero for good measure */
  memset(readtileparams,0,sizeof(tileparamT));
  memset(readtileoutfiles,0,sizeof(outfileT));
  memset(realoutfile,0,MAXSTRLEN);
  memset(readfile,0,MAXSTRLEN);
  memset(tempstring,0,MAXSTRLEN);
  memset(path,0,MAXSTRLEN);
  memset(basename,0,MAXSTRLEN);

  /* set up */
  fprintf(sp1,"Assembling tile connected components\n");
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  rowovrlp=params->rowovrlp;
  colovrlp=params->colovrlp;
  ni=ceil((nlines+(ntilerow-1)*rowovrlp)/(double )ntilerow);
  nj=ceil((linelen+(ntilecol-1)*colovrlp)/(double )ntilecol);
  writeerror=FALSE;
  nrow=0;
  conncompsizes=NULL;
  nconncomp=0;
  nconncompmem=0;
  tileconncompsizes=NULL;
  ntileconncompmem=0;
  iconncompmax=0;
  
  /* get memory */
  tileconncomps=(unsigned int **)Get2DMem(ni,nj,sizeof(unsigned int *),
                                          sizeof(unsigned int));
  tilerowconncomps=(unsigned int **)Get2DMem(ni,linelen,sizeof(unsigned int *),
                                             sizeof(unsigned int));
  ucharbuf=(unsigned char **)Get2DMem(ni,nj,sizeof(unsigned char *),
                                      sizeof(unsigned char));
  ucharoutbuf=(unsigned char *)MAlloc(linelen*sizeof(unsigned char));
  tilemapping=NULL;

  /* open output file */
  outfp=OpenOutputFile(outfiles->conncompfile,realoutfile);

  /* do two passes looping over all tiles */
  for(ipass=0;ipass<2;ipass++){

    /* process each tile row */
    for(tilerow=0;tilerow<ntilerow;tilerow++){
    
      /* process each tile column */
      for(tilecol=0;tilecol<ntilecol;tilecol++){

        /* use SetupTile() to set filenames; tile params overwritten below */
        SetupTile(nlines,linelen,params,readtileparams,outfiles,
                  readtileoutfiles,tilerow,tilecol);
        colstart=readtileparams->firstcol;
        readtilenlines=readtileparams->nrow;
        readtilelinelen=readtileparams->ncol;

        /* set tile read parameters */
        SetTileReadParams(readtileparams,readtilenlines,readtilelinelen,
                          tilerow,tilecol,nlines,linelen,params);
        colstart+=readtileparams->firstcol;
        nrow=readtileparams->nrow;
        ncol=readtileparams->ncol;

        /* set tile number */
        tilenum=tilerow*ntilecol+tilecol;
        
        /* read connected components for tile */
        if(params->conncompouttype==CONNCOMPOUTTYPEUCHAR){
          Read2DArray((void ***)&ucharbuf,readtileoutfiles->conncompfile,
                      readtilelinelen,readtilenlines,readtileparams,
                      sizeof(unsigned char *),sizeof(unsigned char));
          for(row=0;row<nrow;row++){
            for(col=0;col<ncol;col++){
              tileconncomps[row][col]=(unsigned int )ucharbuf[row][col];
            }
          }
        }else{
          Read2DArray((void ***)&tileconncomps,readtileoutfiles->conncompfile,
                      readtilelinelen,readtilenlines,readtileparams,
                      sizeof(unsigned int *),sizeof(unsigned int));
        }
      
        /* see which pass we are in */
        if(ipass==0){

          /* first pass */
          
          /* initialize tileconncomps array for this tile */
          ntileconncomp=0;
          for(k=0;k<ntileconncompmem;k++){
            tileconncompsizes[k].tilenum=tilenum;
            tileconncompsizes[k].icomptile=k+1;
            tileconncompsizes[k].icompfull=0;
            tileconncompsizes[k].npix=0;
          }
          
          /* loop over kept pixels and count pixels in each conncomp */
          for(row=0;row<nrow;row++){
            for(col=0;col<ncol;col++){

              /* see if have connected component (conncomp number not zero) */
              iconncomp=tileconncomps[row][col];
              if(iconncomp>0){

                /* get more memory for tile conncompsizeT array if needed */
                while(iconncomp>ntileconncompmem){
                  nmemold=ntileconncompmem;
                  ntileconncompmem+=CONNCOMPMEMINCR;
                  tileconncompsizes
                    =(conncompsizeT *)ReAlloc(tileconncompsizes,
                                              (ntileconncompmem
                                               *sizeof(conncompsizeT)));
                  for(k=nmemold;k<ntileconncompmem;k++){
                    tileconncompsizes[k].tilenum=tilenum;
                    tileconncompsizes[k].icomptile=k+1;
                    tileconncompsizes[k].icompfull=0;
                    tileconncompsizes[k].npix=0;
                  }
                }

                /* count number of connected components in tile */
                if(tileconncompsizes[iconncomp-1].npix==0){
                  tileconncompsizes[iconncomp-1].icomptile=iconncomp;
                  ntileconncomp++;
                }

                /* count number of pixels in this connected component */
                tileconncompsizes[iconncomp-1].npix++;

                /* keep max number of connected components in any tile */
                if(iconncomp>iconncompmax){
                  iconncompmax=iconncomp;
                }
                
              }
            }
          }

          /* get more memory for full set of connected components sizes */
          nmemold=nconncompmem;
          nconncompmem+=ntileconncomp;
          conncompsizes=(conncompsizeT *)ReAlloc(conncompsizes,
                                                 (nconncompmem
                                                  *sizeof(conncompsizeT)));

          /* store conncomp sizes from tile in full list */
          for(k=0;k<ntileconncompmem;k++){
            if(tileconncompsizes[k].npix>0){
              conncompsizes[nconncomp].tilenum=tileconncompsizes[k].tilenum;
              conncompsizes[nconncomp].icomptile=tileconncompsizes[k].icomptile;
              conncompsizes[nconncomp].icompfull=0;
              conncompsizes[nconncomp].npix=tileconncompsizes[k].npix;
              nconncomp++;
            }
          }
            
        }else{

          /* second pass */
          
          /* build lookup table for tile mapping for this tile */
          /* lookup table index is conncomp number minus one */
          for(k=0;k<iconncompmax;k++){
            tilemapping[k]=0;
          }
          for(k=0;k<nconncomp;k++){
            if(conncompsizes[k].tilenum==tilenum){
              iconncomp=conncompsizes[k].icomptile;
              tilemapping[iconncomp-1]=conncompsizes[k].icompfull;
            }
          }

          /* assign final conncomp number to output */
          for(row=0;row<nrow;row++){
            for(col=0;col<ncol;col++){
              iconncomp=tileconncomps[row][col];
              if(iconncomp>0){
                tilerowconncomps[row][colstart+col]=tilemapping[iconncomp-1];
              }else{
                tilerowconncomps[row][colstart+col]=0;
              }
            }
          }
          
          /* remove temporary files unless told so save them */
          if(params->rmtmptile){
            unlink(readtileoutfiles->conncompfile);
          }

        }

      } /* end loop over tile columns */

      /* write out tile row at end of second pass */
      if(ipass>0){
        for(row=0;row<nrow;row++){
          if(params->conncompouttype==CONNCOMPOUTTYPEUCHAR){
            for(k=0;k<linelen;k++){
              ucharoutbuf[k]=(unsigned char)tilerowconncomps[row][k];
            }
            if(fwrite(ucharoutbuf,sizeof(unsigned char),linelen,outfp)
               !=linelen){
              writeerror=TRUE;
              break;
            }
          }else{
            if(fwrite(tilerowconncomps[row],
                      sizeof(unsigned int),linelen,outfp)!=linelen){
              writeerror=TRUE;
              break;
            }
          }
        }
        if(writeerror){
          fflush(NULL);
          fprintf(sp0,"Error while writing to file %s (device full?)\nAbort\n",
                  realoutfile);
          exit(ABNORMAL_EXIT);
        }
      }

    } /* end loop over tile rows */

    /* at end of first pass, set up tile size array for next pass */
    if(ipass==0){
    
      /* sort tile size array into descending order */
      qsort(conncompsizes,nconncomp,sizeof(conncompsizeT),
            ConnCompSizeNPixCompare);

      /* keep no more than max number of connected components */
      if(nconncomp>params->maxncomps){
        nconncomp=params->maxncomps;
      }
      
      /* assign tile mappings */
      for(k=0;k<nconncomp;k++){
        conncompsizes[k].icompfull=k+1;
      }

      /* get memory for tile mapping lookup table */
      tilemapping=(unsigned int *)MAlloc(iconncompmax*sizeof(unsigned int));
      
    }
    
  } /* end loop over passes of tile reading */

  /* close output file */
  fprintf(sp1,"Assembled connected components (%ld) output written to file %s\n",
          nconncomp,realoutfile);
  if(fclose(outfp)){
    fflush(NULL);
    fprintf(sp0,"WARNING: problem closing file %s (disk full?)\n",
            realoutfile);
  }
  
  /* free memory */
  Free2DArray((void **)tileconncomps,ni);
  Free2DArray((void **)tilerowconncomps,ni);
  Free2DArray((void **)ucharbuf,ni);
  free(ucharoutbuf);
  if(ntileconncompmem>0){
    free(tileconncompsizes);
  }
  if(nconncompmem>0){
    free(conncompsizes);
  }
  free(tilemapping);
  
  /* done */
  return(0);
  
}


/* function: ConnCompSizeNPixCompare()
 * -----------------------------------
 * Compare npix member of conncompsizeT structures pointed to by
 * inputs for use with qsort() into descending order.
 */
static
int ConnCompSizeNPixCompare(const void *ptr1, const void *ptr2){
  return(((conncompsizeT *)ptr2)->npix-((conncompsizeT *)ptr1)->npix);
}


