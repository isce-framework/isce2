/*************************************************************************

  snaphu main source file
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



/* global (external) variable definitions */

/* flags used for signal handling */
char dumpresults_global = FALSE;
char requestedstop_global = FALSE;

/* ouput stream pointers */
/* sp0=error messages, sp1=status output, sp2=verbose, sp3=verbose counter */
FILE *sp0 = NULL;
FILE *sp1 = NULL;
FILE *sp2 = NULL;
FILE *sp3 = NULL;

/* node pointer for marking arc not on tree in apex array */
/* this should be treated as a constant */
nodeT NONTREEARC[1];

/* pointers to functions which calculate arc costs */
void (*CalcCost)(void **, long, long, long, long, long,
                 paramT *, long *, long *) = NULL;
long (*EvalCost)(void **, short **, long, long, long, paramT *) = NULL;


/* static (local) function prototypes */
static
int Unwrap(infileT *infiles, outfileT *outfiles, paramT *params, 
           long linelen, long nlines);
static
int UnwrapTile(infileT *infiles, outfileT *outfiles, paramT *params, 
                tileparamT *tileparams, long nlines, long linelen);



/***************************/
/* main program for snaphu */
/***************************/

int snaphu(infileT *infiles,outfileT *outfiles, paramT *params,long linelen) {

  /* variable declarations
  infileT infiles[1];
  outfileT outfiles[1];
  paramT params[1];
  time_t tstart;
  double cputimestart;
  long linelen, nlines;*/
	long nlines;


  /* get current wall clock and CPU time
  StartTimers(&tstart,&cputimestart);*/

  /* set output stream pointers (may be reset after inputs parsed) */
  SetStreamPointers();

  /* print greeting */
  fprintf(sp1,"\n%s v%s\n",PROGRAMNAME,VERSION);

  /* set default parameters */
  /*SetDefaults(infiles,outfiles,params);
  ReadConfigFile(DEF_SYSCONFFILE,infiles,outfiles,&linelen,params);*/

  /* parse the command line inputs */
  /*ProcessArgs(argc,argv,infiles,outfiles,&linelen,params);*/

  /* set verbose output if specified */
  SetVerboseOut(params);

  /* set names of dump files if necessary */
  SetDumpAll(outfiles,params);

  /* get number of lines in file */
  nlines=GetNLines(infiles,linelen,params);

  /* check validity of parameters */
  CheckParams(infiles,outfiles,linelen,nlines,params);

  /* log the runtime parameters */
  /*WriteConfigLogFile(argc,argv,infiles,outfiles,linelen,params);*/

  /* unwrap, forming tiles and reassembling if necessary */
  Unwrap(infiles,outfiles,params,linelen,nlines);

  /* finish up
  fprintf(sp1,"Program %s done\n",PROGRAMNAME);
  DisplayElapsedTime(tstart,cputimestart);
  exit(NORMAL_EXIT);*/

  return EXIT_SUCCESS;

} /* end of main() */


/* function: Unwrap()
 * ------------------
 * Sets parameters for each tile and calls UnwrapTile() to do the
 * unwrapping.
 */
static
int Unwrap(infileT *infiles, outfileT *outfiles, paramT *params, 
           long linelen, long nlines){

  long optiter, noptiter;
  long nexttilerow, nexttilecol, ntilerow, ntilecol, nthreads, nchildren;
  long sleepinterval;
  tileparamT tileparams[1];
  infileT iterinfiles[1];
  outfileT iteroutfiles[1];
  outfileT tileoutfiles[1];
  paramT iterparams[1];
  char tileinitfile[MAXSTRLEN];
  pid_t pid;
  int childstatus;
  double tilecputimestart;
  time_t tiletstart;
  signed char **dotilemask;


  /* initialize structure stack memory to zero for extra robustness */
  memset(tileparams,0,sizeof(tileparamT));
  memset(iterinfiles,0,sizeof(infileT));
  memset(iteroutfiles,0,sizeof(outfileT));
  memset(tileoutfiles,0,sizeof(outfileT));
  memset(iterparams,0,sizeof(paramT));
  memset(tileinitfile,0,MAXSTRLEN);

  /* see if we need to do single-tile reoptimization and set up if so */
  if(params->onetilereopt){
    noptiter=2;
  }else{
    noptiter=1;
  }
  
  /* iterate if necessary for single-tile reoptimization */
  for(optiter=0;optiter<noptiter;optiter++){

    /* initialize input and output file structures for this iteration */
    memcpy(iterinfiles,infiles,sizeof(infileT));
    memcpy(iteroutfiles,outfiles,sizeof(outfileT));
    memcpy(iterparams,params,sizeof(paramT));

    /* set up for iteration if doing tile init and one-tile reoptimization*/
    if(optiter==0){

      /* first iteration: see if there will be another iteration */
      if(noptiter>1){

        /* set up to write tile-mode unwrapped result to temporary file */
        SetTileInitOutfile(iteroutfiles->outfile,iterparams->parentpid);
        StrNCopy(tileinitfile,iteroutfiles->outfile,MAXSTRLEN);
        iteroutfiles->outfileformat=TILEINITFILEFORMAT;
        fprintf(sp1,"Starting first-round tile-mode unwrapping\n");
        
      }
      
    }else if(optiter==1){

      /* second iteration */
      /* set up to read unwrapped tile-mode result as single tile */
      StrNCopy(iterinfiles->infile,tileinitfile,MAXSTRLEN);
      iterinfiles->unwrappedinfileformat=TILEINITFILEFORMAT;
      iterparams->unwrapped=TRUE;
      iterparams->ntilerow=1;
      iterparams->ntilecol=1;
      iterparams->rowovrlp=0;
      iterparams->colovrlp=0;
      fprintf(sp1,"Starting second-round single-tile unwrapping\n");
      
    }else{
      fprintf(sp0,"ERROR: illegal optiter value in Unwrap()\n");
      exit(ABNORMAL_EXIT);
    }
    
    /* set up for unwrapping */
    ntilerow=iterparams->ntilerow;
    ntilecol=iterparams->ntilecol;
    nthreads=iterparams->nthreads;
    dumpresults_global=FALSE;
    requestedstop_global=FALSE;

    /* do the unwrapping */
    if(ntilerow==1 && ntilecol==1){

      /* only single tile */

      /* do the unwrapping */
      tileparams->firstrow=iterparams->piecefirstrow;
      tileparams->firstcol=iterparams->piecefirstcol;
      tileparams->nrow=iterparams->piecenrow;
      tileparams->ncol=iterparams->piecencol;
      UnwrapTile(iterinfiles,iteroutfiles,iterparams,tileparams,nlines,linelen);

    }else{

      /* don't unwrap if in assemble-only mode */
      if(!iterparams->assembleonly){

        /* set up mask for which tiles should be unwrapped */
        dotilemask=SetUpDoTileMask(iterinfiles,ntilerow,ntilecol);

        /* make a temporary directory into which tile files will be written */
        MakeTileDir(iterparams,iteroutfiles);

        /* different code for parallel or nonparallel operation */
        if(nthreads>1){

          /* parallel code */

          /* initialize */
          nexttilerow=0;
          nexttilecol=0;
          nchildren=0;
          sleepinterval=(long )ceil(nlines*linelen
                                    /((double )(ntilerow*ntilecol))
                                    *SECONDSPERPIXEL);

          /* trap signals so children get killed if parent dies */
          CatchSignals(KillChildrenExit);

          /* loop until we're done unwrapping */
          while(TRUE){

            /* unwrap next tile if there are free processors and tiles left */
            if(nchildren<nthreads && nexttilerow<ntilerow){
            
              /* see if next tile needs to be unwrapped */
              if(dotilemask[nexttilerow][nexttilecol]){

                /* wait to make sure file i/o, threads, and OS are synched */
                sleep(sleepinterval);
                
                /* fork to create new process */
                fflush(NULL);
                pid=fork();

              }else{

                /* tile did not need unwrapping, so set pid to parent pid */
                pid=iterparams->parentpid;

              }

              /* see if parent or child (or error) */
              if(pid<0){

                /* parent kills children and exits if there was a fork error */
                fflush(NULL);
                fprintf(sp0,"Error while forking\nAbort\n");
                kill(0,SIGKILL);
                exit(ABNORMAL_EXIT);

              }else if(pid==0){

                /* child executes this code after fork */

                /* reset signal handlers so that children exit nicely */
                CatchSignals(SignalExit);

                /* start timers for this tile */
                StartTimers(&tiletstart,&tilecputimestart);

                /* set up tile parameters */
                pid=getpid();
                fprintf(sp1,
                        "Unwrapping tile at row %ld, column %ld (pid %ld)\n",
                        nexttilerow,nexttilecol,(long )pid);
                SetupTile(nlines,linelen,iterparams,tileparams,
                          iteroutfiles,tileoutfiles,
                          nexttilerow,nexttilecol);
              
                /* reset stream pointers for logging */
                ChildResetStreamPointers(pid,nexttilerow,nexttilecol,
                                         iterparams);

                /* unwrap the tile */
                UnwrapTile(iterinfiles,tileoutfiles,iterparams,tileparams,
                           nlines,linelen);

                /* log elapsed time */
                DisplayElapsedTime(tiletstart,tilecputimestart);

                /* child exits when done unwrapping */
                exit(NORMAL_EXIT);

              }
              
              /* parent executes this code after fork */

              /* increment tile counters */
              if(++nexttilecol==ntilecol){
                nexttilecol=0;
                nexttilerow++;
              }

              /* increment counter of running child processes */
              if(pid!=iterparams->parentpid){
                nchildren++;
              }

            }else{

              /* wait for a child to finish (only parent gets here) */
              pid=wait(&childstatus);

              /* make sure child exited cleanly */
              if(!(WIFEXITED(childstatus)) || (WEXITSTATUS(childstatus))!=0){
                fflush(NULL);
                fprintf(sp0,"Unexpected or abnormal exit of child process %ld\n"
                        "Abort\n",(long )pid);
                signal(SIGTERM,SIG_IGN);
                kill(0,SIGTERM);
                exit(ABNORMAL_EXIT);
              }

              /* we're done if there are no more active children */
              /* shouldn't really need this sleep(), but be extra sure child */
              /*   outputs are really flushed and written to disk by OS */
              if(--nchildren==0){
                sleep(sleepinterval);
                break;
              }

            } /* end if free processor and tiles remaining */
          } /* end while loop */

          /* return signal handlers to default behavior */
          CatchSignals(SIG_DFL);

        }else{

          /* nonparallel code */

          /* loop over all tiles */
          for(nexttilerow=0;nexttilerow<ntilerow;nexttilerow++){
            for(nexttilecol=0;nexttilecol<ntilecol;nexttilecol++){
              if(dotilemask[nexttilerow][nexttilecol]){

                /* set up tile parameters */
                fprintf(sp1,"Unwrapping tile at row %ld, column %ld\n",
                        nexttilerow,nexttilecol);
                SetupTile(nlines,linelen,iterparams,tileparams,
                          iteroutfiles,tileoutfiles,
                          nexttilerow,nexttilecol);
            
                /* unwrap the tile */
                UnwrapTile(iterinfiles,tileoutfiles,iterparams,tileparams,
                           nlines,linelen);

              }
            }
          }

        } /* end if nthreads>1 */

        /* free tile mask memory */
        Free2DArray((void **)dotilemask,ntilerow);

      } /* end if !iterparams->assembleonly */

      /* reassemble tiles */
      AssembleTiles(iteroutfiles,iterparams,nlines,linelen);
    
    } /* end if multiple tiles */

    /* remove temporary tile file if desired at end of second iteration */
    if(iterparams->rmtileinit && optiter>0){
      unlink(tileinitfile);
    }
    
  } /* end of optiter loop */
  
  /* done */
  return(0);
  
} /* end of Unwrap() */


/* function: UnwrapTile()
 * ----------------------
 * This is the main phase unwrapping function for a single tile.
 */
static
int UnwrapTile(infileT *infiles, outfileT *outfiles, paramT *params, 
               tileparamT *tileparams,  long nlines, long linelen){

  /* variable declarations */
  long nrow, ncol, nnoderow, narcrow, n, ngroundarcs, iincrcostfile;
  long nflow, ncycle, mostflow, nflowdone;
  long candidatelistsize, candidatebagsize;
  long isource, nsource;
  long nnondecreasedcostiter;
  long *nconnectedarr;
  int *nnodesperrow, *narcsperrow;
  short **flows, **mstcosts;
  float **wrappedphase, **unwrappedphase, **mag, **unwrappedest;
  incrcostT **incrcosts;
  void **costs;
  totalcostT totalcost, oldtotalcost, mintotalcost;
  nodeT **sourcelist;
  nodeT *source, ***apexes;
  nodeT **nodes, ground[1];
  candidateT *candidatebag, *candidatelist;
  signed char **iscandidate;
  signed char notfirstloop, allmasked;
  bucketT *bkts;


  /* get size of tile */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;

  /* read input file (memory allocated by read function) */
  ReadInputFile(infiles,&mag,&wrappedphase,&flows,linelen,nlines,
                params,tileparams);

  /* read interferogram magnitude if specified separately */
  ReadMagnitude(mag,infiles,linelen,nlines,tileparams);

  /* read mask file and apply to magnitude */
  ReadByteMask(mag,infiles,linelen,nlines,tileparams,params);

  /* make sure we have at least one pixel that is not masked */
  allmasked=CheckMagMasking(mag,nrow,ncol);  

  /* read the coarse unwrapped estimate, if provided */
  unwrappedest=NULL;
  if(strlen(infiles->estfile)){
    ReadUnwrappedEstimateFile(&unwrappedest,infiles,linelen,nlines,
                              params,tileparams);

    /* subtract the estimate from the wrapped phase (and re-wrap) */
    FlattenWrappedPhase(wrappedphase,unwrappedest,nrow,ncol);

  }

  /* build the cost arrays */  
  BuildCostArrays(&costs,&mstcosts,mag,wrappedphase,unwrappedest,
                  linelen,nlines,nrow,ncol,params,tileparams,infiles,outfiles);

  /* if in quantify-only mode, evaluate cost of unwrapped input then return */
  if(params->eval){
    mostflow=Short2DRowColAbsMax(flows,nrow,ncol);
    fprintf(sp1,"Maximum flow on network: %ld\n",mostflow);
    totalcost=EvaluateTotalCost(costs,flows,nrow,ncol,NULL,params);
    fprintf(sp1,"Total solution cost: %.9g\n",(double )totalcost);
    Free2DArray((void **)costs,2*nrow-1);
    Free2DArray((void **)mag,nrow);
    Free2DArray((void **)wrappedphase,nrow);
    Free2DArray((void **)flows,2*nrow-1);
    return(1);
  }

  /* set network function pointers for grid network */
  SetGridNetworkFunctionPointers();

  /* initialize the flows (find simple unwrapping to get a feasible flow) */
  unwrappedphase=NULL;
  nodes=NULL;
  if(!params->unwrapped){

    /* see which initialization method to use */
    if(params->initmethod==MSTINIT){

      /* use minimum spanning tree (MST) algorithm */
      MSTInitFlows(wrappedphase,&flows,mstcosts,nrow,ncol,
                   &nodes,ground,params->initmaxflow);
    
    }else if(params->initmethod==MCFINIT){

      /* use minimum cost flow (MCF) algorithm */
      MCFInitFlows(wrappedphase,&flows,mstcosts,nrow,ncol,
                   params->cs2scalefactor);

    }else{
      fflush(NULL);
      fprintf(sp0,"Illegal initialization method\nAbort\n");
      exit(ABNORMAL_EXIT);
    }

    /* integrate the phase and write out if necessary */
    if(params->initonly || strlen(outfiles->initfile)){
      fprintf(sp1,"Integrating phase\n");
      unwrappedphase=(float **)Get2DMem(nrow,ncol,
                                        sizeof(float *),sizeof(float));
      IntegratePhase(wrappedphase,unwrappedphase,flows,nrow,ncol);
      if(unwrappedest!=NULL){
        Add2DFloatArrays(unwrappedphase,unwrappedest,nrow,ncol);
      }
      FlipPhaseArraySign(unwrappedphase,params,nrow,ncol);

      /* return if called in init only; otherwise, free memory and continue */
      if(params->initonly){
        fprintf(sp1,"Writing output to file %s\n",outfiles->outfile);
        WriteOutputFile(mag,unwrappedphase,outfiles->outfile,outfiles,
                        nrow,ncol);  
        Free2DArray((void **)mag,nrow);
        Free2DArray((void **)wrappedphase,nrow);
        Free2DArray((void **)unwrappedphase,nrow);
        if(nodes!=NULL){
          Free2DArray((void **)nodes,nrow-1);
        }
        Free2DArray((void **)flows,2*nrow-1);
        return(1);
      }else{
        fprintf(sp2,"Writing initialization to file %s\n",outfiles->initfile);
        WriteOutputFile(mag,unwrappedphase,outfiles->initfile,outfiles,
                        nrow,ncol);  
        Free2DArray((void **)unwrappedphase,nrow);
      }
    }
  }

  /* initialize network variables */
  InitNetwork(flows,&ngroundarcs,&ncycle,&nflowdone,&mostflow,&nflow,
              &candidatebagsize,&candidatebag,&candidatelistsize,
              &candidatelist,&iscandidate,&apexes,&bkts,&iincrcostfile,
              &incrcosts,&nodes,ground,&nnoderow,&nnodesperrow,&narcrow,
              &narcsperrow,nrow,ncol,&notfirstloop,&totalcost,params);
  oldtotalcost=totalcost;
  mintotalcost=totalcost;
  nnondecreasedcostiter=0;

  /* regrow regions with -G parameter */
  if(params->regrowconncomps){

    /* free up some memory */
    Free2DArray((void **)apexes,2*nrow-1);
    Free2DArray((void **)iscandidate,2*nrow-1);
    Free2DArray((void **)nodes,nrow-1);
    free(candidatebag);
    free(candidatelist);  
    free(bkts->bucketbase);

    /* grow connected components */
    GrowConnCompsMask(costs,flows,nrow,ncol,incrcosts,outfiles,params);

    /* free up remaining memory and return */
    Free2DArray((void **)incrcosts,2*nrow-1);
    Free2DArray((void **)costs,2*nrow-1);
    Free2DArray((void **)mag,nrow);
    Free2DArray((void **)wrappedphase,nrow);
    Free2DArray((void **)flows,2*nrow-1);
    free(nnodesperrow);
    free(narcsperrow);
    return(1);
  }

  /* mask zero-magnitude nodes so they are not considered in optimization */
  MaskNodes(nrow,ncol,nodes,ground,mag);

  /* if we have a single tile, trap signals for dumping results */
  if(params->ntilerow==1 && params->ntilecol==1){
    signal(SIGINT,SetDump);
    signal(SIGHUP,SetDump);
  }

  /* main loop: loop over flow increments and sources */
  if(!allmasked){
    fprintf(sp1,"Running nonlinear network flow optimizer\n");
    fprintf(sp1,"Maximum flow on network: %ld\n",mostflow);
    fprintf(sp2,"Number of nodes in network: %ld\n",(nrow-1)*(ncol-1)+1);
    while(TRUE){ 
 
      fprintf(sp1,"Flow increment: %ld  (Total improvements: %ld)\n",
              nflow,ncycle);

      /* set up the incremental (residual) cost arrays */
      SetupIncrFlowCosts(costs,incrcosts,flows,nflow,nrow,narcrow,narcsperrow,
                         params); 
      if(params->dumpall && params->ntilerow==1 && params->ntilecol==1){
        DumpIncrCostFiles(incrcosts,++iincrcostfile,nflow,nrow,ncol);
      }

      /* set the tree root (equivalent to source of shortest path problem) */
      sourcelist=NULL;
      nconnectedarr=NULL;
      nsource=SelectSources(nodes,mag,ground,nflow,flows,ngroundarcs,
                            nrow,ncol,params,&sourcelist,&nconnectedarr);

      /* set up network variables for tree solver */
      SetupTreeSolveNetwork(nodes,ground,apexes,iscandidate,
                            nnoderow,nnodesperrow,narcrow,narcsperrow,
                            nrow,ncol);
      
      /* loop over sources */
      n=0;
      for(isource=0;isource<nsource;isource++){

        /* set source */
        source=sourcelist[isource];

        /* show status if verbose */
        if(source->row==GROUNDROW){
          fprintf(sp3,"Source %ld: (edge ground)\n",isource);
        }else{
          fprintf(sp3,"Source %ld: row, col = %d, %d\n",
                  isource,source->row,source->col);
        }
        
        /* run the solver, and increment nflowdone if no cycles are found */
        n+=TreeSolve(nodes,NULL,ground,source,
                     &candidatelist,&candidatebag,
                     &candidatelistsize,&candidatebagsize,
                     bkts,flows,costs,incrcosts,apexes,iscandidate,
                     ngroundarcs,nflow,mag,wrappedphase,outfiles->outfile,
                     nnoderow,nnodesperrow,narcrow,narcsperrow,nrow,ncol,
                     outfiles,nconnectedarr[isource],params);
      }

      /* free temporary memory */
      free(sourcelist);
      free(nconnectedarr);
    
      /* evaluate and save the total cost (skip if first loop through nflow) */
      fprintf(sp2,"Current solution cost: %.16g\n",
              (double )EvaluateTotalCost(costs,flows,nrow,ncol,NULL,params));
      fflush(NULL);
      if(notfirstloop){
        oldtotalcost=totalcost;
        totalcost=EvaluateTotalCost(costs,flows,nrow,ncol,NULL,params);
        if(totalcost<mintotalcost){
          mintotalcost=totalcost;
        }
        if(totalcost>oldtotalcost || (n>0 && totalcost==oldtotalcost)){
          fflush(NULL);
          fprintf(sp1,"Caution: Unexpected increase in total cost\n");
        }
        if(totalcost > mintotalcost){
          nnondecreasedcostiter++;
        }else{
          nnondecreasedcostiter=0;
        }
      }

      /* consider this flow increment done if not too many neg cycles found */
      ncycle+=n;
      if(n<=params->maxnflowcycles){
        nflowdone++;
      }else{
        nflowdone=1;
      }

      /* find maximum flow on network, excluding arcs affected by masking */
      mostflow=MaxNonMaskFlow(flows,mag,nrow,ncol);
      if(nnondecreasedcostiter>=2*mostflow){
        fflush(NULL);
        fprintf(sp0,"WARNING: No overall cost reduction for too many iterations."
                "  Breaking loop\n");
        break;
      }

      /* break if we're done with all flow increments or problem is convex */
      if(nflowdone>=params->maxflow || nflowdone>=mostflow || params->p>=1.0){
        break;
      }

      /* update flow increment */
      nflow++;
      if(nflow>params->maxflow || nflow>mostflow){
        nflow=1;
        notfirstloop=TRUE;
      }
      fprintf(sp2,"Maximum valid flow on network: %ld\n",mostflow);

      /* dump flow arrays if necessary */
      if(strlen(outfiles->flowfile)){
        FlipFlowArraySign(flows,params,nrow,ncol);
        Write2DRowColArray((void **)flows,outfiles->flowfile,nrow,ncol,
                           sizeof(short));
        FlipFlowArraySign(flows,params,nrow,ncol);
      }

    } /* end loop until no more neg cycles */
  } /* end if all pixels masked */

  /* if we have single tile, return signal handlers to default behavior */
  if(params->ntilerow==1 && params->ntilecol==1){
    signal(SIGINT,SIG_DFL);
    signal(SIGHUP,SIG_DFL);
  }

  /* free some memory */
  Free2DArray((void **)apexes,2*nrow-1);
  Free2DArray((void **)iscandidate,2*nrow-1);
  Free2DArray((void **)nodes,nrow-1);
  free(candidatebag);
  free(candidatelist);  
  free(bkts->bucketbase);

  /* grow connected component mask */
  if(strlen(outfiles->conncompfile)){
    GrowConnCompsMask(costs,flows,nrow,ncol,incrcosts,outfiles,params);
  }

  /* grow regions for tiling */
  if(params->ntilerow!=1 || params->ntilecol!=1){
    GrowRegions(costs,flows,nrow,ncol,incrcosts,outfiles,tileparams,params);
  }

  /* free some more memory */
  Free2DArray((void **)incrcosts,2*nrow-1);

  /* evaluate and display the maximum flow and total cost */
  totalcost=EvaluateTotalCost(costs,flows,nrow,ncol,NULL,params);
  fprintf(sp1,"Maximum flow on network: %ld\n",mostflow);
  fprintf(sp1,"Total solution cost: %.9g\n",(double )totalcost);

  /* integrate the wrapped phase using the solution flow */
  fprintf(sp1,"Integrating phase\n");
  unwrappedphase=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  IntegratePhase(wrappedphase,unwrappedphase,flows,nrow,ncol);

  /* reinsert the coarse estimate, if it was given */
  if(unwrappedest!=NULL){
    Add2DFloatArrays(unwrappedphase,unwrappedest,nrow,ncol);
  }

  /* flip the sign of the unwrapped phase array if it was flipped initially, */
  FlipPhaseArraySign(unwrappedphase,params,nrow,ncol);


  /* write the unwrapped output */
  fprintf(sp1,"Writing output to file %s\n",outfiles->outfile);
  WriteOutputFile(mag,unwrappedphase,outfiles->outfile,outfiles,
                  nrow,ncol);  

  /* free remaining memory and return */
  Free2DArray((void **)costs,2*nrow-1);
  Free2DArray((void **)mag,nrow);
  Free2DArray((void **)wrappedphase,nrow);
  Free2DArray((void **)unwrappedphase,nrow);
  Free2DArray((void **)flows,2*nrow-1);
  free(nnodesperrow);
  free(narcsperrow);
  return(0);

} /* end of UnwrapTile() */
