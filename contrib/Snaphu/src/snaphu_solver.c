/*************************************************************************

  snaphu network-flow solver source file
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
#include <sys/time.h>
#include <sys/resource.h>

#include "snaphu.h"


/* function: TreeSolve()
 * ---------------------
 * Solves the nonlinear network optimization problem.
 */
long TreeSolve(nodeT **nodes, nodesuppT **nodesupp, nodeT *ground, 
	       nodeT *source, candidateT **candidatelistptr, 
	       candidateT **candidatebagptr, long *candidatelistsizeptr,
	       long *candidatebagsizeptr, bucketT *bkts, short **flows, 
	       void **costs, incrcostT **incrcosts, nodeT ***apexes, 
	       signed char **iscandidate, long ngroundarcs, long nflow, 
	       float **mag, float **wrappedphase, char *outfile, 
	       long nnoderow, short *nnodesperrow, long narcrow, 
	       short *narcsperrow, long nrow, long ncol,
	       outfileT *outfiles, paramT *params){

  long i, row, col, arcrow, arccol, arcdir, arcnum, upperarcnum;
  long arcrow1, arccol1, arcdir1, arcrow2, arccol2, arcdir2;
  long treesize, candidatelistsize, candidatebagsize;
  long violation, groupcounter, fromgroup, group1, apexlistbase, apexlistlen;
  long cyclecost, outcostto, startlevel, dlevel, doutcost, dincost;
  long candidatelistlen, candidatebagnext;
  long inondegen, ipivots, nnodes, nnewnodes, maxnewnodes, templong;
  signed char fromside;
  candidateT *candidatelist, *candidatebag, *tempcandidateptr;
  nodeT *from, *to, *cycleapex, *node1, *node2, *leavingparent, *leavingchild;
  nodeT *root, *mntpt, *oldmntpt, *skipthread, *tempnode1, *tempnode2;
  nodeT *firstfromnode, *firsttonode;
  nodeT **apexlist;
  float **unwrappedphase;


  /* dereference some pointers and store as local variables */
  candidatelist=(*candidatelistptr);
  candidatebag=(*candidatebagptr);
  candidatelistsize=(*candidatelistsizeptr);
  candidatebagsize=(*candidatebagsizeptr);
  candidatelistlen=0;
  candidatebagnext=0;

  /* set up */
  bkts->curr=bkts->maxind;
  nnodes=InitTree(source,nodes,nodesupp,ground,ngroundarcs,bkts,nflow,
		  incrcosts,apexes,iscandidate,nnoderow,nnodesperrow,
		  narcrow,narcsperrow,nrow,ncol,params);
  apexlistlen=INITARRSIZE;
  apexlist=MAlloc(apexlistlen*sizeof(nodeT *));
  groupcounter=2;
  ipivots=0;
  inondegen=0;
  maxnewnodes=ceil(nnodes*params->maxnewnodeconst);
  nnewnodes=0;
  treesize=1;
  fprintf(sp3,"Treesize: %-10ld Pivots: %-11ld Improvements: %-11ld",
	  treesize,ipivots,inondegen);

  /* loop over each entering node (note, source already on tree) */
  while(treesize<nnodes){

    nnewnodes=0;
    while(nnewnodes<maxnewnodes && treesize<nnodes){

      /* get node with lowest outcost */
      to=MinOutCostNode(bkts);
      from=to->pred;

      /* add new node to the tree */
      GetArc(from,to,&arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);
      to->group=1;
      to->level=from->level+1;
      to->incost=from->incost+GetCost(incrcosts,arcrow,arccol,-arcdir);
      to->next=from->next;
      to->prev=from;
      to->next->prev=to;
      from->next=to;
    
      /* scan new node's neighbors */
      from=to;
      if(from->row!=GROUNDROW){
	arcnum=-5;
	upperarcnum=-1;
      }else{
	arcnum=-1;
	upperarcnum=ngroundarcs-1;
      }
      while(arcnum<upperarcnum){
	
	/* get row, col indices and distance of next node */
	to=NeighborNode(from,++arcnum,&upperarcnum,nodes,ground,
			&arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);
	
	/* if to node is on tree */
	if(to->group>0){
	  if(to!=from->pred){
	    cycleapex=FindApex(from,to);
	    apexes[arcrow][arccol]=cycleapex;
	    CheckArcReducedCost(from,to,cycleapex,arcrow,arccol,arcdir,nflow,
				nodes,ground,&candidatebag,&candidatebagnext,
				&candidatebagsize,incrcosts,iscandidate,
				params);
	  }else{
	    apexes[arcrow][arccol]=NULL;
	  }

	}else{

	  /* if to is not on tree, update outcost and add to bucket */
	  AddNewNode(from,to,arcdir,bkts,nflow,incrcosts,arcrow,arccol,params);
	  
	}
      }
      nnewnodes++;
      treesize++;
    }

    /* keep looping until no more arcs have negative reduced costs */
    while(candidatebagnext){

      /* if we received SIGINT or SIGHUP signal, dump results */
      /* keep this stuff out of the signal handler so we don't risk */
      /* writing a non-feasible solution (ie, if signal during augment) */
      /* signal handler disabled for all but primary (grid) networks */
      if(dumpresults_global){
	fprintf(sp0,"\n\nDumping current solution to file %s\n",
		outfile);
	if(requestedstop_global){
	  Free2DArray((void **)costs,2*nrow-1);
	}
	unwrappedphase=(float **)Get2DMem(nrow,ncol,sizeof(float *),
					  sizeof(float));
	IntegratePhase(wrappedphase,unwrappedphase,flows,nrow,ncol);
	FlipPhaseArraySign(unwrappedphase,params,nrow,ncol);	
	WriteOutputFile(mag,unwrappedphase,outfiles->outfile,outfiles,
			nrow,ncol);  
	if(requestedstop_global){
	  fprintf(sp0,"Program exiting\n");
	  exit(ABNORMAL_EXIT);
	}
	Free2DArray((void **)unwrappedphase,nrow);
	dumpresults_global=FALSE;
	fprintf(sp0,"\n\nProgram continuing\n");
      }

      /* swap candidate bag and candidate list pointers and sizes */
      tempcandidateptr=candidatebag;
      candidatebag=candidatelist;
      candidatelist=tempcandidateptr;
      templong=candidatebagsize;
      candidatebagsize=candidatelistsize;
      candidatelistsize=templong;
      candidatelistlen=candidatebagnext;
      candidatebagnext=0;

      /* sort candidate list by violation, with augmenting arcs always first */
      qsort((void *)candidatelist,candidatelistlen,sizeof(candidateT),
	    CandidateCompare);

      /* set all arc directions to be plus/minus 1 */
      for(i=0;i<candidatelistlen;i++){
	if(candidatelist[i].arcdir>1){
	  candidatelist[i].arcdir=1;
	}else if(candidatelist[i].arcdir<-1){
	  candidatelist[i].arcdir=-1;
	}
      }      

      /* this doesn't seem to make it any faster, so just do all of them */
      /* set the number of candidates to process */
      /* (must change candidatelistlen to ncandidates in for loop below) */
      /*
      maxcandidates=MAXCANDIDATES;
      if(maxcandidates>candidatelistlen){
	ncandidates=candidatelistlen;
      }else{
	ncandidates=maxcandidates;
      }
      */

      /* now pivot for each arc in the candidate list */
      for(i=0;i<candidatelistlen;i++){

	/* get arc info */
	from=candidatelist[i].from;
	to=candidatelist[i].to;
	arcdir=candidatelist[i].arcdir;
	arcrow=candidatelist[i].arcrow;
	arccol=candidatelist[i].arccol;

	/* unset iscandidate */
	iscandidate[arcrow][arccol]=FALSE;

	/* make sure the next arc still has a negative violation */
	outcostto=from->outcost+
	  GetCost(incrcosts,arcrow,arccol,arcdir);
	cyclecost=outcostto + to->incost 
	  -apexes[arcrow][arccol]->outcost
	  -apexes[arcrow][arccol]->incost;

	/* if violation no longer negative, check reverse arc */
	if(!((outcostto < to->outcost) || (cyclecost < 0))){
	  from=to;
	  to=candidatelist[i].from;
	  arcdir=-arcdir;
	  outcostto=from->outcost+
	    GetCost(incrcosts,arcrow,arccol,arcdir);
	  cyclecost=outcostto + to->incost 
	    -apexes[arcrow][arccol]->outcost
	    -apexes[arcrow][arccol]->incost;
	}

	/* see if the cycle is negative (see if there is a violation) */
	if((outcostto < to->outcost) || (cyclecost < 0)){

	  /* make sure the group counter hasn't gotten too big */
	  if(++groupcounter>MAXGROUPBASE){
	    for(row=0;row<nnoderow;row++){
	      for(col=0;col<nnodesperrow[row];col++){
		if(nodes[row][col].group>0){
		  nodes[row][col].group=1;
		}
	      }
	    }
	    if(ground!=NULL && ground->group>0){
	      ground->group=1;
	    }
	    groupcounter=2;
	  }

	  /* if augmenting cycle (nondegenerate pivot) */
	  if(cyclecost<0){

	    /* augment flow along cycle and select leaving arc */
	    /* if we are augmenting non-zero flow, any arc with zero flow */
	    /* after the augmentation is a blocking arc */
	    while(TRUE){
	      fromside=TRUE;
	      node1=from;
	      node2=to;
	      leavingchild=NULL;
	      flows[arcrow][arccol]+=arcdir*nflow;
	      ReCalcCost(costs,incrcosts,flows[arcrow][arccol],arcrow,arccol,
			 nflow,nrow,params);
	      violation=GetCost(incrcosts,arcrow,arccol,arcdir);
	      if(node1->level > node2->level){
		while(node1->level != node2->level){
		  GetArc(node1->pred,node1,&arcrow1,&arccol1,&arcdir1,
			 nrow,ncol,nodesupp);
		  flows[arcrow1][arccol1]+=(arcdir1*nflow);
		  ReCalcCost(costs,incrcosts,flows[arcrow1][arccol1],
			     arcrow1,arccol1,nflow,nrow,params);
		  if(leavingchild==NULL 
		     && !flows[arcrow1][arccol1]){
		    leavingchild=node1;
		  }
		  violation+=GetCost(incrcosts,arcrow1,arccol1,arcdir1);
		  node1->group=groupcounter+1;
		  node1=node1->pred;
		}
	      }else{
		while(node1->level != node2->level){
		  GetArc(node2->pred,node2,&arcrow2,&arccol2,&arcdir2,
			 nrow,ncol,nodesupp);
		  flows[arcrow2][arccol2]-=(arcdir2*nflow);
		  ReCalcCost(costs,incrcosts,flows[arcrow2][arccol2],
			     arcrow2,arccol2,nflow,nrow,params);
		  if(!flows[arcrow2][arccol2]){
		    leavingchild=node2;
		    fromside=FALSE;
		  }
		  violation+=GetCost(incrcosts,arcrow2,arccol2,-arcdir2);
		  node2->group=groupcounter;
		  node2=node2->pred;
		}
	      }
	      while(node1!=node2){
		GetArc(node1->pred,node1,&arcrow1,&arccol1,&arcdir1,nrow,ncol,
		       nodesupp);
		GetArc(node2->pred,node2,&arcrow2,&arccol2,&arcdir2,nrow,ncol,
		       nodesupp);
		flows[arcrow1][arccol1]+=(arcdir1*nflow);
		flows[arcrow2][arccol2]-=(arcdir2*nflow);
		ReCalcCost(costs,incrcosts,flows[arcrow1][arccol1],
			   arcrow1,arccol1,nflow,nrow,params);
		ReCalcCost(costs,incrcosts,flows[arcrow2][arccol2],
			   arcrow2,arccol2,nflow,nrow,params);
		violation+=(GetCost(incrcosts,arcrow1,arccol1,arcdir1)
			    +GetCost(incrcosts,arcrow2,arccol2,-arcdir2));
		if(!flows[arcrow2][arccol2]){
		  leavingchild=node2;
		  fromside=FALSE;
		}else if(leavingchild==NULL 
			 && !flows[arcrow1][arccol1]){
		  leavingchild=node1;
		}
		node1->group=groupcounter+1;
		node2->group=groupcounter;
		node1=node1->pred;
		node2=node2->pred;
	      }
	      if(violation>=0){
		break;
	      }
	    }
	    inondegen++;

	  }else{

	    /* We are not augmenting flow, but just updating potentials. */
	    /* Arcs with zero flow are implicitly directed upwards to */
	    /* maintain a strongly feasible spanning tree, so arcs with zero */
	    /* flow on the path between to node and apex are blocking arcs. */
	    /* Leaving arc is last one whose child's new outcost is less */
	    /* than its old outcost.  Such an arc must exist, or else */
	    /* we'd be augmenting flow on a negative cycle. */
	    
	    /* trace the cycle and select leaving arc */
	    fromside=FALSE;
	    node1=from;
	    node2=to;
	    leavingchild=NULL;
	    if(node1->level > node2->level){
	      while(node1->level != node2->level){
		node1->group=groupcounter+1;
		node1=node1->pred;
	      }
	    }else{
	      while(node1->level != node2->level){
		if(outcostto < node2->outcost){
		  leavingchild=node2;
		  GetArc(node2->pred,node2,&arcrow2,&arccol2,&arcdir2,
			 nrow,ncol,nodesupp);
		  outcostto+=GetCost(incrcosts,arcrow2,arccol2,-arcdir2);
		}else{
		  outcostto=VERYFAR;
		}
		node2->group=groupcounter;
		node2=node2->pred;
	      }
	    }
	    while(node1!=node2){
	      if(outcostto < node2->outcost){
		leavingchild=node2;
		GetArc(node2->pred,node2,&arcrow2,&arccol2,&arcdir2,nrow,ncol,
		       nodesupp);
		outcostto+=GetCost(incrcosts,arcrow2,arccol2,-arcdir2);
	      }else{
		outcostto=VERYFAR;
	      }
	      node1->group=groupcounter+1;
	      node2->group=groupcounter;
	      node1=node1->pred;
	      node2=node2->pred;
	    }
	  }
	  cycleapex=node1;

          /* set leaving parent */ 
          if(leavingchild==NULL){
	    fromside=TRUE;
	    leavingparent=from;
	  }else{
	    leavingparent=leavingchild->pred;
	  }

          /* swap from and to if leaving arc is on the from side */
	  if(fromside){
	    groupcounter++;
	    fromgroup=groupcounter-1;
	    tempnode1=from;
	    from=to;
	    to=tempnode1;
	  }else{
	    fromgroup=groupcounter+1;
	  }

	  /* if augmenting pivot */
	  if(cyclecost<0){

	    /* find first child of apex on either cycle path */
	    firstfromnode=NULL;
	    firsttonode=NULL;
	    if(cycleapex->row!=GROUNDROW){
	      arcnum=-5;
	      upperarcnum=-1;
	    }else{
	      arcnum=-1;
	      upperarcnum=ngroundarcs-1;
	    }
	    while(arcnum<upperarcnum){
	      tempnode1=NeighborNode(cycleapex,++arcnum,&upperarcnum,nodes,
				     ground,&arcrow,&arccol,&arcdir,nrow,ncol,
				     nodesupp);
	      if(tempnode1->group==groupcounter
		 && apexes[arcrow][arccol]==NULL){
		firsttonode=tempnode1;
		if(firstfromnode!=NULL){
		  break;
		}
	      }else if(tempnode1->group==fromgroup 
		       && apexes[arcrow][arccol]==NULL){
		firstfromnode=tempnode1;
		if(firsttonode!=NULL){
		  break;
		}
	      }
	    }

	    /* update potentials, mark stationary parts of tree */
	    cycleapex->group=groupcounter+2;
	    if(firsttonode!=NULL){
	      NonDegenUpdateChildren(cycleapex,leavingparent,firsttonode,0,
				     ngroundarcs,nflow,nodes,nodesupp,ground,
				     apexes,incrcosts,nrow,ncol,params); 
	    }
	    if(firstfromnode!=NULL){
	      NonDegenUpdateChildren(cycleapex,from,firstfromnode,1,
				     ngroundarcs,nflow,nodes,nodesupp,ground,
				     apexes,incrcosts,nrow,ncol,params);
	    }
	    groupcounter=from->group;
	    apexlistbase=cycleapex->group;

	    /* children of cycleapex are not marked, so we set fromgroup */
	    /*   equal to cycleapex group for use with apex updates below */
	    /* all other children of cycle will be in apexlist if we had an */
	    /*   augmenting pivot, so fromgroup only important for cycleapex */
	    fromgroup=cycleapex->group;

	  }else{

	    /* set this stuff for use with apex updates below */
	    cycleapex->group=fromgroup;
	    groupcounter+=2;
	    apexlistbase=groupcounter+1;
	  }

	  /* remount subtree at new mount point */
	  if(leavingchild==NULL){
	    
	    skipthread=to;

	  }else{

	    root=from;
	    oldmntpt=to;

	    /* for each node on the path from to node to leaving child */
	    while(oldmntpt!=leavingparent){

	      /* remount the subtree at the new mount point */
	      mntpt=root;
	      root=oldmntpt;
	      oldmntpt=root->pred;
	      root->pred=mntpt;
	      GetArc(mntpt,root,&arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);
	      
	      /* calculate differences for updating potentials and levels */
	      dlevel=mntpt->level-root->level+1;
	      doutcost=mntpt->outcost - root->outcost 
		+ GetCost(incrcosts,arcrow,arccol,arcdir);
	      dincost=mntpt->incost - root->incost 
		+ GetCost(incrcosts,arcrow,arccol,-arcdir);

	      /* update all children */
	      /* group of each remounted tree used to reset apexes below */
	      node1=root;
	      startlevel=root->level;
	      groupcounter++;
	      while(TRUE){
		
		/* update the level, potentials, and group of the node */
		node1->level+=dlevel;
		node1->outcost+=doutcost;
		node1->incost+=dincost;
		node1->group=groupcounter;
		
		/* break when node1 is no longer descendent of the root */
		if(node1->next->level <= startlevel){
		  break;
		}
		node1=node1->next;
	      }

	      /* update threads */
	      root->prev->next=node1->next;
	      node1->next->prev=root->prev;
	      node1->next=mntpt->next;  
	      mntpt->next->prev=node1;
	      mntpt->next=root;       
	      root->prev=mntpt;

	    }
	    skipthread=node1->next;

	    /* reset apex pointers for entering and leaving arcs */
	    GetArc(from,to,&arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);
	    apexes[arcrow][arccol]=NULL;
	    GetArc(leavingparent,leavingchild,&arcrow,&arccol,
		   &arcdir,nrow,ncol,nodesupp);
	    apexes[arcrow][arccol]=cycleapex;

	    /* make sure we have enough memory for the apex list */
	    if(groupcounter-apexlistbase+1>apexlistlen){
	      apexlistlen=1.5*(groupcounter-apexlistbase+1); 
	      apexlist=ReAlloc(apexlist,apexlistlen*sizeof(nodeT *));
	    }
        
	    /* set the apex list */
	    node2=leavingchild;
	    for(group1=groupcounter;group1>=apexlistbase;group1--){
	      apexlist[group1-apexlistbase]=node2;
	      node2=node2->pred;
	    }
        
	    /* reset apex pointers on remounted tree */
	    /* only nodes which are in different groups need new apexes */
	    node1=to;
	    startlevel=to->level;
	    while(TRUE){
	      
	      /* loop over outgoing arcs */
	      if(node1->row!=GROUNDROW){
		arcnum=-5;
		upperarcnum=-1;
	      }else{
		arcnum=-1;
		upperarcnum=ngroundarcs-1;
	      }
	      while(arcnum<upperarcnum){
		node2=NeighborNode(node1,++arcnum,&upperarcnum,nodes,ground,
				   &arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);

		/* if node2 on tree */
		if(node2->group>0){

		
		  /* if node2 is either not part of remounted tree or */
		  /*   it is higher on remounted tree than node1, */
		  /*   and arc isn't already on tree */
		  if(node2->group < node1->group 
		     && apexes[arcrow][arccol]!=NULL){

		    /* if new apex in apexlist */
		    /* node2 on remounted tree, if nonaugmenting pivot */
		    if(node2->group >= apexlistbase){
		      
		      apexes[arcrow][arccol]=apexlist[node2->group
						     -apexlistbase];
		      
		    }else{
			
		      /* if old apex below level of cycleapex, */
		      /*   node2 is on "to" node's side of tree */
		      /* implicitly, if old apex above cycleapex, */
		      /*   we do nothing since apex won't change */
		      if(apexes[arcrow][arccol]->level > cycleapex->level){
		      
			/* since new apex not in apexlist (tested above), */
			/* node2 above leaving arc so new apex is cycleapex */
			apexes[arcrow][arccol]=cycleapex;
			
		      }else{

			/* node2 not on "to" side of tree */
			/* if old apex is cycleapex, node2 is on "from" side */
			if(apexes[arcrow][arccol]==cycleapex){

			  /* new apex will be on cycle, so trace node2->pred */
			  /*   until we hit a node with group==fromgroup */
			  tempnode2=node2;
			  while(tempnode2->group != fromgroup){
			    tempnode2=tempnode2->pred;
			  }
                          apexes[arcrow][arccol]=tempnode2;

			}
		      }
		    }

		    /* check outgoing arcs for negative reduced costs */
                    CheckArcReducedCost(node1,node2,apexes[arcrow][arccol],
					arcrow,arccol,arcdir,nflow,nodes,
					ground,&candidatebag,
					&candidatebagnext,&candidatebagsize,
					incrcosts,iscandidate,params);

		  } /* end if node2 below node1 and arc not on tree */

		}else{

		  /* node2 is not on tree, so put it in correct bucket */
		  AddNewNode(node1,node2,arcdir,bkts,nflow,incrcosts,
			     arcrow,arccol,params);

		} /* end if node2 on tree */
	      } /* end loop over node1 outgoing arcs */


	      /* move to next node in thread, break if we left the subtree */
	      node1=node1->next;
	      if(node1->level <= startlevel){
		break;
	      }
	    }
	  } /* end if leavingchild!=NULL */

	  /* if we had an augmenting cycle */
	  /* we need to check outarcs from descendents of any cycle node */
	  /* (except apex, since apex potentials don't change) */
	  if(cyclecost<0){
	    
	    /* check descendents of cycle children of apex */
	    while(TRUE){
	      
	      /* firstfromnode, firsttonode may have changed */
	      if(firstfromnode!=NULL && firstfromnode->pred==cycleapex){
		node1=firstfromnode;
		firstfromnode=NULL;
	      }else if(firsttonode!=NULL && firsttonode->pred==cycleapex){
		node1=firsttonode;
		firsttonode=NULL;
	      }else{
		break;
	      }
	      startlevel=node1->level;

	      /* loop over all descendents */
	      while(TRUE){
	      
		/* loop over outgoing arcs */
		if(node1->row!=GROUNDROW){
		  arcnum=-5;
		  upperarcnum=-1;
		}else{
		  arcnum=-1;
		  upperarcnum=ngroundarcs-1;
		}
		while(arcnum<upperarcnum){
		  node2=NeighborNode(node1,++arcnum,&upperarcnum,nodes,ground,
				     &arcrow,&arccol,&arcdir,nrow,ncol,
				     nodesupp);

		  /* check for outcost updates or negative reduced costs */
		  if(node2->group>0){
		    if(apexes[arcrow][arccol]!=NULL 
		       && (node2->group!=node1->group 
			   || node1->group==apexlistbase)){
		      CheckArcReducedCost(node1,node2,apexes[arcrow][arccol],
					  arcrow,arccol,arcdir,nflow,nodes,
					  ground,&candidatebag,
					  &candidatebagnext,&candidatebagsize,
					  incrcosts,iscandidate,params);
		    }
		  }else{
		    AddNewNode(node1,node2,arcdir,bkts,nflow,incrcosts,
			       arcrow,arccol,params);
		  }			
		}
		
		/* move to next node in thread, break if left the subtree */
		/*   but skip the remounted tree, since we checked it above */
		node1=node1->next;
		if(node1==to){
		  node1=skipthread;
		}
		if(node1->level <= startlevel){
		  break;
		}
	      }
	    }
	  }
	  ipivots++;
	} /* end if cyclecost<0 || outcostto<to->outcost */
      } /* end of for loop over candidates in list */

      /* this is needed only if we don't process all candidates above */
      /* copy remaining candidates into candidatebag */
      /*
      while(candidatebagnext+(candidatelistlen-ncandidates)>candidatebagsize){
	candidatebagsize+=CANDIDATEBAGSTEP;
	candidatebag=ReAlloc(candidatebag,candidatebagsize*sizeof(candidateT));
      }
      for(i=ncandidates;i<candidatelistlen;i++){
	candidatebag[candidatebagnext++]=candidatelist[i];
      }
      */

      /* display status */
      fprintf(sp3,"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
	      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
	      "\b\b\b\b\b\b"
	      "Treesize: %-10ld Pivots: %-11ld Improvements: %-11ld",
	      treesize,ipivots,inondegen);
      fflush(sp3);

    } /* end of while loop on candidatebagnext */    
  } /* end while treesize<number of total nodes */

  
  /* clean up: set pointers for outputs */
  fprintf(sp3,"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
	  "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
	  "\b\b\b\b\b\b"
	  "Treesize: %-10ld Pivots: %-11ld Improvements: %-11ld\n",
	  treesize,ipivots,inondegen);
  fflush(sp3);
  *candidatelistptr=candidatelist;
  *candidatebagptr=candidatebag;
  *candidatelistsizeptr=candidatelistsize;
  *candidatebagsizeptr=candidatebagsize;
  free(apexlist);

  /* return the number of nondegenerate pivots (number of improvements) */
  return(inondegen);

}


/* function: AddNewNode()
 * ----------------------
 * Adds a node to a bucket if it is not already in a bucket.  Updates 
 * outcosts of to node if the new distance is less or if to's pred is
 * from (then we have to do the update).
 */
void AddNewNode(nodeT *from, nodeT *to, long arcdir, bucketT *bkts, 
		long nflow, incrcostT **incrcosts, long arcrow, long arccol, 
		paramT *params){
  
  long newoutcost;

  newoutcost=from->outcost
    +GetCost(incrcosts,arcrow,arccol,arcdir);
  if(newoutcost<to->outcost || to->pred==from){
    if(to->group==-1){      /* if to is already in a bucket */
      if(to->outcost<bkts->maxind){
	if(to->outcost>bkts->minind){
	  BucketRemove(to,to->outcost,bkts);
	}else{
	  BucketRemove(to,bkts->minind,bkts);
	}
      }else{
	BucketRemove(to,bkts->maxind,bkts);
      }
    }      
    to->outcost=newoutcost;
    to->pred=from;
    if(newoutcost<bkts->maxind){
      if(newoutcost>bkts->minind){
	BucketInsert(to,newoutcost,bkts);
	if(newoutcost<bkts->curr){
	  bkts->curr=newoutcost;
	}
      }else{
	BucketInsert(to,bkts->minind,bkts);
	bkts->curr=bkts->minind;
      }
    }else{
      BucketInsert(to,bkts->maxind,bkts);
    }
    to->group=-1;
  }	  
}


/* function: CheckArcReducedCost()
 * -------------------------------
 * Given a from and to node, checks for negative reduced cost, and adds
 * the arc to the entering arc candidate bag if one is found.
 */
void CheckArcReducedCost(nodeT *from, nodeT *to, nodeT *apex, 
			 long arcrow, long arccol, long arcdir, 
			 long nflow, nodeT **nodes, nodeT *ground, 
			 candidateT **candidatebagptr, 
			 long *candidatebagnextptr, 
			 long *candidatebagsizeptr, incrcostT **incrcosts, 
			 signed char **iscandidate, paramT *params){

  long apexcost, fwdarcdist, revarcdist, violation;
  nodeT *temp;
  
  /* do nothing if already candidate */
  /* illegal corner arcs have iscandidate=TRUE set ahead of time */
  if(iscandidate[arcrow][arccol]){
    return;
  }

  /* set the apex cost */
  apexcost=apex->outcost+apex->incost;

  /* check forward arc */
  fwdarcdist=GetCost(incrcosts,arcrow,arccol,arcdir);
  violation=fwdarcdist+from->outcost+to->incost-apexcost;
  if(violation<0){
    arcdir*=2;  /* magnitude 2 for sorting */
  }else{
    revarcdist=GetCost(incrcosts,arcrow,arccol,-arcdir);
    violation=revarcdist+to->outcost+from->incost-apexcost;
    if(violation<0){
      arcdir*=-2;  /* magnitude 2 for sorting */
      temp=from;
      from=to;
      to=temp;
    }else{
      violation=fwdarcdist+from->outcost-to->outcost;
      if(violation>=0){
	violation=revarcdist+to->outcost-from->outcost;
	if(violation<0){
	  arcdir=-arcdir;
	  temp=from;
	  from=to;
	  to=temp;
	}
      }
    }
  }

  /* see if we have a violation, and if so, add arc to candidate bag */
  if(violation<0){
    if((*candidatebagnextptr)>=(*candidatebagsizeptr)){
      (*candidatebagsizeptr)+=CANDIDATEBAGSTEP;
      (*candidatebagptr)=ReAlloc(*candidatebagptr,
				 (*candidatebagsizeptr)*sizeof(candidateT));
    }
    (*candidatebagptr)[*candidatebagnextptr].violation=violation;
    (*candidatebagptr)[*candidatebagnextptr].from=from;
    (*candidatebagptr)[*candidatebagnextptr].to=to;
    (*candidatebagptr)[*candidatebagnextptr].arcrow=arcrow;
    (*candidatebagptr)[*candidatebagnextptr].arccol=arccol;
    (*candidatebagptr)[*candidatebagnextptr].arcdir=arcdir;
    (*candidatebagnextptr)++;
    iscandidate[arcrow][arccol]=TRUE;
  }

}


/* function: InitTree()
 * --------------------
 */
long InitTree(nodeT *source, nodeT **nodes, nodesuppT **nodesupp, 
	      nodeT *ground, long ngroundarcs, bucketT *bkts, long nflow, 
	      incrcostT **incrcosts, nodeT ***apexes, 
	      signed char **iscandidate, long nnoderow, short *nnodesperrow, 
	      long narcrow, short *narcsperrow, long nrow, long ncol, 
	      paramT *params){

  long row, col, arcnum, upperarcnum, arcrow, arccol, arcdir, nnodes;
  nodeT *to;


  /* loop over each node and initialize values */
  nnodes=0;
  for(row=0;row<nnoderow;row++){
    for(col=0;col<nnodesperrow[row];col++){
      nodes[row][col].group=0;
      nodes[row][col].outcost=VERYFAR;
      nodes[row][col].pred=NULL;
      nnodes++;
    }
  }

  /* initialize the ground node */
  if(ground!=NULL){
    ground->group=0;
    ground->outcost=VERYFAR;
    ground->pred=NULL;
    nnodes++;
  }

  /* initialize arcs */
  for(row=0;row<narcrow;row++){
    for(col=0;col<narcsperrow[row];col++){
      apexes[row][col]=NONTREEARC;
      iscandidate[row][col]=FALSE;
    }
  }

  /* if in grid mode, ground will exist */
  if(ground!=NULL){
    
    /* set iscandidate=TRUE for illegal corner arcs so they're never used */
    iscandidate[nrow-1][0]=TRUE;
    iscandidate[2*nrow-2][0]=TRUE;
    iscandidate[nrow-1][ncol-2]=TRUE;
    iscandidate[2*nrow-2][ncol-2]=TRUE;

  }

  /* put source on tree */
  source->group=1;
  source->outcost=0;
  source->incost=0;
  source->pred=NULL;
  source->prev=source;
  source->next=source;
  source->level=0;

  /* loop over outgoing arcs and add to buckets */
  if(source->row!=GROUNDROW){
    arcnum=-5;
    upperarcnum=-1;
  }else{
    arcnum=-1;
    upperarcnum=ngroundarcs-1;
  }
  while(arcnum<upperarcnum){
    
    /* get node reached by outgoing arc */
    to=NeighborNode(source,++arcnum,&upperarcnum,nodes,ground,
		    &arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);

    /* add to node to bucket */
    AddNewNode(source,to,arcdir,bkts,nflow,incrcosts,arcrow,arccol,params);

  }

  /* return the number of nodes in the network */
  return(nnodes);
      
}


/* function: FindApex()
 * --------------------
 * Given pointers to two nodes on a spanning tree, the function finds
 * and returns a pointer to their deepest common ancestor, the apex of
 * a cycle formed by joining the two nodes with an arc.
 */
nodeT *FindApex(nodeT *from, nodeT *to){

  if(from->level > to->level){
    while(from->level != to->level){
      from=from->pred;
    }
  }else{
    while(from->level != to->level){
      to=to->pred;
    }
  }
  while(from != to){
    from=from->pred;
    to=to->pred;
  }
  return(from);
}


/* function: CandidateCompare()
 * ----------------------------
 * Compares the violations of candidate arcs for sorting.  First checks
 * if either candidate has an arcdir magnitude greater than 1, denoting 
 * an augmenting cycle.  Augmenting candidates are always placed before 
 * non-augmenting candidates.  Otherwise, returns positive if the first  
 * candidate has a greater (less negative) violation than the second, 0 
 * if they are the same, and negative otherwise.  
 */
int CandidateCompare(const void *c1, const void *c2){

  if(labs(((candidateT *)c1)->arcdir) > 1){
    if(labs(((candidateT *)c2)->arcdir) < 2){
      return(-1);
    }
  }else if(labs(((candidateT *)c2)->arcdir) > 1){
    return(1);
  }

  return(((candidateT *)c1)->violation - ((candidateT *)c2)->violation);

  /*
  if(((candidateT *)c1)->violation > ((candidateT *)c2)->violation){
    return(1);
  }else if(((candidateT *)c1)->violation < ((candidateT *)c2)->violation){
    return(-1);
  }else{
    return(0);
  }
  */
}


/* function: NeighborNodeGrid()
 * ----------------------------
 * Return the neighboring node of the given node corresponding to the
 * given arc number for a grid network with a ground node.
 */
nodeT *NeighborNodeGrid(nodeT *node1, long arcnum, long *upperarcnumptr,
			nodeT **nodes, nodeT *ground, long *arcrowptr, 
			long *arccolptr, long *arcdirptr, long nrow, 
			long ncol, nodesuppT **nodesupp){
  long row, col;

  row=node1->row;
  col=node1->col;

  switch(arcnum){
  case -4:
    *arcrowptr=row;
    *arccolptr=col+1;
    *arcdirptr=1;
    if(col==ncol-2){
      return(ground);
    }else{
      return(&nodes[row][col+1]);
    }
    break;
  case -3:
    *arcrowptr=nrow+row;
    *arccolptr=col;
    *arcdirptr=1;
    if(row==nrow-2){
      return(ground);
    }else{
      return(&nodes[row+1][col]);
    }
    break;
  case -2:
    *arcrowptr=row;
    *arccolptr=col;
    *arcdirptr=-1;
    if(col==0){
      return(ground);
    }else{
      return(&nodes[row][col-1]);
    }
    break;
  case -1:
    *arcrowptr=nrow-1+row;
    *arccolptr=col;
    *arcdirptr=-1;
    if(row==0){
      return(ground);
    }else{
      return(&nodes[row-1][col]);
    }
    break;
  default:
    if(arcnum<nrow-1){
      *arcrowptr=arcnum;
      *arccolptr=0;
      *arcdirptr=1;
      return(&nodes[*arcrowptr][0]);
    }else if(arcnum<2*(nrow-1)){           
      *arcrowptr=arcnum-(nrow-1);
      *arccolptr=ncol-1;
      *arcdirptr=-1;
      return(&nodes[*arcrowptr][ncol-2]);
    }else if(arcnum<2*(nrow-1)+ncol-3){   
      *arcrowptr=nrow-1;
      *arccolptr=arcnum-2*(nrow-1)+1;    
      *arcdirptr=1;
      return(&nodes[0][*arccolptr]);
    }else{
      *arcrowptr=2*nrow-2;
      *arccolptr=arcnum-(2*(nrow-1)+ncol-3)+1;
      *arcdirptr=-1;
      return(&nodes[nrow-2][*arccolptr]);
    }
    break;
  }

}


/* function: NeighborNodeNonGrid()
 * -------------------------------
 * Return the neighboring node of the given node corresponding to the
 * given arc number for a nongrid network (ie, arbitrary topology).
 */
nodeT *NeighborNodeNonGrid(nodeT *node1, long arcnum, long *upperarcnumptr,
			   nodeT **nodes, nodeT *ground, long *arcrowptr, 
			   long *arccolptr, long *arcdirptr, long nrow, 
			   long ncol, nodesuppT **nodesupp){

  long tilenum, nodenum;
  scndryarcT *outarc;

  /* set up */
  tilenum=node1->row;
  nodenum=node1->col;
  *upperarcnumptr=nodesupp[tilenum][nodenum].noutarcs-5;
  
  /* set the arc row (tilenumber) and column (arcnumber) */
  outarc=nodesupp[tilenum][nodenum].outarcs[arcnum+4];
  *arcrowptr=outarc->arcrow;
  *arccolptr=outarc->arccol;
  if(node1==outarc->from){
    *arcdirptr=1;
  }else{
    *arcdirptr=-1;
  }

  /* return the neighbor node */
  return(nodesupp[tilenum][nodenum].neighbornodes[arcnum+4]);

}


/* function: GetArcGrid()
 * ----------------------
 * Given a from node and a to node, sets pointers for indices into
 * arc arrays, assuming primary (grid) network.
 */
void GetArcGrid(nodeT *from, nodeT *to, long *arcrow, long *arccol, 
		long *arcdir, long nrow, long ncol, nodesuppT **nodesupp){

  long fromrow, fromcol, torow, tocol;

  fromrow=from->row;
  fromcol=from->col;
  torow=to->row;
  tocol=to->col;
  
  if(fromcol==tocol-1){           /* normal arcs (neither endpoint ground) */
    *arcrow=fromrow;
    *arccol=fromcol+1;
    *arcdir=1;
  }else if(fromcol==tocol+1){
    *arcrow=fromrow;
    *arccol=fromcol;
    *arcdir=-1;
  }else if(fromrow==torow-1){
    *arcrow=fromrow+1+nrow-1;
    *arccol=fromcol;
    *arcdir=1;
  }else if(fromrow==torow+1){
    *arcrow=fromrow+nrow-1;
    *arccol=fromcol;
    *arcdir=-1;
  }else if(fromcol==0){           /* arcs to ground */
    *arcrow=fromrow;
    *arccol=0;
    *arcdir=-1;
  }else if(fromcol==ncol-2){
    *arcrow=fromrow;
    *arccol=ncol-1;
    *arcdir=1;
  }else if(fromrow==0){
    *arcrow=nrow-1;
    *arccol=fromcol;
    *arcdir=-1;
  }else if(fromrow==nrow-2){
    *arcrow=2*(nrow-1);
    *arccol=fromcol;
    *arcdir=1;
  }else if(tocol==0){             /* arcs from ground */
    *arcrow=torow;
    *arccol=0;
    *arcdir=1;
  }else if(tocol==ncol-2){
    *arcrow=torow;
    *arccol=ncol-1;
    *arcdir=-1;
  }else if(torow==0){
    *arcrow=nrow-1;
    *arccol=tocol;
    *arcdir=1;
  }else{
    *arcrow=2*(nrow-1);
    *arccol=tocol;
    *arcdir=-1;
  }

}


/* function: GetArcNonGrid()
 * -------------------------
 * Given a from node and a to node, sets pointers for indices into
 * arc arrays, assuming secondary (arbitrary topology) network.
 */
void GetArcNonGrid(nodeT *from, nodeT *to, long *arcrow, long *arccol, 
		   long *arcdir, long nrow, long ncol, nodesuppT **nodesupp){

  long tilenum, nodenum, arcnum;
  scndryarcT *outarc;

  /* get tile and node numbers for from node */
  tilenum=from->row;
  nodenum=from->col;

  /* loop over all outgoing arcs of from node */
  arcnum=0;
  while(TRUE){
    outarc=nodesupp[tilenum][nodenum].outarcs[arcnum++];
    if(outarc->from==to){
      *arcrow=outarc->arcrow;
      *arccol=outarc->arccol;
      *arcdir=-1;
      return;
    }else if(outarc->to==to){
      *arcrow=outarc->arcrow;
      *arccol=outarc->arccol;
      *arcdir=1;
      return;
    }
  }
}


/* Function: NonDegenUpdateChildren()
 * ----------------------------------
 * Updates potentials and groups of all childredn along an augmenting path, 
 * until a stop node is hit.
 */
void NonDegenUpdateChildren(nodeT *startnode, nodeT *lastnode, 
			    nodeT *nextonpath, long dgroup, 
			    long ngroundarcs, long nflow, nodeT **nodes,
			    nodesuppT **nodesupp, nodeT *ground, 
			    nodeT ***apexes, incrcostT **incrcosts, 
			    long nrow, long ncol, paramT *params){

  nodeT *node1, *node2;
  long dincost, doutcost, arcnum, upperarcnum, startlevel;
  long group1, pathgroup, arcrow, arccol, arcdir;

  /* loop along flow path  */
  node1=startnode;
  pathgroup=lastnode->group;
  while(node1!=lastnode){

    /* update potentials along the flow path by calculating arc distances */
    node2=nextonpath;
    GetArc(node2->pred,node2,&arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);
    doutcost=node1->outcost - node2->outcost
      + GetCost(incrcosts,arcrow,arccol,arcdir);
    node2->outcost+=doutcost;
    dincost=node1->incost - node2->incost
      + GetCost(incrcosts,arcrow,arccol,-arcdir);
    node2->incost+=dincost;
    node2->group=node1->group+dgroup;

    /* update potentials of children of this node in the flow path */
    node1=node2;
    if(node1->row!=GROUNDROW){
      arcnum=-5;
      upperarcnum=-1;
    }else{
      arcnum=-1;
      upperarcnum=ngroundarcs-1;
    }
    while(arcnum<upperarcnum){
      node2=NeighborNode(node1,++arcnum,&upperarcnum,nodes,ground,
                         &arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);
      if(node2->pred==node1 && node2->group>0){
        if(node2->group==pathgroup){
          nextonpath=node2;
        }else{
          startlevel=node2->level;
          group1=node1->group;
          while(TRUE){
            node2->group=group1;
            node2->incost+=dincost;
            node2->outcost+=doutcost;
            node2=node2->next;
            if(node2->level <= startlevel){
              break;
            }
          }
        }
      }
    }
  }
}


/* function: InitNetowrk()
 * -----------------------
 */
void InitNetwork(short **flows, long *ngroundarcsptr, long *ncycleptr, 
		 long *nflowdoneptr, long *mostflowptr, long *nflowptr, 
		 long *candidatebagsizeptr, candidateT **candidatebagptr, 
		 long *candidatelistsizeptr, candidateT **candidatelistptr, 
		 signed char ***iscandidateptr, nodeT ****apexesptr, 
		 bucketT **bktsptr, long *iincrcostfileptr, 
		 incrcostT ***incrcostsptr, nodeT ***nodesptr, nodeT *ground, 
		 long *nnoderowptr, short **nnodesperrowptr, long *narcrowptr,
		 short **narcsperrowptr, long nrow, long ncol, 
		 signed char *notfirstloopptr, totalcostT *totalcostptr,
		 paramT *params){

  long i;


  /* get and initialize memory for nodes */
  if(ground!=NULL && *nodesptr==NULL){
    *nodesptr=(nodeT **)Get2DMem(nrow-1,ncol-1,sizeof(nodeT *),sizeof(nodeT));
    InitNodeNums(nrow-1,ncol-1,*nodesptr,ground);
  }

  /* take care of ambiguous flows to ground at corners */
  if(ground!=NULL){
    flows[0][0]+=flows[nrow-1][0];
    flows[nrow-1][0]=0;
    flows[0][ncol-1]-=flows[nrow-1][ncol-2];
    flows[nrow-1][ncol-2]=0;
    flows[nrow-2][0]-=flows[2*nrow-2][0];
    flows[2*nrow-2][0]=0;
    flows[nrow-2][ncol-1]+=flows[2*nrow-2][ncol-2];
    flows[2*nrow-2][ncol-2]=0;
  }

  /* initialize network solver variables */
  *ncycleptr=0;
  *nflowptr=1;
  *candidatebagsizeptr=INITARRSIZE;
  *candidatebagptr=MAlloc(*candidatebagsizeptr*sizeof(candidateT));
  *candidatelistsizeptr=INITARRSIZE;
  *candidatelistptr=MAlloc(*candidatelistsizeptr*sizeof(candidateT));
  if(ground!=NULL){
    *nflowdoneptr=0;
    *mostflowptr=Short2DRowColAbsMax(flows,nrow,ncol);
    if(*mostflowptr*params->nshortcycle>LARGESHORT){
      fprintf(sp1,"Maximum flow on network: %ld\n",*mostflowptr);
      fprintf(sp0,"((Maximum flow) * NSHORTCYCLE) too large\nAbort\n");
      exit(ABNORMAL_EXIT);
    }
    if(ncol>2){
      *ngroundarcsptr=2*(nrow+ncol-2)-4; /* don't include corner column arcs */
    }else{
      *ngroundarcsptr=2*(nrow+ncol-2)-2;
    }
    *iscandidateptr=(signed char **)Get2DRowColMem(nrow,ncol,
						   sizeof(signed char *),
						   sizeof(signed char));
    *apexesptr=(nodeT ***)Get2DRowColMem(nrow,ncol,sizeof(nodeT **),
					 sizeof(nodeT *));
  }

  /* set up buckets for TreeSolve (MSTInitFlows() has local set of buckets) */
  *bktsptr=MAlloc(sizeof(bucketT));
  if(ground!=NULL){
    (*bktsptr)->minind=-LRound((params->maxcost+1)*(nrow+ncol)
			       *NEGBUCKETFRACTION);
    (*bktsptr)->maxind=LRound((params->maxcost+1)*(nrow+ncol)
			      *POSBUCKETFRACTION);
  }else{
    (*bktsptr)->minind=-LRound((params->maxcost+1)*(nrow)
			       *NEGBUCKETFRACTION);
    (*bktsptr)->maxind=LRound((params->maxcost+1)*(nrow)
			      *POSBUCKETFRACTION);
  }
  (*bktsptr)->size=(*bktsptr)->maxind-(*bktsptr)->minind+1;
  (*bktsptr)->bucketbase=(nodeT **)MAlloc((*bktsptr)->size*sizeof(nodeT *));
  (*bktsptr)->bucket=&((*bktsptr)->bucketbase[-(*bktsptr)->minind]);
  for(i=0;i<(*bktsptr)->size;i++){
    (*bktsptr)->bucketbase[i]=NULL;
  }

  /* get memory for incremental cost arrays */
  *iincrcostfileptr=0;
  if(ground!=NULL){
    (*incrcostsptr)=(incrcostT **)Get2DRowColMem(nrow,ncol,sizeof(incrcostT *),
						 sizeof(incrcostT));
  }

  /* set number of nodes and arcs per row */
  if(ground!=NULL){
    (*nnoderowptr)=nrow-1;
    (*nnodesperrowptr)=(short *)MAlloc((nrow-1)*sizeof(short));
    for(i=0;i<nrow-1;i++){
      (*nnodesperrowptr)[i]=ncol-1;
    }
    (*narcrowptr)=2*nrow-1;
    (*narcsperrowptr)=(short *)MAlloc((2*nrow-1)*sizeof(short));
    for(i=0;i<nrow-1;i++){
      (*narcsperrowptr)[i]=ncol;
    }
    for(i=nrow-1;i<2*nrow-1;i++){
      (*narcsperrowptr)[i]=ncol-1;
    }
  }

  /* initialize variables for main optimizer loop */
  (*notfirstloopptr)=FALSE;
  (*totalcostptr)=INITTOTALCOST;
}


/* function: InitNodeNums()
 * ------------------------
 */
void InitNodeNums(long nrow, long ncol, nodeT **nodes, nodeT *ground){

  long row, col;

  /* loop over each element and initialize values */
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      nodes[row][col].row=row;
      nodes[row][col].col=col;
    }
  }

  /* initialize the ground node */
  if(ground!=NULL){
    ground->row=GROUNDROW;
    ground->col=GROUNDCOL;
  }
}
      

/* function: InitBuckets()
 * -----------------------
 */
void InitBuckets(bucketT *bkts, nodeT *source, long nbuckets){
  
  long i;

  /* set up bucket array parameters */
  bkts->curr=0;
  bkts->wrapped=FALSE;

  /* initialize the buckets */
  for(i=0;i<nbuckets;i++){
    bkts->bucketbase[i]=NULL;
  }

  /* put the source in the zeroth distance index bucket */
  bkts->bucket[0]=source;
  source->next=NULL;
  source->prev=NULL;
  source->group=INBUCKET;
  source->outcost=0;
  
}


/* function: InitNodes()
 * ---------------------
 */
void InitNodes(long nnrow, long nncol, nodeT **nodes, nodeT *ground){

  long row, col;

  /* loop over each element and initialize values */
  for(row=0;row<nnrow;row++){
    for(col=0;col<nncol;col++){
      nodes[row][col].group=NOTINBUCKET;
      nodes[row][col].outcost=VERYFAR;
      nodes[row][col].pred=NULL;
    }
  }

  /* initialize the ground node */
  if(ground!=NULL){
    ground->group=NOTINBUCKET;
    ground->outcost=VERYFAR;
    ground->pred=NULL;
  }
  
}


/* function: BucketInsert()
 * ------------------------
 */
void BucketInsert(nodeT *node, long ind, bucketT *bkts){

  /* put node at beginning of bucket list */
  node->next=bkts->bucket[ind];
  if((bkts->bucket[ind])!=NULL){
    bkts->bucket[ind]->prev=node;
  }
  bkts->bucket[ind]=node;
  node->prev=NULL;

  /* mark node in bucket array */
  node->group=INBUCKET;

}

  
/* function: BucketRemove()
 * ------------------------
 */
void BucketRemove(nodeT *node, long ind, bucketT *bkts){
  
  /* remove node from doubly linked list */
  if((node->next)!=NULL){
    node->next->prev=node->prev;
  }
  if(node->prev!=NULL){
    node->prev->next=node->next;
  }else if(node->next==NULL){    
    bkts->bucket[ind]=NULL;
  }else{
    bkts->bucket[ind]=node->next;
  }

}


/* function: ClosestNode()
 * -----------------------
 */
nodeT *ClosestNode(bucketT *bkts){

  nodeT *node;

  /* find the first bucket with nodes in it */
  while(TRUE){

    /* see if we got to the last bucket */
    if((bkts->curr)>(bkts->maxind)){
	return(NULL);
    }

    /* see if we found a nonempty bucket; if so, return it */
    if((bkts->bucket[bkts->curr])!=NULL){
      node=bkts->bucket[bkts->curr];
      node->group=ONTREE;
      bkts->bucket[bkts->curr]=node->next;
      if((node->next)!=NULL){
	node->next->prev=NULL;
      }
      return(node);
    }

    /* move to next bucket */
    bkts->curr++;
  
  }
}


/* function: ClosestNodeCircular()
 * -------------------------------
 * Similar to ClosestNode(), but assumes circular buckets.  This
 * function should NOT be used if negative arc weights exist on the 
 * network; initial value of bkts->minind should always be zero.
 */
nodeT *ClosestNodeCircular(bucketT *bkts){

  nodeT *node;

  /* find the first bucket with nodes in it */
  while(TRUE){

    /* see if we got to the last bucket */
    if((bkts->curr+bkts->minind)>(bkts->maxind)){
      if(bkts->wrapped){
	bkts->wrapped=FALSE;
	bkts->curr=0;
	bkts->minind+=bkts->size;
	bkts->maxind+=bkts->size;
      }else{
	return(NULL);
      }
    }

    /* see if we found a nonempty bucket; if so, return it */
    if((bkts->bucket[bkts->curr])!=NULL){
      node=bkts->bucket[bkts->curr];
      node->group=ONTREE;
      bkts->bucket[bkts->curr]=node->next;
      if((node->next)!=NULL){
	node->next->prev=NULL;
      }
      return(node);
    }

    /* move to next bucket */
    bkts->curr++;
  
  }
}


/* function: MinOutCostNode()
 * --------------------------
 * Similar to ClosestNode(), but always returns closest node even if its
 * outcost is less than the minimum bucket index.  Does not handle circular
 * buckets.  Does not handle no nodes left condition (this should be handled 
 * by calling function).
 */
nodeT *MinOutCostNode(bucketT *bkts){

  long minoutcost;
  nodeT *node1, *node2;

  /* move to next non-empty bucket */
  while(bkts->curr<bkts->maxind && bkts->bucket[bkts->curr]==NULL){
    bkts->curr++;
  }

  /* scan the whole bucket if it is the overflow or underflow bag */
  if(bkts->curr==bkts->minind || bkts->curr==bkts->maxind){

    node2=bkts->bucket[bkts->curr];
    node1=node2;
    minoutcost=node1->outcost;
    while(node2!=NULL){
      if(node2->outcost<minoutcost){
	minoutcost=node2->outcost;
	node1=node2;
      }
      node2=node2->next;
    }
    BucketRemove(node1,bkts->curr,bkts);

  }else{

    node1=bkts->bucket[bkts->curr];
    bkts->bucket[bkts->curr]=node1->next;
    if(node1->next!=NULL){
      node1->next->prev=NULL;
    }

  }

  return(node1);

}


/* function: SelectSource()
 * ------------------------
 * If params->sourcemode is zero, the ground is returned as the source.  
 * Otherwise, the returned source is the endpoint of the longest chain of
 * arcs carrying at least nflow units of flow.  This function does
 * check for the case where two arcs both carry nflow into or out of a node,
 * but if there are flow cycles (not unexpected for nonlinear costs), the
 * longest chain is not guaranteed.  Which end of the longest chain is
 * determined by the sign of params->sourcemode (should be 1 or -1 if not 0).
 */
nodeT *SelectSource(nodeT **nodes, nodeT *ground, long nflow, 
		    short **flows, long ngroundarcs, 
		    long nrow, long ncol, paramT *params){

  long row, col, maxflowlength, arcnum, upperarcnum;
  long arcrow, arccol, arcdir, endptsign;
  signed char checknode;
  nodeT *source, *node1, *node2, *nextnode;
  nodesuppT **nodesupp;
  
  /* if sourcemode==0, return ground node; otherwise, it should be 1 or -1 */
  if(!params->sourcemode){
    return(ground);
  }else{
    endptsign=params->sourcemode;
  }

  /* initialize variables */
  /* group: 0=unvisited, 1=descended, 2=done */ 
  /* outcost: longest distance to a chain end */
  /* pred: parent node */
  nodesupp=NULL;
  source=ground;
  maxflowlength=0;
  ground->group=0;
  ground->outcost=0;
  ground->pred=NULL;
  for(row=0;row<nrow-1;row++){
    for(col=0;col<ncol-1;col++){
      nodes[row][col].group=0;
      nodes[row][col].outcost=0;
      nodes[row][col].pred=NULL;
    }
  }
  
  /* loop over all nodes (upper row limit is nrow-1 so we can check ground) */
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol-1;col++){

      /* set the current node */
      if(row!=nrow-1){
	node1=&nodes[row][col];
      }else{
	if(col==0){
	  node1=ground;
	}else{
	  break;
	}
      }

      /* see if this node is an endpoint */
      checknode=FALSE;
      if(!node1->group){
	if(node1!=ground){
	  arcnum=-5;
	  upperarcnum=-1;
	}else{
	  arcnum=-1;
	  upperarcnum=ngroundarcs-1;
	}
	while(arcnum<upperarcnum){
	  node2=NeighborNode(node1,++arcnum,&upperarcnum,nodes,ground,
			     &arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);
	  
	  /* node is not beginning of a chain (may be the end, though) */
	  if(-endptsign*arcdir*flows[arcrow][arccol] >= nflow){
	    checknode=FALSE;
	    break;
	  }

	  /* node may be beginning of a chain */
	  if(endptsign*arcdir*flows[arcrow][arccol] >= nflow){
	    checknode=TRUE;
	  }
	}
      }

      /* if it is an endpoint, trace the flow and determine longest chain */
      if(checknode){
      
	/* loop until we've walked the whole tree */
	nextnode=node1;
	while(TRUE){

	  node1=nextnode;
	  nextnode=NULL;

	  /* loop over all outgoing arcs */
	  if(node1!=ground){
	    arcnum=-5;
	    upperarcnum=-1;
	  }else{
	    arcnum=-1;
	    upperarcnum=ngroundarcs-1;
	  }
	  while(arcnum<upperarcnum){
	    node2=NeighborNode(node1,++arcnum,&upperarcnum,nodes,ground,
			     &arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);

	    /* see if next node is or should be on tree */
	    /* otherwise, keep node if it is predecessor, but keep looping */
	    if(endptsign*arcdir*flows[arcrow][arccol] >= nflow){
	      if(node2->group==2){
		if(node2->outcost+1 > node1->outcost){
		  node1->outcost=node2->outcost+1;
		}	    
	      }else if(node2->group==0){
		nextnode=node2;
		break;
	      }
	    }else if(node2==node1->pred){
	      nextnode=node2;
	    }
	  }

	  /* we are back to the root if we didn't find any eligible nodes */
	  if(nextnode==NULL){

	    /* see if the tree root should be the new source */
	    if(node1->outcost > maxflowlength){
	      source=node1;
	      maxflowlength=node1->outcost;
	    }
	    node1->group=2;
	    break; 
	  }

	  /* if nextnode is pred, mark current node and go back up the tree */
	  if(nextnode->group==1){
	    node1->group=2;
	  }else{
	    node1->group=1;
	    nextnode->pred=node1;
	  }
      	}
      }
    }
  }
  
  /* return source */
  return(source);

}


/* function: GetCost()
 * -------------------
 * Returns incremental flow cost for current flow increment dflow from
 * lookup array.  
 */
short GetCost(incrcostT **incrcosts, long arcrow, long arccol, 
	      long arcdir){

  /* look up cost and return it for the appropriate arc direction */
  /* we may want add a check here for clipped incremental costs */
  if(arcdir>0){
    return(incrcosts[arcrow][arccol].poscost);
  }else{
    return(incrcosts[arcrow][arccol].negcost);
  }
}


/* function: ReCalcCost()
 * ----------------------
 * Updates the incremental cost for an arc.
 */
long ReCalcCost(void **costs, incrcostT **incrcosts, long flow, 
		long arcrow, long arccol, long nflow, long nrow, 
		paramT *params){

  long poscost, negcost, iclipped;

  /* calculate new positive and negative nflow costs, as long ints */
  CalcCost(costs,flow,arcrow,arccol,nflow,nrow,params,
	   &poscost,&negcost);

  /* clip costs to short int */
  iclipped=0;
  if(poscost>LARGESHORT){
    incrcosts[arcrow][arccol].poscost=LARGESHORT;
    iclipped++;
  }else{
    if(poscost<-LARGESHORT){
      incrcosts[arcrow][arccol].poscost=-LARGESHORT;
      iclipped++;
    }else{
      incrcosts[arcrow][arccol].poscost=poscost;
    }
  }
  if(negcost>LARGESHORT){
    incrcosts[arcrow][arccol].negcost=LARGESHORT;
    iclipped++;
  }else{
    if(negcost<-LARGESHORT){
      incrcosts[arcrow][arccol].negcost=-LARGESHORT;
      iclipped++;
    }else{
      incrcosts[arcrow][arccol].negcost=negcost;
    }
  }

  /* return the number of clipped incremental costs (0, 1, or 2) */
  return(iclipped);
}


/* function: SetupIncrFlowCosts()
 * ------------------------------
 * Calculates the costs for positive and negative dflow flow increment
 * if there is zero flow on the arc.
 */
void SetupIncrFlowCosts(void **costs, incrcostT **incrcosts, short **flows,
			long nflow, long nrow, long narcrow, 
			short *narcsperrow, paramT *params){

  long arcrow, arccol, iclipped, narcs;
  char pl[2];


  /* loop over all rows and columns */
  narcs=0;
  iclipped=0;
  for(arcrow=0;arcrow<narcrow;arcrow++){
    narcs+=narcsperrow[arcrow];
    for(arccol=0;arccol<narcsperrow[arcrow];arccol++){

      /* calculate new positive and negative nflow costs, as long ints */
      iclipped+=ReCalcCost(costs,incrcosts,flows[arcrow][arccol],
			   arcrow,arccol,nflow,nrow,params);
    }
  }

  /* print overflow warning if applicable */
  if(iclipped){
    if(iclipped>1){
      strcpy(pl,"s");
    }else{
      strcpy(pl,"");
    }
    fprintf(sp0,"%ld incremental cost%s clipped to avoid overflow (%.3f%%)\n",
	    iclipped,pl,((double )iclipped)/(2*narcs));
  }
}


/* function: EvaluateTotalCost()
 * -----------------------------
 * Computes the total cost of the flow array and prints it out.  Pass nrow
 * and ncol if in grid mode (primary network), or pass nrow=ntiles and 
 * ncol=0 for nongrid mode (secondary network).
 */
totalcostT EvaluateTotalCost(void **costs, short **flows, long nrow, long ncol,
			     short *narcsperrow,paramT *params){

  totalcostT rowcost, totalcost;
  long row, col, maxrow, maxcol;

  /* sum cost for each row and column arc */
  totalcost=0;
  if(ncol){
    maxrow=2*nrow-1;
  }else{
    maxrow=nrow;
  }
  for(row=0;row<maxrow;row++){
    rowcost=0;
    if(ncol){
      if(row<nrow-1){
	maxcol=ncol;
      }else{
	maxcol=ncol-1;
      }
    }else{
      maxcol=narcsperrow[row];
    }
    for(col=0;col<maxcol;col++){
      rowcost+=EvalCost(costs,flows,row,col,nrow,params);
    }
    totalcost+=rowcost;
  }

  return(totalcost);
}


/* function: MSTInitFlows()
 * ------------------------
 * Initializes the flow on a the network using minimum spanning tree
 * algorithm.  
 */
void MSTInitFlows(float **wrappedphase, short ***flowsptr, 
		  short **mstcosts, long nrow, long ncol, 
		  nodeT ***nodesptr, nodeT *ground, long maxflow){

  long row, col, i, maxcost;
  signed char **residue, **arcstatus;
  short **flows;
  nodeT *source;
  bucketT bkts[1];

  /* get and initialize memory for ground, nodes, buckets, and child array */
  *nodesptr=(nodeT **)Get2DMem(nrow-1,ncol-1,sizeof(nodeT *),sizeof(nodeT));
  InitNodeNums(nrow-1,ncol-1,*nodesptr,ground);

  /* find maximum cost */
  maxcost=0;
  for(row=0;row<2*nrow-1;row++){
    if(row<nrow-1){
      i=ncol;
    }else{
      i=ncol-1;
    }
    for(col=0;col<i;col++){
      if(mstcosts[row][col]>maxcost 
	 && !((row==nrow-1 || 2*nrow-2) && (col==0 || col==ncol-2))){
	maxcost=mstcosts[row][col];
      }
    }
  }

  /* get memory for buckets and arc status */
  bkts->size=LRound((maxcost+1)*(nrow+ncol+1));
  bkts->bucketbase=(nodeT **)MAlloc(bkts->size*sizeof(nodeT *));
  bkts->minind=0;
  bkts->maxind=bkts->size-1;
  bkts->bucket=bkts->bucketbase;
  arcstatus=(signed char **)Get2DRowColMem(nrow,ncol,sizeof(signed char *),
					   sizeof(signed char));

  /* calculate phase residues (integer numbers of cycles) */
  fprintf(sp1,"Initializing flows with MST algorithm\n");
  residue=(signed char **)Get2DMem(nrow-1,ncol-1,sizeof(signed char *),
				   sizeof(signed char));
  CycleResidue(wrappedphase,residue,nrow,ncol);

  /* get memory for flow arrays */
  (*flowsptr)=(short **)Get2DRowColZeroMem(nrow,ncol,
					   sizeof(short *),sizeof(short));
  flows=*flowsptr;

  /* loop until no flows exceed the maximum flow */
  fprintf(sp2,"Running approximate minimum spanning tree solver\n");
  while(TRUE){

    /* set up the source to be the first non-zero residue that we find */
    source=NULL;
    for(row=0;row<nrow-1 && source==NULL;row++){
      for(col=0;col<ncol-1 && source==NULL;col++){
	if(residue[row][col]){
	  source=&(*nodesptr)[row][col];
	}
      }
    }
    if(source==NULL){
      fprintf(sp1,"No residues found\n");
      break;
    }

    /* initialize data structures */
    InitNodes(nrow-1,ncol-1,*nodesptr,ground);
    InitBuckets(bkts,source,bkts->size);
    
    /* solve the mst problem */
    SolveMST(*nodesptr,source,ground,bkts,mstcosts,residue,arcstatus,
	     nrow,ncol);
    
    /* find flows on minimum tree (only one feasible flow exists) */
    DischargeTree(source,mstcosts,flows,residue,arcstatus,
		  *nodesptr,ground,nrow,ncol);
    
    /* do pushes to clip the flows and make saturated arcs ineligible */
    /* break out of loop if there is no flow greater than the limit */
    if(ClipFlow(residue,flows,mstcosts,nrow,ncol,maxflow)){
      break;
    }
  }
   
  /* free memory and return */
  Free2DArray((void **)residue,nrow-1);
  Free2DArray((void **)arcstatus,2*nrow-1);
  Free2DArray((void **)mstcosts,2*nrow-1);
  free(bkts->bucketbase);
  return;
  
}


/* function: SolveMST()
 * --------------------
 * Finds tree which spans all residue nodes of approximately minimal length.
 * Note that this function may produce a Steiner tree (tree may split at 
 * non-residue node), though finding the exactly minimum Steiner tree is 
 * NP-hard.  This function uses Prim's algorithm, nesting Dijkstra's 
 * shortest path algorithm in each iteration to find next closest residue 
 * node to tree.  See Ahuja, Orlin, and Magnanti 1993 for details.  
 *
 * Dijkstra implementation and some associated functions adapted from SPLIB 
 * shortest path codes written by Cherkassky, Goldberg, and Radzik.
 */
void SolveMST(nodeT **nodes, nodeT *source, nodeT *ground, 
	      bucketT *bkts, short **mstcosts, signed char **residue, 
	      signed char **arcstatus, long nrow, long ncol){

  nodeT *from, *to, *pathfrom, *pathto;
  nodesuppT **nodesupp;
  long fromdist, newdist, arcdist, ngroundarcs, groundcharge;
  long fromrow, fromcol, row, col, arcnum, upperarcnum, maxcol;
  long pathfromrow, pathfromcol;
  long arcrow, arccol, arcdir;

  /* initialize some variables */
  nodesupp=NULL;

  /* calculate the number of ground arcs */
  ngroundarcs=2*(nrow+ncol-2)-4;

  /* calculate charge on ground */
  groundcharge=0;
  for(row=0;row<nrow-1;row++){
    for(col=0;col<ncol-1;col++){
      groundcharge-=residue[row][col];
    }
  }

  /* initialize arc status array */
  for(arcrow=0;arcrow<2*nrow-1;arcrow++){
    if(arcrow<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(arccol=0;arccol<maxcol;arccol++){
      arcstatus[arcrow][arccol]=0;
    }
  }

  /* loop until there are no more nodes in any bucket */
  while((from=ClosestNode(bkts))!=NULL){

    /* info for current node */
    fromrow=from->row;
    fromcol=from->col;
    
    /* if we found a residue */
    if(((fromrow!=GROUNDROW && residue[fromrow][fromcol]) || 
       (fromrow==GROUNDROW && groundcharge)) && from!=source){
      
      /* set node and its predecessor */
      pathto=from;
      pathfrom=from->pred;

      /* go back and make arcstatus -1 along path */
      while(TRUE){

	/* give to node zero distance label */
	pathto->outcost=0;

	/* get arc indices for arc between pathfrom and pathto */
	GetArc(pathfrom,pathto,&arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);
	
	/* set arc status to -1 to mark arc on tree */
	arcstatus[arcrow][arccol]=-1;
	
	/* stop when we get to a residue */
	pathfromrow=pathfrom->row;
	pathfromcol=pathfrom->col;
	if((pathfromrow!=GROUNDROW && residue[pathfromrow][pathfromcol])
	   || (pathfromrow==GROUNDROW && groundcharge)){
	  break;
	}
	
	/* move up to previous node pair in path */
	pathto=pathfrom;
	pathfrom=pathfrom->pred;

      } /* end while loop marking costs on path */
      
    } /* end if we found a residue */

    /* set a variable for from node's distance */
    fromdist=from->outcost;

    /* scan from's neighbors */
    if(fromrow!=GROUNDROW){
      arcnum=-5;
      upperarcnum=-1;
    }else{
      arcnum=-1;
      upperarcnum=ngroundarcs-1;
    }
    while(arcnum<upperarcnum){

      /* get row, col indices and distance of next node */
      to=NeighborNode(from,++arcnum,&upperarcnum,nodes,ground,
                      &arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);
      row=to->row;
      col=to->col;

      /* get cost of arc to new node (if arc on tree, cost is 0) */
      if(arcstatus[arcrow][arccol]<0){
	arcdist=0;
      }else if((arcdist=mstcosts[arcrow][arccol])==LARGESHORT){
	arcdist=VERYFAR;
      }

      /* compare distance of new nodes to temp labels */
      if((newdist=fromdist+arcdist)<(to->outcost)){

	/* if to node is already in a bucket, remove it */
	if(to->group==INBUCKET){
	  if(to->outcost<bkts->maxind){
	    BucketRemove(to,to->outcost,bkts);
	  }else{
	    BucketRemove(to,bkts->maxind,bkts);
	  }
	}
		
	/* update to node */
	to->outcost=newdist;
	to->pred=from;

	/* insert to node into appropriate bucket */
	if(newdist<bkts->maxind){
	  BucketInsert(to,newdist,bkts);
	  if(newdist<bkts->curr){
	    bkts->curr=newdist;
	  }
	}else{
	  BucketInsert(to,bkts->maxind,bkts);
	}
	
      } /* end if newdist < old dist */
      
    } /* end loop over outgoing arcs */
  } /* end while ClosestNode()!=NULL */

}


/* function: DischargeTree()
 * -------------------------
 * Does depth-first search on result tree from SolveMST.  Integrates
 * charges from tree leaves back up to set arc flows.  This implementation
 * is non-recursive; a recursive implementation might be faster, but 
 * would also use much more stack memory.  This method is equivalent to 
 * walking the tree, so it should be nore more than a factor of 2 slower.
 */
long DischargeTree(nodeT *source, short **mstcosts, short **flows,
		   signed char **residue, signed char **arcstatus, 
		   nodeT **nodes, nodeT *ground, long nrow, long ncol){

  long row, col, todir, arcrow, arccol, arcdir;
  long arcnum, upperarcnum, ngroundarcs;
  nodeT *from, *to, *nextnode;
  nodesuppT **nodesupp;


  /* set up */
  /* use group member of node structure to temporarily store charge */
  nextnode=source;
  ground->group=0;
  for(row=0;row<nrow-1;row++){
    for(col=0;col<ncol-1;col++){
      nodes[row][col].group=residue[row][col];
      ground->group-=residue[row][col];
    }
  }
  ngroundarcs=2*(nrow+ncol-2)-4;
  nodesupp=NULL;

  /* keep looping unitl we've walked the entire tree */
  while(TRUE){
    
    from=nextnode;
    nextnode=NULL;

    /* loop over outgoing arcs from this node */
    if(from->row!=GROUNDROW){
      arcnum=-5;
      upperarcnum=-1;
    }else{
      arcnum=-1;
      upperarcnum=ngroundarcs-1;
    }
    while(arcnum<upperarcnum){

      /* get row, col indices and distance of next node */
      to=NeighborNode(from,++arcnum,&upperarcnum,nodes,ground,
		      &arcrow,&arccol,&arcdir,nrow,ncol,nodesupp);
      
      /* see if the arc is on the tree and if it has been followed yet */
      if(arcstatus[arcrow][arccol]==-1){

	/* arc has not yet been followed: move down the tree */
	nextnode=to;
	row=arcrow;
	col=arccol;
	break;

      }else if(arcstatus[arcrow][arccol]==-2){

	/* arc has already been followed and leads back up the tree: */
	/* save it, but keep looking for downwards arc */
	nextnode=to;
	row=arcrow;
	col=arccol;
	todir=arcdir;

      }
    }

    /* break if no unfollowed arcs (ie, we are done examining the tree) */
    if(nextnode==NULL){
      break;
    }

    /* if we found leaf and we're moving back up the tree, do a push */
    /* otherwise, just mark the path by decrementing arcstatus */
    if((--arcstatus[row][col])==-3){
      flows[row][col]+=todir*from->group;
      nextnode->group+=from->group;
      from->group=0;
    }
  }

  /* finish up */
  return(from->group);
  
} /* end of DischargeTree() */


/* function: ClipFlow()
 * ---------------------
 * Given a flow, clips flow magnitudes to a computed limit, resets 
 * residues so sum of solution of network problem with new residues 
 * and solution of clipped problem give total solution.  Upper flow limit
 * is 2/3 the maximum flow on the network or the passed value maxflow, 
 * whichever is greater.  Clipped flow arcs get costs of passed variable 
 * maxcost.  Residues should have been set to zero by DischargeTree().
 */
signed char ClipFlow(signed char **residue, short **flows, 
		     short **mstcosts, long nrow, long ncol, 
		     long maxflow){

  long row, col, cliplimit, maxcol, excess, tempcharge, sign;
  long mostflow, maxcost;


  /* find maximum flow */
  mostflow=Short2DRowColAbsMax(flows,nrow,ncol);

  /* if there is no flow greater than the maximum, return TRUE */
  if(mostflow<=maxflow){
    return(TRUE);
  }
  fprintf(sp2,"Maximum flow on network: %ld\n",mostflow);

  /* set upper flow limit */
  cliplimit=(long )ceil(mostflow*CLIPFACTOR)+1;
  if(maxflow>cliplimit){
    cliplimit=maxflow;
  }

  /* find maximum cost (excluding ineligible corner arcs) */
  maxcost=0;
  for(row=0;row<2*nrow-1;row++){
    if(row<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(col=0;col<maxcol;col++){
      if(mstcosts[row][col]>maxcost && mstcosts[row][col]<LARGESHORT){
	maxcost=mstcosts[row][col];
      }
    }
  }

  /* set the new maximum cost and make sure it doesn't overflow short int */
  maxcost+=INITMAXCOSTINCR;
  if(maxcost>=LARGESHORT){
    fprintf(sp0,"WARNING: escaping ClipFlow loop to prevent cost overflow\n");
    return(TRUE);
  }

  /* clip flows and do pushes */
  for(row=0;row<2*nrow-1;row++){
    if(row<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(col=0;col<maxcol;col++){
      if(labs(flows[row][col])>cliplimit){
	if(flows[row][col]>0){
	  sign=1;
	  excess=flows[row][col]-cliplimit;
	}else{
	  sign=-1;
	  excess=flows[row][col]+cliplimit;
	}
	if(row<nrow-1){
	  if(col!=0){
	    tempcharge=residue[row][col-1]+excess;
	    if(tempcharge>MAXRES || tempcharge<MINRES){
	      fprintf(sp0,"Overflow of residue data type\nAbort\n");
	      exit(ABNORMAL_EXIT);
	    }
	    residue[row][col-1]=tempcharge;
	  }
	  if(col!=ncol-1){
	    tempcharge=residue[row][col]-excess;
	    if(tempcharge<MINRES || tempcharge>MAXRES){
	      fprintf(sp0,"Overflow of residue data type\nAbort\n");
	      exit(ABNORMAL_EXIT);
	    }
	    residue[row][col]=tempcharge;
	  }
	}else{
	  if(row!=nrow-1){
	    tempcharge=residue[row-nrow][col]+excess;
	    if(tempcharge>MAXRES || tempcharge<MINRES){
	      fprintf(sp0,"Overflow of residue data type\nAbort\n");
	      exit(ABNORMAL_EXIT);
	    }
	    residue[row-nrow][col]=tempcharge;
	  }
	  if(row!=2*nrow-2){
	    tempcharge=residue[row-nrow+1][col]-excess;
	    if(tempcharge<MINRES || tempcharge>MAXRES){
	      fprintf(sp0,"Overflow of residue data type\nAbort\n");
	      exit(ABNORMAL_EXIT);
	    }
	    residue[row-nrow+1][col]=tempcharge;
	  }
	}
	flows[row][col]=sign*cliplimit;
	mstcosts[row][col]=maxcost;
      }
    }
  }

  /* return value indicates that flows have been clipped */
  fprintf(sp2,"Flows clipped to %ld.  Rerunning MST solver.\n",cliplimit);
  return(FALSE);

}


/* function: MCFInitFlows()
 * ------------------------
 * Initializes the flow on a the network using minimum cost flow
 * algorithm.  
 */
void MCFInitFlows(float **wrappedphase, short ***flowsptr, short **mstcosts, 
		  long nrow, long ncol, long cs2scalefactor){

  signed char **residue;

#ifndef NO_CS2

  /* calculate phase residues (integer numbers of cycles) */
  fprintf(sp1,"Initializing flows with MCF algorithm\n");
  residue=(signed char **)Get2DMem(nrow-1,ncol-1,sizeof(signed char *),
				   sizeof(signed char));
  CycleResidue(wrappedphase,residue,nrow,ncol);

  /* run the solver (memory freed within solver) */
  SolveCS2(residue,mstcosts,nrow,ncol,cs2scalefactor,flowsptr);

#endif
}
