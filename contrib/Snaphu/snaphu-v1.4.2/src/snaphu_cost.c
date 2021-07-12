/*************************************************************************

  snaphu statistical cost model source file
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


/* function: BuildCostArrays()
 * ---------------------------
 * Builds cost arrays for arcs based on interferogram intensity 
 * and correlation, depending on options and passed parameters.  
 */
void BuildCostArrays(void ***costsptr, short ***mstcostsptr, 
		     float **mag, float **wrappedphase, 
		     float **unwrappedest, long linelen, long nlines, 
		     long nrow, long ncol, paramT *params, 
		     tileparamT *tileparams, infileT *infiles, 
		     outfileT *outfiles){
  
  long row, col, maxcol, tempcost;
  long poscost, negcost, costtypesize;
  float **pwr, **corr;
  short **weights, **rowweight, **colweight, **scalarcosts;
  void **costs, **rowcost, **colcost;
  void (*CalcStatCost)();

  /* read weights */
  weights=NULL;
  ReadWeightsFile(&weights,infiles->weightfile,linelen,nlines,tileparams);
  rowweight=weights;
  colweight=&weights[nrow-1];

  /* if we're only initializing and we don't want statistical weights */
  if(params->initonly && params->costmode==NOSTATCOSTS){
    *mstcostsptr=weights;
    return;
  }

  /* size of the data type for holding cost data depends on cost mode */
  if(params->costmode==TOPO){
    costtypesize=sizeof(costT);
  }else if(params->costmode==DEFO){
    costtypesize=sizeof(costT);
  }else if(params->costmode==SMOOTH){
    costtypesize=sizeof(smoothcostT);
  }

  /* build or read the statistical cost arrays unless we were told not to */
  if(strlen(infiles->costinfile)){
    fprintf(sp1,"Reading cost information from file %s\n",infiles->costinfile);
    costs=NULL;
    Read2DRowColFile((void ***)&costs,infiles->costinfile,
		     linelen,nlines,tileparams,costtypesize);
    (*costsptr)=costs;

  }else if(params->costmode!=NOSTATCOSTS){

    /* get intensity and correlation info */
    /* correlation generated from interferogram and amplitude if not given */
    GetIntensityAndCorrelation(mag,wrappedphase,&pwr,&corr,infiles,
			       linelen,nlines,nrow,ncol,outfiles,
			       params,tileparams);

    /* call specific functions for building cost array and */
    /* set global pointers to functions for calculating and evaluating costs */
    if(params->costmode==TOPO){
      fprintf(sp1,"Calculating topography-mode cost parameters\n");
      costs=BuildStatCostsTopo(wrappedphase,mag,unwrappedest,pwr,corr,
			       rowweight,colweight,nrow,ncol,tileparams,
			       outfiles,params);
    }else if(params->costmode==DEFO){
      fprintf(sp1,"Calculating deformation-mode cost parameters\n");
      costs=BuildStatCostsDefo(wrappedphase,mag,unwrappedest,corr,
			       rowweight,colweight,nrow,ncol,tileparams,
			       outfiles,params);
    }else if(params->costmode==SMOOTH){
      fprintf(sp1,"Calculating smooth-solution cost parameters\n");
      costs=BuildStatCostsSmooth(wrappedphase,mag,unwrappedest,corr,
				 rowweight,colweight,nrow,ncol,tileparams,
				 outfiles,params);
    }else{
      fprintf(sp0,"unrecognized cost mode\n");
      exit(ABNORMAL_EXIT);
    }
    (*costsptr)=costs;
    

  }/* end if(params->costmode!=NOSTATCOSTS) */
  
  /* set array subpointers and temporary cost-calculation function pointer */
  if(params->costmode==TOPO){
    rowcost=costs;
    colcost=(void **)&(((costT **)costs)[nrow-1]);
    CalcStatCost=CalcCostTopo;
  }else if(params->costmode==DEFO){
    rowcost=costs;
    colcost=(void **)&(((costT **)costs)[nrow-1]);
    CalcStatCost=CalcCostDefo;
  }else if(params->costmode==SMOOTH){
    rowcost=costs;
    colcost=(void **)&(((smoothcostT **)costs)[nrow-1]);
    CalcStatCost=CalcCostSmooth;
  }

  /* dump statistical cost arrays */
  if(strlen(infiles->costinfile) || params->costmode!=NOSTATCOSTS){
    if(strlen(outfiles->costoutfile)){
      Write2DRowColArray((void **)costs,outfiles->costoutfile,
			nrow,ncol,costtypesize);
    }else{
      if(strlen(outfiles->rowcostfile)){
	Write2DArray((void **)rowcost,outfiles->rowcostfile,
		     nrow-1,ncol,costtypesize);
      }
      if(strlen(outfiles->colcostfile)){
	Write2DArray((void **)colcost,outfiles->colcostfile,
		     nrow,ncol-1,costtypesize);
      }
    }
  }

  /* get memory for scalar costs if in Lp mode */
  if(params->p>=0){
    scalarcosts=(short **)Get2DRowColMem(nrow,ncol,
					 sizeof(short *),sizeof(short));
    (*costsptr)=(void **)scalarcosts;
  }

  /* now, set scalar costs for MST initialization or optimization if needed */
  if(params->costmode==NOSTATCOSTS){    

    /* if in no-statistical-costs mode, copy weights to scalarcosts array */
    if(!params->initonly){
      for(row=0;row<2*nrow-1;row++){
	if(row<nrow-1){
	  maxcol=ncol;
	}else{
	  maxcol=ncol-1;
	}
	for(col=0;col<maxcol;col++){
	  scalarcosts[row][col]=weights[row][col];
	}
      }
    }

    /* unless input is already unwrapped, use weights memory for mstcosts */
    if(!params->unwrapped){
      (*mstcostsptr)=weights;
    }else{
      Free2DArray((void **)weights,2*nrow-1);
      (*mstcostsptr)=NULL;
    }

  }else if(!params->unwrapped || params->p>=0){

    /* if we got here, we had statistical costs and we need scalar weights */
    /*   from them for MST initialization or for Lp optimization */
    for(row=0;row<2*nrow-1;row++){
      if(row<nrow-1){
	maxcol=ncol;
      }else{
	maxcol=ncol-1;
      }
      for(col=0;col<maxcol;col++){

	/* calculate incremental costs for flow=0, nflow=1 */
	CalcStatCost((void **)costs,0,row,col,1,nrow,params,
		     &poscost,&negcost);

	/* take smaller of positive and negative incremental cost */
	if(poscost<negcost){
	  tempcost=poscost;
	}else{
	  tempcost=negcost;
	}

	/* clip scalar cost so it is between 1 and params->maxcost */
	if(tempcost<params->maxcost){
	  if(tempcost>MINSCALARCOST){
	    weights[row][col]=tempcost;
	  }else{
	    weights[row][col]=MINSCALARCOST;
	  }
	}else{
	  weights[row][col]=params->maxcost;
	}
	if(params->p>=0){
	  scalarcosts[row][col]=weights[row][col];
	}
      }
    }

    /* set costs for corner arcs to prevent ambiguous flows */
    weights[nrow-1][0]=LARGESHORT;
    weights[nrow-1][ncol-2]=LARGESHORT;
    weights[2*nrow-2][0]=LARGESHORT;
    weights[2*nrow-2][ncol-2]=LARGESHORT;
    if(params->p>=0){
      scalarcosts[nrow-1][0]=LARGESHORT;
      scalarcosts[nrow-1][ncol-2]=LARGESHORT;
      scalarcosts[2*nrow-2][0]=LARGESHORT;
      scalarcosts[2*nrow-2][ncol-2]=LARGESHORT;
    }
    
    /* dump mst initialization costs */
    if(strlen(outfiles->mstrowcostfile)){
      Write2DArray((void **)rowweight,outfiles->mstrowcostfile,
		   nrow-1,ncol,sizeof(short));
    }
    if(strlen(outfiles->mstcolcostfile)){
      Write2DArray((void **)colweight,outfiles->mstcolcostfile,
		   nrow,ncol-1,sizeof(short)); 
    }
    if(strlen(outfiles->mstcostsfile)){
      Write2DRowColArray((void **)rowweight,outfiles->mstcostsfile,
			 nrow,ncol,sizeof(short));
    }

    /* unless input is unwrapped, calculate initialization max flow */
    if(params->initmaxflow==AUTOCALCSTATMAX && !params->unwrapped){
      CalcInitMaxFlow(params,(void **)costs,nrow,ncol);
    }

    /* free costs memory if in init-only or Lp mode */
    if(params->initonly || params->p>=0){
      Free2DArray((void **)costs,2*nrow-1);
    }

    /* use memory allocated for weights arrays for mstcosts if needed */
    if(!params->unwrapped){
      (*mstcostsptr)=weights;
    }else{
      Free2DArray((void **)weights,2*nrow-1);
    }
 
  }else{
    Free2DArray((void **)weights,2*nrow-1);
  }

}


/* function: BuildStatCostsTopo()
 * ------------------------------
 * Builds statistical cost arrays for topography mode.
 */
void **BuildStatCostsTopo(float **wrappedphase, float **mag, 
			  float **unwrappedest, float **pwr, 
			  float **corr, short **rowweight, short **colweight,
			  long nrow, long ncol, tileparamT *tileparams, 
			  outfileT *outfiles, paramT *params){

  long row, col, iei, nrho, nominctablesize;
  long kperpdpsi, kpardpsi, sigsqshortmin;
  double a, re, dr, slantrange, nearrange, nominc0, dnominc;
  double nomincangle, nomincind, sinnomincangle, cosnomincangle, bperp;
  double baseline, baselineangle, lambda, lookangle;
  double dzlay, dzei, dzr0, dzrcrit, dzeimin, dphilaypeak, dzrhomax;
  double azdzfactor, dzeifactor, dzeiweight, dzlayfactor;
  double avgei, eicrit, layminei, laywidth, slope1, const1, slope2, const2;
  double rho, rho0, rhomin, drho, rhopow;
  double sigsqrho, sigsqrhoconst, sigsqei, sigsqlay;
  double glay, costscale, ambiguityheight, ztoshort, ztoshortsq;
  double nshortcycle, midrangeambight;
  float **ei, **dpsi, **avgdpsi, *dzrcrittable, **dzrhomaxtable;
  costT **costs, **rowcost, **colcost;
  signed char noshadow, nolayover;


  /* get memory and set cost array pointers */
  costs=(costT **)Get2DRowColMem(nrow,ncol,sizeof(costT *),sizeof(costT));
  rowcost=(costT **)costs;
  colcost=(costT **)&costs[nrow-1];

  /* set up */
  rho0=(params->rhosconst1)/(params->ncorrlooks)+(params->rhosconst2);
  rhomin=params->rhominfactor*rho0;
  rhopow=2*(params->cstd1)+(params->cstd2)*log(params->ncorrlooks)
    +(params->cstd3)*(params->ncorrlooks);
  sigsqshortmin=params->sigsqshortmin;
  kperpdpsi=params->kperpdpsi;
  kpardpsi=params->kpardpsi;
  dr=params->dr;
  nearrange=params->nearrange+dr*tileparams->firstcol;
  drho=params->drho;
  nrho=(long )floor((1-rhomin)/drho)+1;
  nshortcycle=params->nshortcycle;
  layminei=params->layminei;
  laywidth=params->laywidth;
  azdzfactor=params->azdzfactor;
  dzeifactor=params->dzeifactor;
  dzeiweight=params->dzeiweight;
  dzeimin=params->dzeimin;
  dzlayfactor=params->dzlayfactor;
  sigsqei=params->sigsqei;
  lambda=params->lambda;
  noshadow=!(params->shadow);
  a=params->orbitradius;
  re=params->earthradius;

  /* despeckle the interferogram intensity */
  fprintf(sp2,"Despeckling intensity image\n");
  ei=NULL;
  Despeckle(pwr,&ei,nrow,ncol);
  Free2DArray((void **)pwr,nrow);

  /* remove large-area average intensity */
  fprintf(sp2,"Normalizing intensity\n");
  RemoveMean(ei,nrow,ncol,params->krowei,params->kcolei);

  /* dump normalized, despeckled intensity */
  if(strlen(outfiles->eifile)){
    Write2DArray((void **)ei,outfiles->eifile,nrow,ncol,sizeof(float));
  }

  /* compute some midswath parameters */
  slantrange=nearrange+ncol/2*dr;
  sinnomincangle=sin(acos((a*a-slantrange*slantrange-re*re)
			  /(2*slantrange*re)));
  lookangle=asin(re/a*sinnomincangle);

  /* see if we were passed bperp rather than baseline and baselineangle */
  if(params->bperp){
    if(params->bperp>0){
      params->baselineangle=lookangle;
    }else{
      params->baselineangle=lookangle+PI;
    }
    params->baseline=fabs(params->bperp);
  }

  /* the baseline should be halved if we are in single antenna transmit mode */
  if(params->transmitmode==SINGLEANTTRANSMIT){
    params->baseline/=2.0;
  }
  baseline=params->baseline;
  baselineangle=params->baselineangle;

  /* build lookup table for dzrcrit vs incidence angle */
  dzrcrittable=BuildDZRCritLookupTable(&nominc0,&dnominc,&nominctablesize,
				       tileparams,params);

  /* build lookup table for dzrhomax vs incidence angle */
  dzrhomaxtable=BuildDZRhoMaxLookupTable(nominc0,dnominc,nominctablesize,
					 rhomin,drho,nrho,params);
  
  /* set cost autoscale factor based on midswath parameters */
  bperp=baseline*cos(lookangle-baselineangle);
  midrangeambight=fabs(lambda*slantrange*sinnomincangle/(2*bperp));
  costscale=params->costscale*fabs(params->costscaleambight/midrangeambight);
  glay=-costscale*log(params->layconst);

  /* get memory for wrapped difference arrays */
  dpsi=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  avgdpsi=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));

  /* build array of mean wrapped phase differences in range */
  /* simple average of phase differences is biased, but mean phase */
  /*   differences usually near zero, so don't bother with complex average */
  fprintf(sp2,"Building range cost arrays\n");
  CalcWrappedRangeDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
			nrow,ncol);

  /* build colcost array (range slopes) */
  /* loop over range */
  for(col=0;col<ncol-1;col++){

    /* compute range dependent parameters */
    slantrange=nearrange+col*dr;
    cosnomincangle=(a*a-slantrange*slantrange-re*re)/(2*slantrange*re);
    nomincangle=acos(cosnomincangle);
    sinnomincangle=sin(nomincangle);
    lookangle=asin(re/a*sinnomincangle);
    dzr0=-dr*cosnomincangle;
    bperp=baseline*cos(lookangle-baselineangle);
    ambiguityheight=-(lambda*slantrange*sinnomincangle)/(2*bperp);
    sigsqrhoconst=2.0*ambiguityheight*ambiguityheight/12.0;  
    ztoshort=nshortcycle/ambiguityheight;
    ztoshortsq=ztoshort*ztoshort;
    sigsqlay=ambiguityheight*ambiguityheight*params->sigsqlayfactor;

    /* interpolate scattering model parameters */
    nomincind=(nomincangle-nominc0)/dnominc;
    dzrcrit=LinInterp1D(dzrcrittable,nomincind,nominctablesize);
    SolveEIModelParams(&slope1,&slope2,&const1,&const2,dzrcrit,dzr0,
		       sinnomincangle,cosnomincangle,params);
    eicrit=(dzrcrit-const1)/slope1;
    dphilaypeak=params->dzlaypeak/ambiguityheight;

    /* loop over azimuth */
    for(row=0;row<nrow;row++){
	
      /* see if we have a masked pixel */
      if(mag[row][col]==0 || mag[row][col+1]==0){
	  
	/* masked pixel */
	colcost[row][col].laycost=0;
	colcost[row][col].offset=LARGESHORT/2;
	colcost[row][col].dzmax=LARGESHORT;
	colcost[row][col].sigsq=LARGESHORT;

      }else{

	/* topography-mode costs */

	/* calculate variance due to decorrelation */
	/* factor of 2 in sigsqrhoconst for pdf convolution */
	rho=corr[row][col];
	if(rho<rhomin){
	  rho=0;
	}
	sigsqrho=sigsqrhoconst*pow(1-rho,rhopow);

	/* calculate dz expected from EI if no layover */
	if(ei[row][col]>eicrit){
	  dzei=(slope2*ei[row][col]+const2)*dzeifactor;
	}else{
	  dzei=(slope1*ei[row][col]+const1)*dzeifactor;
	}
	if(noshadow && dzei<dzeimin){
	  dzei=dzeimin;
	}

	/* calculate dz expected from EI if layover exists */
	dzlay=0;
	if(ei[row][col]>layminei){
	  for(iei=0;iei<laywidth;iei++){
	    if(ei[row][col+iei]>eicrit){
	      dzlay+=slope2*ei[row][col+iei]+const2;
	    }else{
	      dzlay+=slope1*ei[row][col+iei]+const1;
	    }
	    if(col+iei>ncol-2){
	      break;
	    }
	  }
	}
	if(dzlay){
	  dzlay=(dzlay+iei*(-2.0*dzr0))*dzlayfactor;
	}
	  
	/* set maximum dz based on unbiased correlation and layover max */ 
	if(rho>0){
	  dzrhomax=LinInterp2D(dzrhomaxtable,nomincind,(rho-rhomin)/drho,
			       nominctablesize,nrho);
	  if(dzrhomax<dzlay){  
	    dzlay=dzrhomax;
	  }
	}

	/* set cost parameters in terms of flow, represented as shorts */
	nolayover=TRUE;
	if(dzlay){
	  if(rho>0){
	    colcost[row][col].offset=nshortcycle*
	      (dpsi[row][col]-0.5*(avgdpsi[row][col]+dphilaypeak));
	  }else{
	    colcost[row][col].offset=nshortcycle*
	      (dpsi[row][col]-0.25*avgdpsi[row][col]-0.75*dphilaypeak);
	  }
	  colcost[row][col].sigsq=(sigsqrho+sigsqei+sigsqlay)*ztoshortsq
	    /(costscale*colweight[row][col]);
	  if(colcost[row][col].sigsq<sigsqshortmin){
	    colcost[row][col].sigsq=sigsqshortmin;
	  }
	  colcost[row][col].dzmax=dzlay*ztoshort;
	  colcost[row][col].laycost=colweight[row][col]*glay;
	  if(labs(colcost[row][col].dzmax)
	     >floor(sqrt(colcost[row][col].laycost*colcost[row][col].sigsq))){
	    nolayover=FALSE;
	  }
	}
	if(nolayover){
	  colcost[row][col].sigsq=(sigsqrho+sigsqei)*ztoshortsq
	    /(costscale*colweight[row][col]);
	  if(colcost[row][col].sigsq<sigsqshortmin){
	    colcost[row][col].sigsq=sigsqshortmin;
	  }
	  if(rho>0){
	    colcost[row][col].offset=ztoshort*
	      (ambiguityheight*(dpsi[row][col]-0.5*avgdpsi[row][col])
	       -0.5*dzeiweight*dzei);
	  }else{
	    colcost[row][col].offset=ztoshort*
	      (ambiguityheight*(dpsi[row][col]-0.25*avgdpsi[row][col])
	       -0.75*dzeiweight*dzei);
	  }
	  colcost[row][col].laycost=NOCOSTSHELF;
	  colcost[row][col].dzmax=LARGESHORT;
	}

	/* shift PDF to account for flattening by coarse unwrapped estimate */
	if(unwrappedest!=NULL){
	  colcost[row][col].offset+=(nshortcycle/TWOPI*
				     (unwrappedest[row][col+1]
				      -unwrappedest[row][col]));
	}

      }
    }
  } /* end of range gradient cost calculation */

  /* reset layover constant for row (azimuth) costs */
  glay+=(-costscale*log(azdzfactor)); 

  /* build array of mean wrapped phase differences in azimuth */
  /* biased, but not much, so don't bother with complex averaging */
  fprintf(sp2,"Building azimuth cost arrays\n");
  CalcWrappedAzDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
		     nrow,ncol);
  
  /* build rowcost array */
  /* for the rowcost array, there is symmetry between positive and */
  /*   negative flows, so we average ei[][] and corr[][] values in azimuth */
  /* loop over range */
  for(col=0;col<ncol;col++){

    /* compute range dependent parameters */
    slantrange=nearrange+col*dr;
    cosnomincangle=(a*a-slantrange*slantrange-re*re)/(2*slantrange*re);
    nomincangle=acos(cosnomincangle);
    sinnomincangle=sin(nomincangle);
    lookangle=asin(re/a*sinnomincangle);
    dzr0=-dr*cosnomincangle;
    bperp=baseline*cos(lookangle-baselineangle);
    ambiguityheight=-lambda*slantrange*sinnomincangle/(2*bperp);
    sigsqrhoconst=2.0*ambiguityheight*ambiguityheight/12.0;  
    ztoshort=nshortcycle/ambiguityheight;
    ztoshortsq=ztoshort*ztoshort;
    sigsqlay=ambiguityheight*ambiguityheight*params->sigsqlayfactor;

    /* interpolate scattering model parameters */
    nomincind=(nomincangle-nominc0)/dnominc;
    dzrcrit=LinInterp1D(dzrcrittable,nomincind,nominctablesize);
    SolveEIModelParams(&slope1,&slope2,&const1,&const2,dzrcrit,dzr0,
		       sinnomincangle,cosnomincangle,params);
    eicrit=(dzrcrit-const1)/slope1;
    dphilaypeak=params->dzlaypeak/ambiguityheight;

    /* loop over azimuth */
    for(row=0;row<nrow-1;row++){
	
      /* see if we have a masked pixel */
      if(mag[row][col]==0 || mag[row+1][col]==0){
	  
	/* masked pixel */
	rowcost[row][col].laycost=0;
	rowcost[row][col].offset=LARGESHORT/2;
	rowcost[row][col].dzmax=LARGESHORT;
	rowcost[row][col].sigsq=LARGESHORT;

      }else{

	/* topography-mode costs */

	/* variance due to decorrelation */
	/* get correlation and clip small values because of estimator bias */
	rho=(corr[row][col]+corr[row+1][col])/2.0;
	if(rho<rhomin){
	  rho=0;
	}
	sigsqrho=sigsqrhoconst*pow(1-rho,rhopow);

	/* if no layover, the expected dz for azimuth will always be 0 */
	dzei=0;
	
	/* calculate dz expected from EI if layover exists */
	dzlay=0;
	avgei=(ei[row][col]+ei[row+1][col])/2.0;
	if(avgei>layminei){
	  for(iei=0;iei<laywidth;iei++){
	    avgei=(ei[row][col+iei]+ei[row+1][col+iei])/2.0;
	    if(avgei>eicrit){
	      dzlay+=slope2*avgei+const2;
	    }else{
	      dzlay+=slope1*avgei+const1;
	    }
	    if(col+iei>ncol-2){
	      break;
	    }
	  }
	}
	if(dzlay){
	  dzlay=(dzlay+iei*(-2.0*dzr0))*dzlayfactor;
	}
	  
	/* set maximum dz based on correlation max and layover max */ 
	if(rho>0){
	  dzrhomax=LinInterp2D(dzrhomaxtable,nomincind,(rho-rhomin)/drho,
			       nominctablesize,nrho);
	  if(dzrhomax<dzlay){
	    dzlay=dzrhomax;
	  }
	}

	/* set cost parameters in terms of flow, represented as shorts */
	if(rho>0){
	  rowcost[row][col].offset=nshortcycle*
	    (dpsi[row][col]-avgdpsi[row][col]);
	}else{
	  rowcost[row][col].offset=nshortcycle*
	    (dpsi[row][col]-0.5*avgdpsi[row][col]);
	}
	nolayover=TRUE;
	if(dzlay){
	  rowcost[row][col].sigsq=(sigsqrho+sigsqei+sigsqlay)*ztoshortsq
	    /(costscale*rowweight[row][col]);
	  if(rowcost[row][col].sigsq<sigsqshortmin){
	    rowcost[row][col].sigsq=sigsqshortmin;
	  }
	  rowcost[row][col].dzmax=fabs(dzlay*ztoshort);
	  rowcost[row][col].laycost=rowweight[row][col]*glay;
	  if(labs(rowcost[row][col].dzmax)
	     >floor(sqrt(rowcost[row][col].laycost*rowcost[row][col].sigsq))){
	    nolayover=FALSE;
	  }
	}
	if(nolayover){
	  rowcost[row][col].sigsq=(sigsqrho+sigsqei)*ztoshortsq
	    /(costscale*rowweight[row][col]);
	  if(rowcost[row][col].sigsq<sigsqshortmin){
	    rowcost[row][col].sigsq=sigsqshortmin;
	  }
	  rowcost[row][col].laycost=NOCOSTSHELF;
	  rowcost[row][col].dzmax=LARGESHORT;
	}

	/* shift PDF to account for flattening by coarse unwrapped estimate */
	if(unwrappedest!=NULL){
	  rowcost[row][col].offset+=(nshortcycle/TWOPI*
				     (unwrappedest[row+1][col]
				      -unwrappedest[row][col]));
	}

      }
    }
  }  /* end of azimuth gradient cost calculation */

  /* free temporary arrays */
  Free2DArray((void **)corr,nrow);
  Free2DArray((void **)dpsi,nrow);
  Free2DArray((void **)avgdpsi,nrow);
  Free2DArray((void **)ei,nrow);
  free(dzrcrittable);
  Free2DArray((void **)dzrhomaxtable,nominctablesize);

  /* return pointer to costs arrays */
  return((void **)costs);

}


/* function: BuildStatCostsDefo()
 * ------------------------------
 * Builds statistical cost arrays for deformation mode.
 */
void **BuildStatCostsDefo(float **wrappedphase, float **mag, 
			  float **unwrappedest, float **corr, 
			  short **rowweight, short **colweight,
			  long nrow, long ncol, tileparamT *tileparams, 
			  outfileT *outfiles, paramT *params){

  long row, col;
  long kperpdpsi, kpardpsi, sigsqshortmin, defomax;
  double rho, rho0, rhopow;
  double defocorrthresh, sigsqcorr, sigsqrho, sigsqrhoconst;
  double glay, costscale;
  double nshortcycle, nshortcyclesq;
  float **dpsi, **avgdpsi;
  costT **costs, **rowcost, **colcost;

  /* get memory and set cost array pointers */
  costs=(costT **)Get2DRowColMem(nrow,ncol,sizeof(costT *),sizeof(costT));
  rowcost=(costT **)costs;
  colcost=(costT **)&costs[nrow-1];

  /* set up */
  rho0=(params->rhosconst1)/(params->ncorrlooks)+(params->rhosconst2);
  defocorrthresh=params->defothreshfactor*rho0;
  rhopow=2*(params->cstd1)+(params->cstd2)*log(params->ncorrlooks)
    +(params->cstd3)*(params->ncorrlooks);
  sigsqrhoconst=2.0/12.0;
  sigsqcorr=params->sigsqcorr;
  sigsqshortmin=params->sigsqshortmin;
  kperpdpsi=params->kperpdpsi;
  kpardpsi=params->kpardpsi;
  costscale=params->costscale; 
  nshortcycle=params->nshortcycle;
  nshortcyclesq=nshortcycle*nshortcycle;
  glay=-costscale*log(params->defolayconst);
  defomax=(long )ceil(params->defomax*nshortcycle);

  /* get memory for wrapped difference arrays */
  dpsi=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  avgdpsi=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));

  /* build array of mean wrapped phase differences in range */
  /* simple average of phase differences is biased, but mean phase */
  /*   differences usually near zero, so don't bother with complex average */
  fprintf(sp2,"Building range cost arrays\n");
  CalcWrappedRangeDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
			nrow,ncol);

  /* build colcost array (range slopes) */
  for(col=0;col<ncol-1;col++){
    for(row=0;row<nrow;row++){

      /* see if we have a masked pixel */
      if(mag[row][col]==0 || mag[row][col+1]==0){
	  
	/* masked pixel */
	colcost[row][col].laycost=0;
	colcost[row][col].offset=0;
	colcost[row][col].dzmax=LARGESHORT;
	colcost[row][col].sigsq=LARGESHORT;

      }else{

	/* deformation-mode costs */
	
	/* calculate variance due to decorrelation */
	/* need symmetry for range if deformation */
	rho=(corr[row][col]+corr[row][col+1])/2.0;
	if(rho<defocorrthresh){
	  rho=0;
	}
	sigsqrho=(sigsqrhoconst*pow(1-rho,rhopow)+sigsqcorr)*nshortcyclesq;

	/* set cost paramaters in terms of flow, represented as shorts */
	if(rho>0){
	  colcost[row][col].offset=nshortcycle*
	    (dpsi[row][col]-avgdpsi[row][col]);
	}else{
	  colcost[row][col].offset=nshortcycle*
	    (dpsi[row][col]-0.5*avgdpsi[row][col]);       
	}
	colcost[row][col].sigsq=sigsqrho/(costscale*colweight[row][col]);
	if(colcost[row][col].sigsq<sigsqshortmin){
	  colcost[row][col].sigsq=sigsqshortmin;
	}
	if(rho<defocorrthresh){
	  colcost[row][col].dzmax=defomax;
	  colcost[row][col].laycost=colweight[row][col]*glay;
	  if(colcost[row][col].dzmax<floor(sqrt(colcost[row][col].laycost
						*colcost[row][col].sigsq))){
	    colcost[row][col].laycost=NOCOSTSHELF;
	    colcost[row][col].dzmax=LARGESHORT;
	  }
	}else{
	  colcost[row][col].laycost=NOCOSTSHELF;
	  colcost[row][col].dzmax=LARGESHORT;
	}
      }

      /* shift PDF to account for flattening by coarse unwrapped estimate */
      if(unwrappedest!=NULL){
	colcost[row][col].offset+=(nshortcycle/TWOPI*
				   (unwrappedest[row][col+1]
				    -unwrappedest[row][col]));
      }
    }
  }  /* end of range gradient cost calculation */

  /* build array of mean wrapped phase differences in azimuth */
  /* biased, but not much, so don't bother with complex averaging */
  fprintf(sp2,"Building azimuth cost arrays\n");
  CalcWrappedAzDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
		     nrow,ncol);

  /* build rowcost array */
  for(col=0;col<ncol;col++){
    for(row=0;row<nrow-1;row++){

      /* see if we have a masked pixel */
      if(mag[row][col]==0 || mag[row+1][col]==0){
	  
	/* masked pixel */
	rowcost[row][col].laycost=0;
	rowcost[row][col].offset=0;
	rowcost[row][col].dzmax=LARGESHORT;
	rowcost[row][col].sigsq=LARGESHORT;

      }else{

	/* deformation-mode costs */

	/* variance due to decorrelation */
	/* get correlation and clip small values because of estimator bias */
	rho=(corr[row][col]+corr[row+1][col])/2.0;
	if(rho<defocorrthresh){
	  rho=0;
	}
        sigsqrho=(sigsqrhoconst*pow(1-rho,rhopow)+sigsqcorr)*nshortcyclesq;

	/* set cost paramaters in terms of flow, represented as shorts */
	if(rho>0){
	  rowcost[row][col].offset=nshortcycle*
	    (dpsi[row][col]-avgdpsi[row][col]);
	}else{
	  rowcost[row][col].offset=nshortcycle*
	    (dpsi[row][col]-0.5*avgdpsi[row][col]);
	}
	rowcost[row][col].sigsq=sigsqrho/(costscale*rowweight[row][col]);
	if(rowcost[row][col].sigsq<sigsqshortmin){
	  rowcost[row][col].sigsq=sigsqshortmin;
	}
	if(rho<defocorrthresh){
	  rowcost[row][col].dzmax=defomax;
	  rowcost[row][col].laycost=rowweight[row][col]*glay;
	  if(rowcost[row][col].dzmax<floor(sqrt(rowcost[row][col].laycost
						*rowcost[row][col].sigsq))){
	    rowcost[row][col].laycost=NOCOSTSHELF;
	    rowcost[row][col].dzmax=LARGESHORT;
	  }
	}else{
	  rowcost[row][col].laycost=NOCOSTSHELF;
	  rowcost[row][col].dzmax=LARGESHORT;
	}
      }

      /* shift PDF to account for flattening by coarse unwrapped estimate */
      if(unwrappedest!=NULL){
	rowcost[row][col].offset+=(nshortcycle/TWOPI*
				   (unwrappedest[row+1][col]
				    -unwrappedest[row][col]));
      }
    }
  } /* end of azimuth cost calculation */

  /* free temporary arrays */
  Free2DArray((void **)corr,nrow);
  Free2DArray((void **)dpsi,nrow);
  Free2DArray((void **)avgdpsi,nrow);

  /* return pointer to costs arrays */
  return((void **)costs);

}


/* function: BuildStatCostsSmooth()
 * --------------------------------
 * Builds statistical cost arrays for smooth-solution mode.
 */
void **BuildStatCostsSmooth(float **wrappedphase, float **mag, 
			    float **unwrappedest, float **corr, 
			    short **rowweight, short **colweight,
			    long nrow, long ncol, tileparamT *tileparams, 
			    outfileT *outfiles, paramT *params){

  long row, col;
  long kperpdpsi, kpardpsi, sigsqshortmin;
  double rho, rho0, rhopow;
  double defocorrthresh, sigsqcorr, sigsqrho, sigsqrhoconst;
  double costscale;
  double nshortcycle, nshortcyclesq;
  float **dpsi, **avgdpsi;
  smoothcostT **costs, **rowcost, **colcost;

  /* get memory and set cost array pointers */
  costs=(smoothcostT **)Get2DRowColMem(nrow,ncol,sizeof(smoothcostT *),
				       sizeof(smoothcostT));
  rowcost=(smoothcostT **)costs;
  colcost=(smoothcostT **)&costs[nrow-1];

  /* set up */
  rho0=(params->rhosconst1)/(params->ncorrlooks)+(params->rhosconst2);
  defocorrthresh=params->defothreshfactor*rho0;
  rhopow=2*(params->cstd1)+(params->cstd2)*log(params->ncorrlooks)
    +(params->cstd3)*(params->ncorrlooks);
  sigsqrhoconst=2.0/12.0;
  sigsqcorr=params->sigsqcorr;
  sigsqshortmin=params->sigsqshortmin;
  kperpdpsi=params->kperpdpsi;
  kpardpsi=params->kpardpsi;
  costscale=params->costscale; 
  nshortcycle=params->nshortcycle;
  nshortcyclesq=nshortcycle*nshortcycle;

  /* get memory for wrapped difference arrays */
  dpsi=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
  avgdpsi=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));

  /* build array of mean wrapped phase differences in range */
  /* simple average of phase differences is biased, but mean phase */
  /*   differences usually near zero, so don't bother with complex average */
  fprintf(sp2,"Building range cost arrays\n");
  CalcWrappedRangeDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
			nrow,ncol);

  /* build colcost array (range slopes) */
  for(col=0;col<ncol-1;col++){
    for(row=0;row<nrow;row++){

      /* see if we have a masked pixel */
      if(mag[row][col]==0 || mag[row][col+1]==0){
	  
	/* masked pixel */
	colcost[row][col].offset=0;
	colcost[row][col].sigsq=LARGESHORT;

      }else{

	/* deformation-mode costs */
	
	/* calculate variance due to decorrelation */
	/* need symmetry for range if deformation */
	rho=(corr[row][col]+corr[row][col+1])/2.0;
	if(rho<defocorrthresh){
	  rho=0;
	}
	sigsqrho=(sigsqrhoconst*pow(1-rho,rhopow)+sigsqcorr)*nshortcyclesq;

	/* set cost paramaters in terms of flow, represented as shorts */
	if(rho>0){
	  colcost[row][col].offset=nshortcycle*
	    (dpsi[row][col]-avgdpsi[row][col]);
	}else{
	  colcost[row][col].offset=nshortcycle*
	    (dpsi[row][col]-0.5*avgdpsi[row][col]);       
	}
	colcost[row][col].sigsq=sigsqrho/(costscale*colweight[row][col]);
	if(colcost[row][col].sigsq<sigsqshortmin){
	  colcost[row][col].sigsq=sigsqshortmin;
	}
      }

      /* shift PDF to account for flattening by coarse unwrapped estimate */
      if(unwrappedest!=NULL){
	colcost[row][col].offset+=(nshortcycle/TWOPI*
				   (unwrappedest[row][col+1]
				    -unwrappedest[row][col]));
      }
    }
  }  /* end of range gradient cost calculation */

  /* build array of mean wrapped phase differences in azimuth */
  /* biased, but not much, so don't bother with complex averaging */
  fprintf(sp2,"Building azimuth cost arrays\n");
  CalcWrappedAzDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
		     nrow,ncol);

  /* build rowcost array */
  for(col=0;col<ncol;col++){
    for(row=0;row<nrow-1;row++){

      /* see if we have a masked pixel */
      if(mag[row][col]==0 || mag[row+1][col]==0){
	  
	/* masked pixel */
	rowcost[row][col].offset=0;
	rowcost[row][col].sigsq=LARGESHORT;

      }else{

	/* deformation-mode costs */

	/* variance due to decorrelation */
	/* get correlation and clip small values because of estimator bias */
	rho=(corr[row][col]+corr[row+1][col])/2.0;
	if(rho<defocorrthresh){
	  rho=0;
	}
        sigsqrho=(sigsqrhoconst*pow(1-rho,rhopow)+sigsqcorr)*nshortcyclesq;

	/* set cost paramaters in terms of flow, represented as shorts */
	if(rho>0){
	  rowcost[row][col].offset=nshortcycle*
	    (dpsi[row][col]-avgdpsi[row][col]);
	}else{
	  rowcost[row][col].offset=nshortcycle*
	    (dpsi[row][col]-0.5*avgdpsi[row][col]);
	}
	rowcost[row][col].sigsq=sigsqrho/(costscale*rowweight[row][col]);
	if(rowcost[row][col].sigsq<sigsqshortmin){
	  rowcost[row][col].sigsq=sigsqshortmin;
	}
      }

      /* shift PDF to account for flattening by coarse unwrapped estimate */
      if(unwrappedest!=NULL){
	rowcost[row][col].offset+=(nshortcycle/TWOPI*
				   (unwrappedest[row+1][col]
				    -unwrappedest[row][col]));
      }
    }
  } /* end of azimuth cost calculation */

  /* free temporary arrays */
  Free2DArray((void **)corr,nrow);
  Free2DArray((void **)dpsi,nrow);
  Free2DArray((void **)avgdpsi,nrow);

  /* return pointer to costs arrays */
  return((void **)costs);

}


/* function: GetIntensityAndCorrelation()
 * --------------------------------------
 * Reads amplitude and correlation info from files if specified.  If ampfile
 * not given, uses interferogram magnitude.  If correlation file not given,
 * generates correlatin info from interferogram and amplitude.
 */
void GetIntensityAndCorrelation(float **mag, float **wrappedphase, 
				float ***pwrptr, float ***corrptr, 
				infileT *infiles, long linelen, long nlines,
				long nrow, long ncol, outfileT *outfiles, 
				paramT *params, tileparamT *tileparams){

  long row, col, krowcorr, kcolcorr, iclipped;
  float **pwr, **corr;
  float **realcomp, **imagcomp;
  float **padreal, **padimag, **avgreal, **avgimag;
  float **pwr1, **pwr2, **padpwr1, **padpwr2, **avgpwr1, **avgpwr2; 
  double rho0, rhomin, biaseddefaultcorr;

  /* initialize */
  pwr=NULL;
  corr=NULL;
  pwr1=NULL;
  pwr2=NULL;
  
  /* read intensity, if specified */
  if(strlen(infiles->ampfile)){
    ReadIntensity(&pwr,&pwr1,&pwr2,infiles,linelen,nlines,params,tileparams);
  }else{
    if(params->costmode==TOPO){
      fprintf(sp1,"No brightness file specified.  ");
      fprintf(sp1,"Using interferogram magnitude as intensity\n");
    }
    pwr=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
	pwr[row][col]=mag[row][col];
      }
    }
  }

  /* read corrfile, if specified */
  if(strlen(infiles->corrfile)){
    ReadCorrelation(&corr,infiles,linelen,nlines,tileparams); 
  }else if(pwr1!=NULL && pwr2!=NULL && params->havemagnitude){

    /* generate the correlation info from the interferogram and amplitude */
    fprintf(sp1,"Generating correlation from interferogram and intensity\n");

    /* get the correct number of looks, and make sure its odd */
    krowcorr=1+2*floor(params->ncorrlooksaz/(double )params->nlooksaz/2);
    kcolcorr=1+2*floor(params->ncorrlooksrange/(double )params->nlooksrange/2);

    /* calculate equivalent number of independent looks */
    params->ncorrlooks=(kcolcorr*(params->dr/params->rangeres))
      *(krowcorr*(params->da/params->azres))*params->nlooksother;
    fprintf(sp1,"   (%.1f equivalent independent looks)\n",
	    params->ncorrlooks);
    
    /* get real and imaginary parts of interferogram */
    realcomp=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
    imagcomp=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
	realcomp[row][col]=mag[row][col]*cos(wrappedphase[row][col]);
	imagcomp[row][col]=mag[row][col]*sin(wrappedphase[row][col]);
      }
    }

    /* do complex spatial averaging on the interferogram */
    padreal=MirrorPad(realcomp,nrow,ncol,(krowcorr-1)/2,(kcolcorr-1)/2);
    padimag=MirrorPad(imagcomp,nrow,ncol,(krowcorr-1)/2,(kcolcorr-1)/2);
    if(padreal==realcomp || padimag==imagcomp){
      fprintf(sp0,"Correlation averaging box too large for input array size\n"
	      "Abort\n");
      exit(ABNORMAL_EXIT);
    }
    avgreal=realcomp;
    BoxCarAvg(avgreal,padreal,nrow,ncol,krowcorr,kcolcorr);
    avgimag=imagcomp;
    BoxCarAvg(avgimag,padimag,nrow,ncol,krowcorr,kcolcorr);
    Free2DArray((void **)padreal,nrow);
    Free2DArray((void **)padimag,nrow);

    /* spatially average individual SAR power images */
    padpwr1=MirrorPad(pwr1,nrow,ncol,(krowcorr-1)/2,(kcolcorr-1)/2);
    avgpwr1=pwr1;
    BoxCarAvg(avgpwr1,padpwr1,nrow,ncol,krowcorr,kcolcorr);
    padpwr2=MirrorPad(pwr2,nrow,ncol,(krowcorr-1)/2,(kcolcorr-1)/2);
    avgpwr2=pwr2;
    BoxCarAvg(avgpwr2,padpwr2,nrow,ncol,krowcorr,kcolcorr);
    Free2DArray((void **)padpwr1,nrow);
    Free2DArray((void **)padpwr2,nrow);

    /* build correlation data */
    corr=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
	if(avgpwr1[row][col]<=0 || avgpwr2[row][col]<=0){
	  corr[row][col]=0.0;
	}else{
	  corr[row][col]=sqrt((avgreal[row][col]*avgreal[row][col]
			       +avgimag[row][col]*avgimag[row][col])
			      /(avgpwr1[row][col]*avgpwr2[row][col]));
	}
      }
    }

    /* free temporary arrays */
    Free2DArray((void **)avgreal,nrow);
    Free2DArray((void **)avgimag,nrow);
    Free2DArray((void **)avgpwr1,nrow);
    Free2DArray((void **)avgpwr2,nrow);
    pwr1=NULL;
    pwr2=NULL;

  }else{

    /* no file specified: set corr to default value */
    /* find biased default correlation using */
    /* inverse of unbias method used by BuildCostArrays() */
    corr=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));
    fprintf(sp1,"No correlation file specified.  Assuming correlation = %g\n",
	   params->defaultcorr);
    rho0=(params->rhosconst1)/(params->ncorrlooks)+(params->rhosconst2);
    rhomin=params->rhominfactor*rho0;
    if(params->defaultcorr>rhomin){
      biaseddefaultcorr=params->defaultcorr;
    }else{
      biaseddefaultcorr=0.0;
    }
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
	corr[row][col]=biaseddefaultcorr;
      }
    }
  }

  /* dump correlation data if necessary */
  if(strlen(outfiles->rawcorrdumpfile)){
    Write2DArray((void **)corr,outfiles->rawcorrdumpfile,
		 nrow,ncol,sizeof(float)); 
  }

  /* check correlation data validity */
  iclipped=0;
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      if(!IsFinite(corr[row][col])){
	fprintf(sp0,"NaN or infinity found in correlation data\nAbort\n");
	exit(ABNORMAL_EXIT);
      }else if(corr[row][col]>1.0){
	if(corr[row][col]>1.001){
	  iclipped++;               /* don't warn for minor numerical errors */
	}
	corr[row][col]=1.0;
      }else if(corr[row][col]<0.0){
	if(corr[row][col]<-0.001){
	  iclipped++;               /* don't warn for minor numerical errors */
	}
	corr[row][col]=0.0;
      }
    }
  }
  if(iclipped){
    fprintf(sp0,"WARNING: %ld illegal correlation values clipped to [0,1]\n",
	    iclipped);
  }
  
  /* dump correlation data if necessary */
  if(strlen(outfiles->corrdumpfile)){
    Write2DArray((void **)corr,outfiles->corrdumpfile,
		 nrow,ncol,sizeof(float)); 
  }

  /* free memory and set output pointers */
  if(pwr1!=NULL){
    Free2DArray((void **)pwr1,nrow);
  }
  if(pwr2!=NULL){
    Free2DArray((void **)pwr2,nrow);
  }
  if(params->costmode==DEFO && pwr!=NULL){
    Free2DArray((void **)pwr,nrow);
    pwr=NULL;
  }
  *pwrptr=pwr;
  *corrptr=corr;

}


/* function: RemoveMean()
 * -------------------------
 * Divides intensity by average over sliding window.
 */
void RemoveMean(float **ei, long nrow, long ncol, 
		       long krowei, long kcolei){

  float **avgei, **padei;
  long row, col;

  /* make sure krowei, kcolei are odd */
  if(!(krowei % 2)){
    krowei++;
  }
  if(!(kcolei % 2)){
    kcolei++;
  }

  /* get memory */
  avgei=(float **)Get2DMem(nrow,ncol,sizeof(float *),sizeof(float));

  /* pad ei in new array */
  padei=MirrorPad(ei,nrow,ncol,(krowei-1)/2,(kcolei-1)/2);
  if(padei==ei){
    fprintf(sp0,"Intensity-normalization averaging box too large "
	    "for input array size\nAbort\n");
    exit(ABNORMAL_EXIT);
  }

  /* calculate average ei by using sliding window */
  BoxCarAvg(avgei,padei,nrow,ncol,krowei,kcolei);

  /* divide ei by avgei */
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      if(avgei){
	ei[row][col]/=(avgei[row][col]);
      }
    }
  }

  /* free memory */
  Free2DArray((void **)padei,nrow+krowei-1);
  Free2DArray((void **)avgei,nrow);

}


/* function: BuildDZRCritLookupTable()
 * -----------------------------------
 * Builds a 1-D lookup table of dzrcrit values indexed by incidence angle 
 * (in rad).
 */
float *BuildDZRCritLookupTable(double *nominc0ptr, double *dnomincptr, 
			       long *tablesizeptr, tileparamT *tileparams, 
			       paramT *params){

  long tablesize, k;
  double nominc, nominc0, nomincmax, dnominc;
  double a, re, slantrange;
  float *dzrcrittable;
  
  /* compute nominal spherical earth incidence angle for near and far range */
  a=params->orbitradius;
  re=params->earthradius;
  slantrange=params->nearrange+params->dr*tileparams->firstcol;
  nominc0=acos((a*a-slantrange*slantrange-re*re)/(2*slantrange*re));
  slantrange+=params->dr*tileparams->ncol;
  nomincmax=acos((a*a-slantrange*slantrange-re*re)/(2*slantrange*re));
  if(!IsFinite(nominc0) || !IsFinite(nomincmax)){
    fprintf(sp0,"Geometry error detected.  " 
	    "Check altitude, near range, and earth radius parameters\n"
	    "Abort\n");
    exit(ABNORMAL_EXIT);
  }

  /* build lookup table */
  dnominc=params->dnomincangle;
  tablesize=(long )floor((nomincmax-nominc0)/dnominc)+1;
  dzrcrittable=MAlloc(tablesize*sizeof(float));
  nominc=nominc0;
  for(k=0;k<tablesize;k++){
    dzrcrittable[k]=(float )SolveDZRCrit(sin(nominc),cos(nominc),params,
				 params->threshold);
    nominc+=dnominc;
    if(nominc>PI/2.0){
      nominc-=dnominc;
    }
  }
  
  /* set return variables */
  (*nominc0ptr)=nominc;
  (*dnomincptr)=dnominc;
  (*tablesizeptr)=tablesize;
  return(dzrcrittable);

}


/* function: SolveDZRCrit()
 * ------------------------
 * Numerically solve for the transition point of the linearized scattering 
 * model.
 */
double SolveDZRCrit(double sinnomincangle, double cosnomincangle, 
		    paramT *params, double threshold){

  double residual, thetai, kds, n, dr, dzr, dx;
  double costhetai, cos2thetai, step;
  double dzrcritfactor, diffuse, specular;
  long i;

  /* get parameters */
  kds=params->kds;
  n=params->specularexp;
  dr=params->dr;  
  dzrcritfactor=params->dzrcritfactor;

  /* solve for critical incidence angle */
  thetai=PI/4;
  step=PI/4-1e-6;
  i=0;
  while(TRUE){
    if((cos2thetai=cos(2*thetai))<0){
      cos2thetai=0;
    }
    diffuse=dzrcritfactor*kds*cos(thetai);
    specular=pow(cos2thetai,n);
    if(fabs(residual=diffuse-specular)<threshold*diffuse){
      break;
    }
    if(residual<0){
      thetai+=step;
    }else{
      thetai-=step;
    }
    step/=2.0;
    if(++i>MAXITERATION){
      fprintf(sp0,"Couldn't find critical incidence angle ");
      fprintf(sp0,"(check scattering parameters)\nAbort\n");
      exit(ABNORMAL_EXIT);
    }
  }

  /* solve for critical height change */
  costhetai=cos(thetai);
  dzr=params->initdzr;
  step=dzr+dr*cosnomincangle-1e-2;
  i=0;
  while(TRUE){
    dx=(dr+dzr*cosnomincangle)/sinnomincangle;
    if(fabs(residual=costhetai-(dzr*sinnomincangle+dx*cosnomincangle)
	    /sqrt(dzr*dzr+dx*dx))
       <threshold*costhetai){
      return(dzr);
    }
    if(residual<0){
      dzr-=step;
    }else{
      dzr+=step;
    }
    step/=2.0;
    if(++i>MAXITERATION){
      fprintf(sp0,"Couldn't find critical slope ");
      fprintf(sp0,"(check geometry parameters)\nAbort\n");
      exit(ABNORMAL_EXIT);
    }
  }
}


/* function: SolveEIModelParams()
 * ------------------------------
 * Calculates parameters for linearized model of EI vs. range slope
 * relationship.
 */
void SolveEIModelParams(double *slope1ptr, double *slope2ptr, 
			double *const1ptr, double *const2ptr, 
			double dzrcrit, double dzr0, double sinnomincangle, 
			double cosnomincangle, paramT *params){

  double slope1, slope2, const1, const2, sloperatio;
  double dzr3, ei3;
  
  /* set up */
  sloperatio=params->kds*params->sloperatiofactor;

  /* find normalized intensity at 15(dzrcrit-dzr0)+dzr0 */
  dzr3=15.0*(dzrcrit-dzr0)+dzr0;
  ei3=EIofDZR(dzr3,sinnomincangle,cosnomincangle,params)
    /EIofDZR(0,sinnomincangle,cosnomincangle,params);

  /* calculate parameters */
  const1=dzr0;
  slope2=(sloperatio*(dzrcrit-const1)-dzrcrit+dzr3)/ei3;
  slope1=slope2/sloperatio;
  const2=dzr3-slope2*ei3;

  /* set return values */
  *slope1ptr=slope1;
  *slope2ptr=slope2;
  *const1ptr=const1;
  *const2ptr=const2;

}


/* function: EIofDZR()
 * -------------------
 * Calculates expected value of intensity with arbitrary units for given
 * parameters.  Assumes azimuth slope is zero.
 */
double EIofDZR(double dzr, double sinnomincangle, double cosnomincangle,
	       paramT *params){

  double dr, da, dx, kds, n, dzr0, projarea;
  double costhetai, cos2thetai, sigma0;

  dr=params->dr;
  da=params->da;
  dx=dr/sinnomincangle+dzr*cosnomincangle/sinnomincangle;
  kds=params->kds;
  n=params->specularexp;
  dzr0=-dr*cosnomincangle;
  projarea=da*fabs((dzr-dzr0)/sinnomincangle);
  costhetai=projarea/sqrt(dzr*dzr*da*da + da*da*dx*dx);
  if(costhetai>SQRTHALF){
    cos2thetai=2*costhetai*costhetai-1;
    sigma0=kds*costhetai+pow(cos2thetai,n);
  }else{
    sigma0=kds*costhetai;
  }
  return(sigma0*projarea);

}


/* function: BuildDZRhoMaxLookupTable()
 * ------------------------------------
 * Builds a 2-D lookup table of dzrhomax values vs nominal incidence angle
 * (rad) and correlation.
 */
float **BuildDZRhoMaxLookupTable(double nominc0, double dnominc, 
				 long nominctablesize, double rhomin, 
				 double drho, long nrho, paramT *params){

  long krho, knominc;
  double nominc, rho;
  float **dzrhomaxtable;

  dzrhomaxtable=(float **)Get2DMem(nominctablesize,nrho,
				   sizeof(float *),sizeof(float));
  nominc=nominc0;
  for(knominc=0;knominc<nominctablesize;knominc++){
    rho=rhomin;
    for(krho=0;krho<nrho;krho++){
      dzrhomaxtable[knominc][krho]=(float )CalcDZRhoMax(rho,nominc,params,
							params->threshold);
      rho+=drho;
    }
    nominc+=dnominc;
  }
  return(dzrhomaxtable);

}


/* function: CalcDZRhoMax()
 * ------------------------
 * Calculates the maximum slope (in range) for the given unbiased correlation
 * using spatial decorrelation as an upper limit (Zebker & Villasenor,
 * 1992).
 */
double CalcDZRhoMax(double rho, double nominc, paramT *params, 
		    double threshold){

  long i;
  double dx, dr, dz, dzstep, rhos, sintheta, costheta, numerator;
  double a, re, bperp, slantrange, lookangle;
  double costhetairsq, rhosfactor, residual;


  /* set up */
  i=0;
  dr=params->dr;
  costheta=cos(nominc);
  sintheta=sin(nominc);
  dzstep=params->initdzstep;
  a=params->orbitradius;
  re=params->earthradius;
  lookangle=asin(re/a*sintheta);
  bperp=params->baseline*cos(lookangle-params->baselineangle);
  slantrange=sqrt(a*a+re*re-2*a*re*cos(nominc-lookangle));
  rhosfactor=2.0*fabs(bperp)*(params->rangeres)/((params->lambda)*slantrange);

  /* take care of the extremes */
  if(rho>=1.0){
    return(-dr*costheta);
  }else if(rho<=0){
    return(LARGEFLOAT);
  }

  /* start with slope for unity correlation, step slope upwards */
  dz=-dr*costheta;
  rhos=1.0;
  while(rhos>rho){
    dz+=dzstep;
    dx=(dr+dz*costheta)/sintheta;
    numerator=dz*sintheta+dx*costheta;
    costhetairsq=numerator*numerator/(dz*dz+dx*dx);
    rhos=1-rhosfactor*sqrt(costhetairsq/(1-costhetairsq));
    if(rhos<0){
      rhos=0;
    }
    if(dz>BIGGESTDZRHOMAX){
      return(BIGGESTDZRHOMAX);
    }
  }

  /* now iteratively decrease step size and narrow in on correct slope */
  while(fabs(residual=rhos-rho)>threshold*rho){
    dzstep/=2.0;
    if(residual<0){
      dz-=dzstep;
    }else{
      dz+=dzstep;
    }
    dx=(dr+dz*costheta)/sintheta;
    numerator=dz*sintheta+dx*costheta;
    costhetairsq=numerator*numerator/(dz*dz+dx*dx);
    rhos=1-rhosfactor*sqrt(costhetairsq/(1-costhetairsq));
    if(rhos<0){
      rhos=0;
    }
    if(++i>MAXITERATION){
      fprintf(sp0,"Couldn't find slope for correlation of %f\n",rho);
      fprintf(sp0,"(check geometry and spatial decorrelation parameters)\n");
      fprintf(sp0,"Abort\n");
      exit(ABNORMAL_EXIT);
    }
  }

  return(dz);
}


/* function: CalcCostTopo()
 * ------------------------
 * Calculates topography arc distance given an array of cost data structures.
 */
void CalcCostTopo(void **costs, long flow, long arcrow, long arccol, 
		  long nflow, long nrow, paramT *params, 
		  long *poscostptr, long *negcostptr){

  long idz1, idz2pos, idz2neg, cost1, nflowsq, poscost, negcost;
  long nshortcycle, layfalloffconst;
  long offset, sigsq, laycost, dzmax;
  costT *cost;


  /* get arc info */
  cost=&((costT **)(costs))[arcrow][arccol];
  dzmax=cost->dzmax;
  offset=cost->offset;
  sigsq=cost->sigsq;
  laycost=cost->laycost;
  nshortcycle=params->nshortcycle;
  layfalloffconst=params->layfalloffconst;
  if(arcrow<nrow-1){

    /* row cost: dz symmetric with respect to origin */
    idz1=labs(flow*nshortcycle+offset);
    idz2pos=labs((flow+nflow)*nshortcycle+offset);
    idz2neg=labs((flow-nflow)*nshortcycle+offset);

  }else{

    /* column cost: non-symmetric dz */
    /* dzmax will only be < 0 if we have a column arc */
    if(dzmax<0){
      dzmax*=-1;
      idz1=-(flow*nshortcycle+offset);
      idz2pos=-((flow+nflow)*nshortcycle+offset);
      idz2neg=-((flow-nflow)*nshortcycle+offset);
    }else{
      idz1=flow*nshortcycle+offset;
      idz2pos=(flow+nflow)*nshortcycle+offset;
      idz2neg=(flow-nflow)*nshortcycle+offset;
    }

  }

  /* calculate cost1 */
  if(idz1>dzmax){
    idz1-=dzmax;
    cost1=(idz1*idz1)/(layfalloffconst*sigsq)+laycost; 
  }else{
    cost1=(idz1*idz1)/sigsq;
    if(laycost!=NOCOSTSHELF && idz1>0 && cost1>laycost){
      cost1=laycost;
    }
  }

  /* calculate positive cost increment */
  if(idz2pos>dzmax){
    idz2pos-=dzmax;
    poscost=(idz2pos*idz2pos)/(layfalloffconst*sigsq)
      +laycost-cost1;
  }else{
    poscost=(idz2pos*idz2pos)/sigsq;
    if(laycost!=NOCOSTSHELF && idz2pos>0 && poscost>laycost){
      poscost=laycost-cost1;
    }else{
      poscost-=cost1;
    }
  }

  /* calculate negative cost increment */
  if(idz2neg>dzmax){
    idz2neg-=dzmax;
    negcost=(idz2neg*idz2neg)/(layfalloffconst*sigsq)
      +laycost-cost1;
  }else{
    negcost=(idz2neg*idz2neg)/sigsq;
    if(laycost!=NOCOSTSHELF && idz2neg>0 && negcost>laycost){
      negcost=laycost-cost1;
    }else{
      negcost-=cost1;
    }
  }

  /* scale costs for this nflow */
  nflowsq=nflow*nflow;
  if(poscost>0){
    *poscostptr=(long )ceil((float )poscost/nflowsq);
  }else{
    *poscostptr=(long )floor((float )poscost/nflowsq);
  }
  if(negcost>0){
    *negcostptr=(long )ceil((float )negcost/nflowsq);
  }else{
    *negcostptr=(long )floor((float )negcost/nflowsq);
  }

}


/* function: CalcCostDefo()
 * ------------------------
 * Calculates deformation arc distance given an array of cost data structures.
 */
void CalcCostDefo(void **costs, long flow, long arcrow, long arccol, 
		  long nflow, long nrow, paramT *params, 
		  long *poscostptr, long *negcostptr){

  long idz1, idz2pos, idz2neg, cost1, nflowsq, poscost, negcost;
  long nshortcycle, layfalloffconst;
  costT *cost;


  /* get arc info */
  cost=&((costT **)(costs))[arcrow][arccol];
  nshortcycle=params->nshortcycle;
  layfalloffconst=params->layfalloffconst;
  idz1=labs(flow*nshortcycle+cost->offset);
  idz2pos=labs((flow+nflow)*nshortcycle+cost->offset);
  idz2neg=labs((flow-nflow)*nshortcycle+cost->offset);

  /* calculate cost1 */
  if(idz1>cost->dzmax){
    idz1-=cost->dzmax;
    cost1=(idz1*idz1)/(layfalloffconst*(cost->sigsq))+cost->laycost; 
  }else{
    cost1=(idz1*idz1)/cost->sigsq;
    if(cost->laycost!=NOCOSTSHELF && cost1>cost->laycost){
      cost1=cost->laycost;
    }
  }

  /* calculate positive cost increment */
  if(idz2pos>cost->dzmax){
    idz2pos-=cost->dzmax;
    poscost=(idz2pos*idz2pos)/(layfalloffconst*(cost->sigsq))
      +cost->laycost-cost1;
  }else{
    poscost=(idz2pos*idz2pos)/cost->sigsq;
    if(cost->laycost!=NOCOSTSHELF && poscost>cost->laycost){
      poscost=cost->laycost-cost1;
    }else{
      poscost-=cost1;
    }
  }

  /* calculate negative cost increment */
  if(idz2neg>cost->dzmax){
    idz2neg-=cost->dzmax;
    negcost=(idz2neg*idz2neg)/(layfalloffconst*(cost->sigsq))
      +cost->laycost-cost1;
  }else{
    negcost=(idz2neg*idz2neg)/cost->sigsq;
    if(cost->laycost!=NOCOSTSHELF && negcost>cost->laycost){
      negcost=cost->laycost-cost1;
    }else{
      negcost-=cost1;
    }
  }

  /* scale costs for this nflow */
  nflowsq=nflow*nflow;
  if(poscost>0){
    *poscostptr=(long )ceil((float )poscost/nflowsq);
  }else{
    *poscostptr=(long )floor((float )poscost/nflowsq);
  }
  if(negcost>0){
    *negcostptr=(long )ceil((float )negcost/nflowsq);
  }else{
    *negcostptr=(long )floor((float )negcost/nflowsq);
  }

}


/* function: CalcCostSmooth()
 * --------------------------
 * Calculates smooth-solution arc distance given an array of smoothcost
 *  data structures.
 */
void CalcCostSmooth(void **costs, long flow, long arcrow, long arccol, 
		    long nflow, long nrow, paramT *params, 
		    long *poscostptr, long *negcostptr){

  long idz1, idz2pos, idz2neg, cost1, nflowsq, poscost, negcost;
  long nshortcycle;
  smoothcostT *cost;


  /* get arc info */
  cost=&((smoothcostT **)(costs))[arcrow][arccol];
  nshortcycle=params->nshortcycle;
  idz1=labs(flow*nshortcycle+cost->offset);
  idz2pos=labs((flow+nflow)*nshortcycle+cost->offset);
  idz2neg=labs((flow-nflow)*nshortcycle+cost->offset);

  /* calculate cost1 */
  cost1=(idz1*idz1)/cost->sigsq;

  /* calculate positive cost increment */
  poscost=(idz2pos*idz2pos)/cost->sigsq-cost1;

  /* calculate negative cost increment */
  negcost=(idz2neg*idz2neg)/cost->sigsq-cost1;

  /* scale costs for this nflow */
  nflowsq=nflow*nflow;
  if(poscost>0){
    *poscostptr=(long )ceil((float )poscost/nflowsq);
  }else{
    *poscostptr=(long )floor((float )poscost/nflowsq);
  }
  if(negcost>0){
    *negcostptr=(long )ceil((float )negcost/nflowsq);
  }else{
    *negcostptr=(long )floor((float )negcost/nflowsq);
  }

}


/* function: CalcCostL0()
 * ----------------------
 * Calculates the L0 arc distance given an array of short integer weights.
 */
void CalcCostL0(void **costs, long flow, long arcrow, long arccol, 
		long nflow, long nrow, paramT *params, 
		long *poscostptr, long *negcostptr){

  /* L0-norm */
  if(flow){
    if(flow+nflow){
      *poscostptr=0;
    }else{
      *poscostptr=-((short **)costs)[arcrow][arccol];
    }
    if(flow-nflow){
      *negcostptr=0;
    }else{
      *negcostptr=-((short **)costs)[arcrow][arccol];
    }
  }else{
    *poscostptr=((short **)costs)[arcrow][arccol];
    *negcostptr=((short **)costs)[arcrow][arccol];
  }
}


/* function: CalcCostL1()
 * ----------------------
 * Calculates the L1 arc distance given an array of short integer weights.
 */
void CalcCostL1(void **costs, long flow, long arcrow, long arccol, 
		long nflow, long nrow, paramT *params, 
		long *poscostptr, long *negcostptr){

  /* L1-norm */
  *poscostptr=((short **)costs)[arcrow][arccol]*(labs(flow+nflow)-labs(flow));
  *negcostptr=((short **)costs)[arcrow][arccol]*(labs(flow-nflow)-labs(flow));

}


/* function: CalcCostL2()
 * ----------------------
 * Calculates the L2 arc distance given an array of short integer weights.
 */
void CalcCostL2(void **costs, long flow, long arcrow, long arccol, 
		long nflow, long nrow, paramT *params, 
		long *poscostptr, long *negcostptr){

  long flow2, flowsq;

  /* L2-norm */
  flowsq=flow*flow;
  flow2=flow+nflow;
  *poscostptr=((short **)costs)[arcrow][arccol]*(flow2*flow2-flowsq);
  flow2=flow-nflow;
  *negcostptr=((short **)costs)[arcrow][arccol]*(flow2*flow2-flowsq);
}


/* function: CalcCostLP()
 * ----------------------
 * Calculates the Lp arc distance given an array of short integer weights.
 */
void CalcCostLP(void **costs, long flow, long arcrow, long arccol, 
		long nflow, long nrow, paramT *params, 
		long *poscostptr, long *negcostptr){

  long p;
  short flow2;

  /* Lp-norm */
  flow2=flow+nflow;
  p=params->p;
  *poscostptr=LRound(((short **)costs)[arcrow][arccol]*
		     (pow(labs(flow2),p)-pow(labs(flow),p)));
  flow2=flow-nflow;
  *negcostptr=LRound(((short **)costs)[arcrow][arccol]*
		     (pow(labs(flow2),p)-pow(labs(flow),p)));
}


/* function: CalcCostNonGrid()
 * ---------------------------
 * Calculates the arc cost given an array of long integer cost lookup tables.
 */
void CalcCostNonGrid(void **costs, long flow, long arcrow, long arccol, 
		     long nflow, long nrow, paramT *params, 
		     long *poscostptr, long *negcostptr){

  long xflow, flowmax, poscost, negcost, nflowsq, arroffset, sumsigsqinv;
  long abscost0;
  long *costarr;
  float c1;


  /* set up */
  flowmax=params->scndryarcflowmax;
  costarr=((long ***)costs)[arcrow][arccol];
  arroffset=costarr[0];
  sumsigsqinv=costarr[2*flowmax+1];

  /* return zero costs if this is a zero cost arc */
  if(sumsigsqinv==ZEROCOSTARC){
    *poscostptr=0;
    *negcostptr=0;
    return;
  }

  /* compute cost of current flow */
  xflow=flow+arroffset;
  if(xflow>flowmax){
    c1=costarr[flowmax]/(float )flowmax-sumsigsqinv*flowmax;
    abscost0=(sumsigsqinv*xflow+LRound(c1))*xflow;
  }else if(xflow<-flowmax){
    c1=costarr[2*flowmax]/(float )flowmax-sumsigsqinv*flowmax;
    abscost0=(sumsigsqinv*xflow+LRound(c1))*xflow;
  }else{
    if(xflow>0){
      abscost0=costarr[xflow];
    }else if(xflow<0){
      abscost0=costarr[flowmax-xflow];  
    }else{
      abscost0=0;
    }
  }

  /* compute costs of positive and negative flow increments */
  xflow=flow+arroffset+nflow;
  if(xflow>flowmax){
    c1=costarr[flowmax]/(float )flowmax-sumsigsqinv*flowmax;    
    poscost=((sumsigsqinv*xflow+LRound(c1))*xflow)-abscost0;
  }else if(xflow<-flowmax){
    c1=costarr[2*flowmax]/(float )flowmax-sumsigsqinv*flowmax;    
    poscost=((sumsigsqinv*xflow+LRound(c1))*xflow)-abscost0;
  }else{
    if(xflow>0){
      poscost=costarr[xflow]-abscost0;
    }else if(xflow<0){
      poscost=costarr[flowmax-xflow]-abscost0;
    }else{
      poscost=-abscost0;
    }
  }
  xflow=flow+arroffset-nflow;
  if(xflow>flowmax){
    c1=costarr[flowmax]/(float )flowmax-sumsigsqinv*flowmax;    
    negcost=((sumsigsqinv*xflow+LRound(c1))*xflow)-abscost0;
  }else if(xflow<-flowmax){
    c1=costarr[2*flowmax]/(float )flowmax-sumsigsqinv*flowmax;    
    negcost=((sumsigsqinv*xflow+LRound(c1))*xflow)-abscost0;
  }else{
    if(xflow>0){
      negcost=costarr[xflow]-abscost0;
    }else if(xflow<0){
      negcost=costarr[flowmax-xflow]-abscost0;
    }else{
      negcost=-abscost0;
    }
  }

  /* scale for this flow increment and set output values */
  nflowsq=nflow*nflow;
  if(poscost>0){
    *poscostptr=(long )ceil((float )poscost/nflowsq);
  }else{
    *poscostptr=(long )floor((float )poscost/nflowsq);
  }
  if(negcost>0){
    *negcostptr=(long )ceil((float )negcost/nflowsq);
  }else{
    *negcostptr=(long )floor((float )negcost/nflowsq);
  }

}


/* function: EvalCostTopo()
 * ------------------------
 * Calculates topography arc cost given an array of cost data structures.
 */
long EvalCostTopo(void **costs, short **flows, long arcrow, long arccol,
		  long nrow, paramT *params){

  long idz1, cost1, dzmax;
  costT *cost;

  /* get arc info */
  cost=&((costT **)(costs))[arcrow][arccol];
  if(arcrow<nrow-1){

    /* row cost: dz symmetric with respect to origin */
    idz1=labs(flows[arcrow][arccol]*(params->nshortcycle)+cost->offset);
    dzmax=cost->dzmax;

  }else{

    /* column cost: non-symmetric dz */
    idz1=flows[arcrow][arccol]*(params->nshortcycle)+cost->offset;
    if((dzmax=cost->dzmax)<0){
      idz1*=-1;
      dzmax*=-1;
    }

  }

  /* calculate and return cost */
  if(idz1>dzmax){
    idz1-=dzmax;
    cost1=(idz1*idz1)/((params->layfalloffconst)*(cost->sigsq))+cost->laycost;
  }else{
    cost1=(idz1*idz1)/cost->sigsq;
    if(cost->laycost!=NOCOSTSHELF && idz1>0 && cost1>cost->laycost){
      cost1=cost->laycost;
    }
  }
  return(cost1);
}


/* function: EvalCostDefo()
 * ------------------------
 * Calculates deformation arc cost given an array of cost data structures.
 */
long EvalCostDefo(void **costs, short **flows, long arcrow, long arccol,
		  long nrow, paramT *params){

  long idz1, cost1;
  costT *cost;

  /* get arc info */
  cost=&((costT **)(costs))[arcrow][arccol];
  idz1=labs(flows[arcrow][arccol]*(params->nshortcycle)+cost->offset);

  /* calculate and return cost */
  if(idz1>cost->dzmax){
    idz1-=cost->dzmax;
    cost1=(idz1*idz1)/((params->layfalloffconst)*(cost->sigsq))+cost->laycost; 
  }else{
    cost1=(idz1*idz1)/cost->sigsq;
    if(cost->laycost!=NOCOSTSHELF && cost1>cost->laycost){
      cost1=cost->laycost;
    }
  }
  return(cost1);
}


/* function: EvalCostSmooth()
 * --------------------------
 * Calculates smooth-solution arc cost given an array of 
 * smoothcost data structures.
 */
long EvalCostSmooth(void **costs, short **flows, long arcrow, long arccol,
		    long nrow, paramT *params){

  long idz1;
  smoothcostT *cost;

  /* get arc info */
  cost=&((smoothcostT **)(costs))[arcrow][arccol];
  idz1=labs(flows[arcrow][arccol]*(params->nshortcycle)+cost->offset);

  /* calculate and return cost */
  return((idz1*idz1)/cost->sigsq);

}


/* function: EvalCostL0()
 * ----------------------
 * Calculates the L0 arc cost given an array of cost data structures.
 */
long EvalCostL0(void **costs, short **flows, long arcrow, long arccol, 
		long nrow, paramT *params){

  /* L0-norm */
  if(flows[arcrow][arccol]){
    return((long)((short **)costs)[arcrow][arccol]);
  }else{
    return(0);
  }
}


/* function: EvalCostL1()
 * ----------------------
 * Calculates the L1 arc cost given an array of cost data structures.
 */
long EvalCostL1(void **costs, short **flows, long arcrow, long arccol, 
		long nrow, paramT *params){

  /* L1-norm */
  return( (((short **)costs)[arcrow][arccol]) * labs(flows[arcrow][arccol]) );
}


/* function: EvalCostL2()
 * ----------------------
 * Calculates the L2 arc cost given an array of cost data structures.
 */
long EvalCostL2(void **costs, short **flows, long arcrow, long arccol, 
		long nrow, paramT *params){

  /* L2-norm */
  return( (((short **)costs)[arcrow][arccol]) * 
	  (flows[arcrow][arccol]*flows[arcrow][arccol]) );
}


/* function: EvalCostLP()
 * ----------------------
 * Calculates the Lp arc cost given an array of cost data structures.
 */
long EvalCostLP(void **costs, short **flows, long arcrow, long arccol, 
		long nrow, paramT *params){

  /* Lp-norm */
  return( (((short **)costs)[arcrow][arccol]) * 
	  pow(labs(flows[arcrow][arccol]),params->p) );
}


/* function: EvalCostNonGrid()
 * ---------------------------
 * Calculates the arc cost given an array of long integer cost lookup tables.
 */
long EvalCostNonGrid(void **costs, short **flows, long arcrow, long arccol, 
		     long nrow, paramT *params){

  long flow, xflow, flowmax, arroffset, sumsigsqinv;
  long *costarr;
  float c1;

  /* set up */
  flow=flows[arcrow][arccol];
  flowmax=params->scndryarcflowmax;
  costarr=((long ***)costs)[arcrow][arccol];
  arroffset=costarr[0];
  sumsigsqinv=costarr[2*flowmax+1];

  /* return zero costs if this is a zero cost arc */
  if(sumsigsqinv==ZEROCOSTARC){
    return(0);
  }

  /* compute cost of current flow */
  xflow=flow+arroffset;
  if(xflow>flowmax){
    c1=costarr[flowmax]/(float )flowmax-sumsigsqinv*flowmax;
    return((sumsigsqinv*xflow+LRound(c1))*xflow);
  }else if(xflow<-flowmax){
    c1=costarr[2*flowmax]/(float )flowmax-sumsigsqinv*flowmax;
    return((sumsigsqinv*xflow+LRound(c1))*xflow);
  }else{
    if(xflow>0){
      return(costarr[xflow]);
    }else if(xflow<0){
      return(costarr[flowmax-xflow]);
    }else{
      return(0);
    }
  }
}


/* function: CalcInitMaxFlow()
 * ---------------------------
 * Calculates the maximum flow magnitude to allow in the initialization
 * by examining the dzmax members of arc statistical cost data structures.
 */
void CalcInitMaxFlow(paramT *params, void **costs, long nrow, long ncol){

  long row, col, maxcol, initmaxflow, arcmaxflow;

  if(params->initmaxflow<=0){
    if(params->costmode==NOSTATCOSTS){
      params->initmaxflow=NOSTATINITMAXFLOW;
    }else{
      if(params->costmode==TOPO || params->costmode==DEFO){
	initmaxflow=0;
	for(row=0;row<2*nrow-1;row++){
	  if(row<nrow-1){
	    maxcol=ncol;
	  }else{
	    maxcol=ncol-1;
	  }
	  for(col=0;col<maxcol;col++){
	    if(((costT **)costs)[row][col].dzmax!=LARGESHORT){
	      arcmaxflow=ceil(labs((long )((costT **)costs)[row][col].dzmax)/
			      (double )(params->nshortcycle)
			      +params->arcmaxflowconst);
	      if(arcmaxflow>initmaxflow){
		initmaxflow=arcmaxflow;
	      }
	    }
	  }
	}
	params->initmaxflow=initmaxflow;
      }else{
	params->initmaxflow=DEF_INITMAXFLOW;
      }
    }
  }
}
