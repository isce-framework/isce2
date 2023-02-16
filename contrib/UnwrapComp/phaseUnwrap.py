#!/usr/bin/env python3

from __future__ import division

import argparse, pdb
import numpy as np
import os

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree
from scipy.spatial import Delaunay, ConvexHull

import pulp
import timeit as T

class Vertex(object):
    '''
    Defines vertex.
    '''

    def __init__(self, x=None, y=None, phase=None, compNumber=None, index=None):
        self.x          = x
        self.y          = y
        self.phase      = phase
        self.compNumber = compNumber
        self.index      = index
        self.pts        = None
        self.sigma      = None
        self.source     = None
        self.dist       = None
        self.n          = None

    def __str__(self):
        ostr = 'Location: (%d, %d)'%(self.y, self.x)
        #ostr += '\nComponent: (%d, %d)'%(self.source, self.compNumber)
        return ostr

    def __eq__(self, other):
#        if other is not None:
        try:
            return (self.x == other.x) and (self.y == other.y)
        except:
            pass
        return None

    def __hash__(self):
        return hash((self.x,self.y))

    def updatePhase(self, n):
        self.n = n

    def getIndex(self):
        return self.index

    def getPhase(self):
        return self.n

    def getUnwrappedPhase(self):
        return self.phase - 2 * self.n * np.pi

class Edge(object):
    '''
    Defines edge of Delaunay Triangulation.
    '''

    def __init__(self, source, dest, dist=1, name=None):
        self.__CONST = 10000
        self.__MAXFLOW = 100
        self.src = source
        self.dst = dest
        self.triIdx = None
        self.adjtriIdx = None
        self.flow = None
        self.cost = self.__computeCost()
        self.__dist__ = dist

        if name is None:
          name = "E(%d,%d)"%(source.getIndex(), dest.getIndex())

        # Using PuLP to define the variable
        self.__var__ = pulp.LpVariable(name, 0, 1, pulp.LpContinuous)

    def isBarrier(self):
        if None not in (self.src.compNumber, self.dst.compNumber):
          return (self.src.compNumber == self.dst.compNumber)
        else:
          return False

    def isCorner(self):
        return self.adjtriIdx == None

    def __computeCost(self):
        if self.isBarrier():
            return self.__CONST
        else:
            return 1

    def isUnwrapped(self):
        return None not in (self.src.getPhase(), self.dst.getPhase())

    def diff(self):
        return int(np.round((self.dst.phase - self.src.phase)/(2*np.pi)))

    def updateTri(self, index):
        if self.triIdx is not None:
          self.triIdx.append(index)
        else:
          self.triIdx = [index]

    def updateAdj(self, index):
        if self.adjtriIdx is not None:
          self.adjtriIdx.append(index)
        else:
          self.adjtriIdx = [index]

    def updateFlow(self, flow):
        if self.isBarrier():
          # Check if the solver's solution for high cost node is zero
          if (flow != 0):
            raise ValueError("Solver Solution Incorrect")
        self.flow = flow

    def unwrap(self, rEdge):
        if self.src.getPhase() is not None:
            self.dst.updatePhase(self.src.getPhase() + (self.flow - rEdge.flow) + self.diff())
        else:
            return None

    def plot(self, plt, c='g'):
        plt.plot([self.src.x, self.dst.x], [self.src.y, self.dst.y], c=c)

        if self.flow != 0 and self.isBarrier():
            plt.plot([self.src.x, self.dst.x], [self.src.y, self.dst.y], c=c)

    def getFlow(self, neutralNode):
        if self.adjtriIdx == None:
            return 'a %d %d 0 %d %d\n' % (neutralNode, self.triIdx[0], self.__MAXFLOW, self.cost)
        else:
            return 'a %d %d 0 %d %d\n' % (self.adjtriIdx[0], self.triIdx[0], self.__MAXFLOW, self.cost)

    def getCost(self):
        return self.cost

    def getNeutralWeight(self):
        # Number of times the edge was used with negative weight
        if self.adjtriIdx is None:
          numReverseLoop = 0
        else:
          numReverseLoop = len(self.adjtriIdx)

        # Number of times the edge was used with positive weight
        if self.triIdx is None:
          numLoop = 0
        else:
          numLoop = len(self.triIdx)

        return numReverseLoop - numLoop

    def getLPVar(self):
        return self.__var__

    def sign(self):
        if self.src.compNumber < self.dst.compNumber:
            return 1
        elif self.src.compNumber > self.dst.compNumber:
            return -1
        else:
            return 0

    def __str__(self):
        ostr = 'Edge between : \n'
        ostr += str(self.src) + '\n'
        ostr += str(self.dst) + '\n'
        ostr += 'with Residue %f'%(self.flow)
        return ostr

    def __eq__(self, other):
        return (self.src == other.src) and (self.dst == other.dst)

    def __neg__(self):
        return Edge(self.dst, self.src)

# Defines each delaunay traingle
class Loop(object):
    '''
    Collection of edges in loop - Expects the vertices to be in sequence
    '''
    def __init__(self, vertices=None, edges=None, index=None):
        self.edges = None
        self.index = None
        self.residue = None

        # Reverse edges that are used during unwrapping - Done contribute
        # in the residue computation
        self.__edges = None
        self.__center = None

        def_vertices = True
        if 'any' not in dir(vertices) and vertices == None:
            def_vertices = False

        def_edges = True
        if 'any' not in dir(edges) and edges == None:
            def_edges = False

        def_index = True
        if 'any' not in dir(index) and index == None:
            def_index = False

        # initializes only when all the vertices are availables
#        if None not in (vertices, edges, index):
        if def_vertices and def_edges and def_index:
            self.index = index
            self.edges = self.__getEdges(vertices, edges)
            self.__edges = self.__getReverseEdges(vertices, edges)
            self.residue = self.computeResidue()
            self.__updateEdges()
            self.__center = self.__getCenter(vertices)

    # Edges traversing the vertices in a sequence
    @staticmethod
    def __getReverseEdges(vertices, edges):
        rSeqEdges = []
        for vx, vy in zip(vertices[1:] + [vertices[0]], vertices):
          rSeqEdges.append(edges[vx, vy])
        return rSeqEdges

    # Edges traversing the vertices in a sequence
    @staticmethod
    def __getEdges(vertices, edges):
        seqEdges = []
        for vx, vy in zip(vertices, vertices[1:] + [vertices[0]]):
          seqEdges.append(edges[vx, vy])
        return seqEdges

    # Returns a string in the RelaxIV format
    def getNodeSupply(self):
        return "n %d %d\n"%(self.index, self.residue)

    # Returns a string in the RelaxIV format for flows
    def getFlowConstraint(self, neutralNode):
        edgeFlow = []
        for edge in self.edges:
            edgeFlow.append(edge.getFlow(neutralNode))
        return edgeFlow

    def getLPFlowConstraint(self):
        edgeFlow = []
        lpConstraint = []
        for edge, rEdge in zip(self.edges, self.__edges):
            lpConstraint.extend([edge.getLPVar(), -(rEdge.getLPVar())])
        return (pulp.lpSum(lpConstraint) == -self.residue)

    # Updates the cost of the edges
    def updateEdgeFlow(self, flow):
        for cost, edge in zip(flow, self.edges):
            edge.updateFlow(cost)

    # Computes the Residue in the triangle
    def computeResidue(self):
        isBarrier = map(lambda x: x.isBarrier(), self.edges)
        if any(isBarrier):
          return 0
        else:
          residue = 0
          for edge in self.edges:
            residue = residue + edge.diff()
          return residue

    # Each edge keeps a list of triangles it is part of
    def __updateEdges(self):
        for edge in self.edges:
            edge.updateTri(self.index)
        for edge in self.__edges:
            edge.updateAdj(self.index)

    def unwrap(self):
        for edge, redge in zip(self.edges, self.__edges):
          edge.unwrap(redge)

    # Return None if not corner; else the rEdge
    def Corner(self):
        cornerEdge = []
        for edge, redge in zip(self.edges, self.__edges):
          if edge.isCorner():
              cornerEdge.append(redge)
        return cornerEdge

    # test function
    @staticmethod
    def __getCenter(vertices):
        center = (0,0)
        for v in vertices:
          center = center + (v.x, v.y)
        return (center[0]/len(vertices), center[1]/len(vertices))

    def isUnwrapped(self):
        unWrapped = map(lambda x: x.isUnwrapped(), self.edges)
        return all(unWrapped)

    def printEdges(self):
        for v in self.edges:
          print(v)

    def printFlow(self):
        flow = []
        for edge in self.edges:
          flow.append(edge.flow)
        print(flow)

    def plot(self, ax):
        if self.residue != 0:
            ax.plot(self.__center[0], self.__center[1], '*', c='r')

# Packs all the traingles together
class PhaseUnwrap(object):
    def __init__(self, x=None, y=None, phase=None, compNum=None, redArcs=0):
        # Expects a list of ve:tices
        self.loops      = None
        self.neutralResidue = None
        self.neutralNodeIdx = None
        self.__unwrapSeq    = None
        self.__unwrapped    = None
        self.__cornerEdges  = []

        # used only for plotting, and finally returning in sequence
        # Dont use these variables for computation
        self.__x             = None
        self.__y             = None
        self.__delauneyTri   = None
        self.__vertices      = None
        self.__edges         = None
        self.__compNum       = None
        self.__adjMat        = None
        self.__spanningTree  = None
        self.__CSRspanTree   = None
        self.__redArcs       = None

        # Edges used for unwrapping
        self.__unwrapEdges   = None

        # Using PuLP as an interface for defining the LP problem
        self.__prob__ = pulp.LpProblem("Unwrapping as LP optimization problem", pulp.LpMinimize)

        if compNum is None:
            compNum = [None]*len(x)
        else:
            self.__compNum  = compNum

        def_x = True
        if 'any' not in dir(x) and x == None:
            def_x = False

        def_y = True
        if 'any' not in dir(y) and y == None:
            def_y = False

        def_phase = True
        if 'any' not in dir(phase) and phase == None:
            def_phase = False

#        if None not in (x, y, phase):
        if def_x and def_y and def_phase:
            # Create
            vertices                = self.__createVertices(x, y, phase, compNum)
            self.nVertices          = len(vertices)
            delauneyTri             = self.__createDelaunay(vertices)
            self.__adjMat           = self.__getAdjMat(delauneyTri.vertices)
            self.__spanningTree     = self.__getSpanningTree()
            edges                   = self.__createEdges(vertices, redArcs)

            if (redArcs >= 0):
              self.loops  = self.__createLoop(vertices, edges)
            else:
              self.loops  = self.__createTriangulation(delauneyTri, vertices, edges)
              self.neutralResidue, self.neutralNodeIdx = self.__computeNeutralResidue()

            # Saving some variables for plotting
            self.__redArcs      = redArcs
            self.__x            = x
            self.__y            = y
            self.__vertices     = vertices
            self.__delauneyTri  = delauneyTri
            self.__edges        = edges
            self.__unwrapEdges  = []

    def __getSpanningTree(self):
      '''
      Computes spanning tree from adjcency matrix
      '''
      from scipy.sparse import csr_matrix
      from scipy.sparse.csgraph import minimum_spanning_tree

      # Spanning Tree
      spanningTree = minimum_spanning_tree(csr_matrix(self.__adjMat))
      spanningTree = spanningTree.toarray().astype(int)

      # Converting into a bi-directional graph
      spanningTree = np.logical_or(spanningTree, spanningTree.T).astype(int)
      return spanningTree

    @staticmethod
    def __getTriSeq(va, vb, vc):
        '''
        Get Sequence of triangle points
        '''
        line = lambda va, vb, vc: (((vc.y - va.y)*(vb.x - va.x))) - ((vc.x - va.x)*(vb.y - va.y))

        # Line equation through pt0 and pt1
        # Test for pt3 - Does it lie to the left or to the right ?
        pos3 = line(va, vb, vc)
        if(pos3 > 0):
            # left
            return (va, vc, vb)
        else: # right
            return (va, vb, vc)

    # Create Delaunay Triangulation.
    def __createDelaunay(self, vertices, ratio=1.0):
        pts = np.zeros((len(vertices), 2))

        for index, point in enumerate(vertices):
            pts[index,:] = [point.x, ratio*point.y]

        return Delaunay(pts)

    def __createVertices(self, x, y, phase, compNum):
        vertices = []
        for i, (cx, cy, cphase, cNum) in enumerate(zip(x, y, phase, compNum)):
            vertex = Vertex(cx, cy, cphase, cNum, i)
            vertices.append(vertex)
        return vertices

    # Edges indexed with the vertices
    def __createEdges(self, vertices, redArcs):
        edges = {}
        def add_edge(i,j,dist):
            if (i,j) not in edges:
                cand = Edge(i,j,dist,name="E(%d,%d)"%(i.getIndex(),j.getIndex()))
                edges[i, j] = cand
            if (j,i) not in edges:
                cand = Edge(j,i,dist,name="E(%d,%d)"%(j.getIndex(),i.getIndex()))
                edges[j, i] = cand
            return

        # Mask used to keep track of already created edges for efficiency
        mask = np.zeros((self.nVertices, self.nVertices))

        # Loop count depending on redArcs or MCF
        if redArcs <= 0:
          loopCount = 0
        else:
          loopCount = redArcs

        # Edges are added using the adj Matrix
        distMat = self.__adjMat.copy()
        dist = 1
        while (loopCount >= 0):
          # Remove self loops
          np.fill_diagonal(distMat, 0)

          # Add edges with incrementing length
          Va, Vb = np.where(np.logical_and((distMat > 0), np.invert(mask.astype(bool))))
          for (va, vb) in zip(Va, Vb):
              add_edge(vertices[va], vertices[vb], dist)

          # Update mask
          mask = np.logical_or(self.__adjMat.astype(bool), mask)

          # Generating adj matrix with redundant arcs
          distMat = np.dot(distMat, self.__adjMat.T)
          loopCount = loopCount-1

        return edges

    def __getAdjMat(self, vertices):
        '''
        Get adjacency matrix using the vertices
        '''
        adjMat = np.zeros((self.nVertices, self.nVertices))
        for ia, ib, ic in vertices:
          adjMat[ia, ib] = adjMat[ib, ic] = adjMat[ic, ia] = 1
          adjMat[ib, ia] = adjMat[ic, ib] = adjMat[ia, ic] = 1

        return adjMat

    # Inefficient - need to be replaced
    def __getSequence(self, idxA, idxB):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import depth_first_order

        def traverseToRoot(nodeSeq, pred):
          # scipy uses -9999
          __SCIPY_END = -9999
          seqV = [idxB]
          parent = pred[idxB]
          while parent != __SCIPY_END:
            seqV = [parent] + seqV
            parent = pred[parent]
          return seqV

        if self.__CSRspanTree is None:
            self.__CSRspanTree = csr_matrix(self.__spanningTree)

        (nodeSeq, pred) = depth_first_order(self.__CSRspanTree, i_start=idxA, \
                                            directed=False, \
                                            return_predecessors=True)

        # Traverse through predecessors to the root node
        seqV = traverseToRoot(nodeSeq, pred)
        if (seqV[0] != idxA):
          raise ValueError("Traversal Incorrect")
        else:
          return seqV

    def __createLoop(self, vertices, edges):
        # Creating the traingle object
        # triangles.append(Loop([va, vb, vc], edges, i+1))
        loops = []
        zeroVertex = Vertex(0,0)
        index = 0
        loopExist = {}
        for (va, vb), edge in edges.items():
            if self.__spanningTree[va.index, vb.index]:
                # Edge belongs to spanning tree
                continue
            else:
                # Traverse through the spanning tree
                seqV = self.__getSequence(va.index, vb.index)
                seqV = [vertices[i] for i in seqV]
                orientV = self.__getTriSeq(seqV[0], seqV[1], seqV[-1])

                # Reverse SeqV if needed to get proper loop direction
                if orientV[1] != seqV[1]:
                  seqV = seqV[::-1]

                # Get a uniform direction of loop
                if (seqV[0], seqV[-1]) in loopExist:
                    # Loop already added
                    continue
                else:
                    loopExist[(seqV[0], seqV[-1])] = 1

                # Create Loop
                loops.append(Loop(seqV, edges, index+1))
                index = index + 1

        return loops

    def __createTriangulation(self, delauneyTri, vertices, edges):
        '''
        Creates the Triangle residule for MCF formulation
        '''

        # Generate the points in a sequence
        def getSeq(va, vb, vc):
          line = lambda va, vb, vc: (((vc.y - va.y)*(vb.x - va.x))) - ((vc.x - va.x)*(vb.y - va.y))

          # Line equation through pt0 and pt1
          # Test for pt3 - Does it lie to the left or to the right ?
          pos3 = line(va, vb, vc)
          if(pos3 > 0):
            # left
            return (va, vc, vb)
          else: # right
            return (va, vb, vc)

        loop = []
        for i, triIdx in enumerate(delauneyTri.simplices):
            va, vb, vc = self.__indexList(vertices, triIdx)
            va, vb, vc = getSeq(va, vb, vc)
            loop.append(Loop([va, vb, vc], edges, i+1))
        return loop

    def __unwrapNode__(self, idx, nodes, predecessors):
        '''
        Unwraps a specific node while traversing to the root
        '''
        if self.__unwrapped[predecessors[idx]] == 0:
          # Go unwrap the predecessor before unwrapping current idx
          self.__unwrapNode__(predecessors[idx], nodes, predecessors)

        # Unwrap Node
        srcV = self.__vertices[predecessors[idx]]
        destV = self.__vertices[idx]
        edge = self.__edges[srcV, destV]
        rEdge = self.__edges[destV, srcV]
        self.__unwrapEdges.append([edge, rEdge])
        edge.unwrap(rEdge)

        # Update Flag
        self.__unwrapped[idx] = 1
        return

    def unwrapLP(self):
        from scipy.sparse.csgraph import breadth_first_order as DFS

        # Depth First search to get sequence of nodes
        (nodes, predecessor) = DFS(self.__spanningTree, 0)

        # Init Vertex
        self.__unwrapped = np.zeros((len(self.__vertices)))
        self.__vertices[nodes[0]].updatePhase(0)
        self.__unwrapped[0] = 1

        # Loop until there is a node unwrapped
        while (0 in self.__unwrapped):
          idx = next((i for i, x in enumerate(self.__unwrapped) if not(x)), None)
          self.__unwrapNode__(idx, nodes, predecessor)

        # Returns unwrapped vertices n values
        nValue = []
        for vertex in self.__vertices:
            nValue.append(vertex.getPhase())

        return nValue

    # Unwrapping the triangles
    # Traingle unwrap
    def __unwrap__(self):
        # Start unwrapping with the root
        initTriangle = self.loops[self.__unwrapSeq[0]]
        initTriangle.edges[0].src.updatePhase(0)

        for triIdx in self.__unwrapSeq:
            tri = self.loops[triIdx]
            tri.unwrap()

        # Returns unwrapped vertices n values
        nValue = []
        for vertex in self.__vertices:
            nValue.append(vertex.getPhase())

        return nValue

    # Unwrap triangle wise
    def __unwrapSequence(self, neighbors):
        # Generate sequence
        nodeSequence = [0]
        for i in range(len(self.loops)):

            # Finding adjacent nodes to current triangle
            cNode = nodeSequence[i]
            adjlist = np.array(neighbors[cNode])
            adjNode = adjlist[np.where(adjlist != -1)[0]]

            # adding list of new nodes by carefully removing already existing nodes
            newNodes = list(set(adjNode) - set(nodeSequence))
            nodeSequence = nodeSequence + newNodes

        return nodeSequence

    # Balances the residue in the entire network
    def __computeNeutralResidue(self):
        neutralResidue = 0
        for t in self.loops:
            neutralResidue = neutralResidue - t.residue
        return (neutralResidue, len(self.loops) + 1)

    def __nodeSupply(self):
        nodeSupply = []
        for i, loop in enumerate(self.loops):
            nodeSupply.append(loop.getNodeSupply())
        nodeSupply.append("n %d %d\n"%(self.neutralNodeIdx, self.neutralResidue))
        return nodeSupply

    def __flowConstraints(self):
        edgeFlow = []
        for loop in self.loops:
            edgeFlow.extend(loop.getFlowConstraint(self.neutralNodeIdx))

        # Edges between neutral and corner trianglea
        for loop in self.loops:
            cornerEdge = loop.Corner()
            if cornerEdge != []:
              for rEdge in cornerEdge:
                # Dirty - To be cleaned later
                rEdge.updateTri(self.neutralNodeIdx)

                # Edge flow constraints for corner edges
                edgeFlow.append(rEdge.getFlow(self.neutralNodeIdx))
                self.__cornerEdges.append(rEdge)

        return edgeFlow

    #  Phase unwrap using relax IV
    @staticmethod
    def __createRelaxInput(nodes, edges, fileName):
        # Input file for Relax IV
        f = open(fileName, "w")

        # Prepending nodes and edges
        f.write("p min %d %d\n"%(len(nodes), len(edges)))

        # Write node
        for line in nodes:
            f.write(line)

        # Write edge
        for line in edges:
            f.write(line)

        # Done writting network
        f.close()

    @staticmethod
    def __MCFRelaxIV(edgeLen, fileName="network.dmx"):
        ''' Uses MCF from RelaxIV '''
        try:
          from . import unwcomp
        except:
          raise Exception("MCF requires RelaxIV solver - Please drop the RelaxIV software \
                          into the src folder and re-make")
        return unwcomp.relaxIVwrapper_Py(fileName)

    def solve(self, solver, filename="network.dmx"):
        # Choses LP or Min cost using redArcs
        if (self.__redArcs == -1):
          self.__solveMinCost__(filename)
        else:
          self.__solveEdgeCost__(solver)

    def __solveEdgeCost__(self, solver, fileName="network.dmx"):
        # Add objective
        objective = []
        for v, edge in self.__edges.items():
          objective.append(edge.getCost() * edge.getLPVar())
        self.__prob__ += pulp.lpSum(objective)

        # Add Constraints
        for loop in self.loops:
          self.__prob__.addConstraint(loop.getLPFlowConstraint())

        # Solve the objective function
        if solver == 'glpk':
          print('Using GLPK MIP solver')
          MIPsolver = lambda: self.__prob__.solve(pulp.GLPK(msg=0))
        elif solver == 'pulp':
          print('Using PuLP MIP solver')
          MIPsolver = lambda: self.__prob__.solve()
        elif solver == 'gurobi':
          print('Using Gurobi MIP solver')
          MIPsolver = lambda: self.__prob__.solve(pulp.GUROBI_CMD())

        print('Time Taken (in sec) to solve: %f'%(T.timeit(MIPsolver, number=1)))

        # Get solution
        for v, edge in self.__edges.items():
          flow = pulp.value(edge.getLPVar())
          edge.updateFlow(flow)

    def __writeMPS__(self, filename):
        self.__prob__.writeMPS(filename)

    # Phase Unwrap using picos
    def __solveMinCost__(self, fileName="network.dmx"):

        # Creating RelaxIV input
        nodes = self.__nodeSupply()
        edgeFlow = self.__flowConstraints()

        # Running RelaxIV Interface
        self.__createRelaxInput(nodes, edgeFlow, fileName)
        edgeWeight = self.__MCFRelaxIV(len(edgeFlow), fileName)

        # Creating dictionary for interior edges
        triEdgeCost = zip(*(iter(edgeWeight[:3*len(self.loops)]),) *3)
        for triCost, tri in zip(triEdgeCost, self.loops):
            tri.updateEdgeFlow(triCost)

        # Dirty - to be changed
        cornerEdgeCost = edgeWeight[3*len(self.loops):]
        for rEdge, cost in zip(self.__cornerEdges, cornerEdgeCost):
            rEdge.updateFlow(cost)

    @staticmethod
    def __indexList(a,b):
        from operator import itemgetter

        return itemgetter(*b)(a)

    # Test function
    def isUnwrapped(self):
        for tri in self.loops:
            if tri.isUnwrapped():
                continue
            else:
                return False
        return True

    # Plot functions
    def plotResult(self, fileName):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.title('Phase as 3D plot')
        ax = fig.add_subplot(111, projection='3d')
        for v in self.__vertices:
            ax.scatter(v.x, v.y, v.phase, marker='o', label='Wrapped Phase')
            ax.scatter(v.x, v.y, v.getUnwrappedPhase(), marker='^', c='r', label='Unwrapped Phase')
        plt.savefig(fileName)

    def plotDelaunay(self, fileName):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.title('Delaunay Traingulation with Residue and non-zero edge cost')
        ax.triplot(self.__x, self.__y, self.__delauneyTri.simplices.copy(), c='b', linestyle='dotted')

        cmap = plt.get_cmap('hsv')
        numComponents = np.unique(self.__compNum).shape[0]
        colors = [cmap(i) for i in np.linspace(0, 1, numComponents)]

        ax.plot(self.__x, self.__y, '.')

        for v, edge in self.__edges.items():
            edge.plot(ax)
        for tri in self.loops:
            tri.plot(ax)

        plt.savefig(fileName)

    def plotSpanningTree(self, fileName):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.__x, self.__y, '.')

        for edge, rEdge in self.__unwrapEdges:
          if edge.flow == rEdge.flow:
            edge.plot(ax, c='g')
          else:
            edge.plot(ax, c='r')
        plt.savefig(fileName)

# Removes collinear points on the convex hull and removing duplicates
def filterPoints(x, y):
    def indexPoints(points):
        # Dirty: To be cleaned later
        # Numbering points
        pt2Idx = {}
        for idx, pt in enumerate(points):
            pt2Idx[pt] = idx
        return list(pt2Idx)

    # Removes duplicate
    points = list(zip(x, y))
    points = np.array(indexPoints(points))

    return (points[:,0], points[:,1])

def wrapValue(x):
  if((x <= np.pi) and (x > -np.pi)):
    return x
  elif(x > np.pi):
    return wrapValue(x - 2*np.pi)
  else:
    return wrapValue(x + 2*np.pi)

# Command Line argument parser
def firstPassCommandLine():

  # Creating the parser for the input arguments
  parser = argparse.ArgumentParser(description='Phase Unwrapping')

  # Positional argument - Input XML file
  parser.add_argument('--inputType', choices=['plane', 'sinc', 'sine'],
                      help='Type of input to unwrap', default='plane', dest='inputType')
  parser.add_argument('--dim', type=int, default=100,
                      help='Dimension of the image (square)', dest='dim')
  parser.add_argument('-c', action='store_true',
                      help='Component-wise unwrap test', dest='compTest')
  parser.add_argument('-MCF', action='store_true',
                      help='Minimum Cost Flow', dest='mcf')
  parser.add_argument('--redArcs', type=int, default=0,
                      help='Redundant Arcs', dest='redArcs')
  parser.add_argument('--solver', choices=['glpk', 'pulp', 'gurobi'],
                      help='Type of solver', default='pulp', dest='solver')

  # Parse input
  args = parser.parse_args()
  return args

def main():
  # Parsing the input arguments
  args = firstPassCommandLine()
  inputType   = args.inputType
  dim         = args.dim
  compTest    = args.compTest
  mcf         = args.mcf

  print("Input Type: %s"%(inputType))
  print("Dimension: %d"%(dim))

  # Random seeding to reapeat random numbers in case
  np.random.seed(100)

  inputImg = np.empty((dim, dim))
  if inputType == 'plane':
    # z = a*x + b*y
    a = np.random.randint(0,10,size=1)/50
    b = np.random.randint(0,10,size=1)/50
    for i in range(dim):
      for j in range(dim):
          inputImg[i,j] = (a*i) + (b*j)

  elif inputType == 'sinc':
    mag = np.random.randint(1,10,size=1)
    n = np.random.randint(1,3,size=1)
    for i in range(dim):
      for j in range(dim):
        inputImg[i,j] = mag * np.sinc(i*n*np.pi/100) * np.sinc(i*n*np.pi/100)

  elif inputType == 'sine':
    mag = np.random.randint(1,10,size=1)
    n = np.random.randint(1,3,size=1)
    for i in range(dim):
      for j in range(dim):
        inputImg[i,j] = mag * np.sin(i*n*np.pi/100) * np.sin(i*n*np.pi/100)

  if compTest:
    # Component wise unwrap testing
    n1 = np.random.randint(0,10,size=4)
    n2 = np.random.randint(0,10,size=4)
    compImg = np.empty((dim, dim))
    wrapImg = np.empty((dim, dim))
    for i in range(dim):
      for j in range(dim):
        compImg[i,j] = (i > (dim/2)) + 2*(j > (dim/2))
        n = n1[compImg[i,j]] + n2[compImg[i,j]]
        wrapImg[i,j] = inputImg[i,j] + n*(2*np.pi)
    wrapImg = np.array(wrapImg)
    compImg = np.array(compImg)
  else:
    # Wrapping input image
    wrapImg = np.array([[wrapValue(x) for x in row] for row in inputImg])
    compImg = None

  # Choosing random samples
  xidx = np.random.randint(0,dim,size=400).tolist()
  yidx = np.random.randint(0,dim,size=400).tolist()
  xidx, yidx = filterPoints(xidx, yidx)

  # Creating the Minimum Cost Flow problem
  if mcf is True:
    # We use redArcs to run Minimum Cost Flow
    redArcs = -1
    print('Relax IV used as the solver for Minimum Cost Flow')
    solver = None
  else:
    redArcs = args.redArcs
    solver = args.solver

  if compImg is None:
    phaseunwrap = PhaseUnwrap(xidx, yidx, wrapImg[xidx, yidx], redArcs=redArcs)
  else:
    phaseunwrap = PhaseUnwrap(xidx, yidx, wrapImg[xidx, yidx], compImg[xidx, yidx], redArcs=redArcs)

  # Including the neutral node for min cost flow
  phaseunwrap.solve(solver)

  # Unwrap
  phaseunwrap.unwrapLP()
  #phaseunwrap.plotDelaunay("final%d.png"%(redArcs))
  #phaseunwrap.plotSpanningTree("spanningTree%d.png"%(redArcs))
  phaseunwrap.plotResult("final%d.png"%(redArcs))

  #import pdb; pdb.set_trace()
  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection='3d')

if __name__ == '__main__':

  # Main Engine
  main()
