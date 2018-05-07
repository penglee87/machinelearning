#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
http://www.cnblogs.com/linxiyue/p/3825241.html
'''
class Vertex(object):
    def __init__(self,key):
        self.id=key
        self.adj={}
    def addNeighbor(self,nbr,weight=0):
        self.adj[nbr]=weight
    def getNeighbors(self):
        return self.adj.keys()
    def getId(self):
        return self.id
    def getWeight(self,key):
        return self.adj[key]
class Graph(object):
    def __init__(self):
        self.vertexlist={}
        self.size=0
    def addVertex(self,key):
        vertex=Vertex(key)
        self.vertexlist[key]=vertex
        self.size+=1
        return vertex
    def getVertex(self,key):
        return self.vertexlist.get(key)
    def __contains__(self,key):
        if key in self.vertexlist:
            return True
        else:
            return False
    def addEdge(self,f,t,weight=0):
        if f not in self.vertexlist:
            self.addVertex(f)
        if t not in self.vertexlist:
            self.addVertex(t)
        self.vertexlist[f].addNeighbor(self.vertexlist[t],weight)
    def getVertices(self):
        return self.vertexlist.keys()
    def __iter__(self):
        return iter(self.vertexlist.values())
        
        
# 拓扑排序
# http://www.cnblogs.com/linxiyue/p/3826390.html
def topSort(G):
    top=[]
    queue=[]
    inDegree={}
    for v in G:
        inDegree[v]=0
    for v in G:
        for w in v.getNeighbors():
            inDegree[w]+=1
    for v in inDegree:
        if inDegree[v]==0:
            queue.append(v)
    while queue:
        v=queue.pop(0)
        top.append(v)
        for i in v.getNeighbors():
            inDegree[i]-=1
            if inDegree[i]==0:
                queue.append(i)
    return top
    
    
# 广度优先搜索与单源无权最短路径
# http://www.cnblogs.com/linxiyue/p/3832251.html
def unweighted(G,v):
    queue=[]
    path_length={}
    path_length[v]=0
    queue.append(v)
    while queue:
        v=queue.pop(0)
        for i in v.getNeighbors():
            if i not in path_length:
                path_length[i]=path_length[v]+1
                queue.append(i)
    return path_length
    
    
# 非负权有向图的单源最短路径问题，Dijkstra算法
# http://www.cnblogs.com/linxiyue/p/3833971.html
import sys
class Vertex(object):
    def __init__(self,key):
        self.id=key
        self.adj={}
    def addNeighbor(self,nbr,weight=0):
        self.adj[nbr]=weight
    def getNeighbors(self):
        return self.adj.keys()
    def getId(self):
        return self.id
    def getWeight(self,key):
        return self.adj[key]
class Graph(object):
    def __init__(self):
        self.vertexlist={}
        self.size=0
    def addVertex(self,key):
        vertex=Vertex(key)
        self.vertexlist[key]=vertex
        self.size+=1
        return vertex
    def getVertex(self,key):
        return self.vertexlist.get(key)
    def __contains__(self,key):
        if key in self.vertexlist:
            return True
        else:
            return False
    def addEdge(self,f,t,weight=0):
        if f not in self.vertexlist:
            self.addVertex(f)
        if t not in self.vertexlist:
            self.addVertex(t)
        self.vertexlist[f].addNeighbor(self.vertexlist[t],weight)
    def getVertices(self):
        return self.vertexlist.keys()
    def __iter__(self):
        return iter(self.vertexlist.values())
def Dijkstra(G,s):
    path={}
    vertexlist=[]
    for v in G:
        vertexlist.append(v)
        path[v]=sys.maxsize
    path[s]=0
    queue=PriorityQueue(path)
    queue.buildHeap(vertexlist)
    while queue.size>0:
        vertex=queue.delMin()
        for v in vertex.getNeighbors():
            newpath=path[vertex]+vertex.getWeight(v)
            if newpath<path[v]:
                path[v]=newpath
                queue.perUp(v)
    return path      
class PriorityQueue(object):
    def __init__(self,path):
        self.path=path
        self.queue=[]
        self.size=0
    def buildHeap(self,alist):
        self.queue=alist
        self.size=len(alist)
        for i in xrange(self.size/2-1,0,-1):
            self._perDown(i)
    def delMin(self):
        self.queue[0],self.queue[-1]=self.queue[-1],self.queue[0]
        minvertex=self.queue.pop()
        self.size-=1
        self._perDown(0)
        return minvertex
     
    def perUp(self,v):
        i=self.queue.index(v)
        self._perUp(i)
    def _perUp(self,i):
        if i>0:
            if self.path[self.queue[i]]<=self.path[self.queue[(i-1)/2]]:
                self.queue[i],self.queue[(i-1)/2]=self.queue[(i-1)/2],self.queue[i]
                self._perUp((i-1)/2)
    def _perDown(self,i):
        left=2*i+1
        right=2*i+2
        little=i
        if left<=self.size-1 and self.path[self.queue[left]]<=self.path[self.queue[i]]:
            little=left
        if right<=self.size-1 and self.path[self.queue[right]]<=self.path[self.queue[little]]:
            little=right
        if little!=i:
            self.queue[i],self.queue[little]=self.queue[little],self.queue[i]
            self._perDown(little)
        
if __name__=='__main__':
    g= Graph()
    g.addEdge('u','x',1)
    g.addEdge('u','v',2)
    g.addEdge('u','w',5)
    g.addEdge('x','v',2)
    g.addEdge('x','y',1)
    g.addEdge('x','w',3)
    g.addEdge('v','w',3)
    g.addEdge('y','w',1)
    g.addEdge('y','z',1)
    g.addEdge('w','z',5)
    u=g.getVertex('u')
    path=Dijkstra(g,u)
    for v in path:
        print v.id,path[v]

        
# 最小生成树
# http://www.cnblogs.com/linxiyue/p/3849239.html
# Prim算法
def Prim(G,s):
    path={}
    pre={}
    alist=[]
    for v in G:
        alist.append(v)
        path[v]=sys.maxsize
        pre[v]=s
    path[s]=0
    queue=PriorityQueue(path)
    queue.buildHeap(alist)
    while queue.size>0:
        vertex=queue.delMin()
        for v in vertex.getNeighbors():
            newpath=vertex.getWeight(v)
            if v in queue.queue and newpath<path[v]:
                path[v]=newpath
                pre[v]=vertex
                queue.perUp(v)
    return pre
if __name__=='__main__':
    g= Graph()
    g.addEdge('a','b',2)
    g.addEdge('b','a',2)
    g.addEdge('a','c',3)
    g.addEdge('c','a',3)
    g.addEdge('b','c',1)
    g.addEdge('c','b',1)
    g.addEdge('b','d',1)
    g.addEdge('d','b',1)
    g.addEdge('d','e',1)
    g.addEdge('e','d',1)
    g.addEdge('b','e',4)
    g.addEdge('e','b',4)
    g.addEdge('c','f',5)
    g.addEdge('f','c',5)
    g.addEdge('e','f',1)
    g.addEdge('f','e',1)
    g.addEdge('f','g',1)
    g.addEdge('g','f',1)
    u=g.getVertex('a')
    path=Prim(g,u)
    for v in path:
        print v.id,' after ',path[v].id
        
        
#Kruskal算法
class Vertex(object):
    def __init__(self,key):
        self.id=key
        self.adj={}
        self.parent=None
        self.rank=0
    def addNeighbor(self,nbr,weight=0):
        self.adj[nbr]=weight
    def getNeighbors(self):
        return self.adj.keys()
    def getId(self):
        return self.id
    def getWeight(self,key):
        return self.adj[key]
def Kruskal(G):
    elist=[]
    accpeted_e_list=[]
    for v in G:
        for vertex in v.getNeighbors():
            e=Edge(v,vertex,v.getWeight(vertex))
            elist.append(e)
    queue=KruskalQueue(elist)
    queue.buildHeap()
    edge_num=0
    while edge_num<G.size-1:
        e=queue.delMin()
        u=e.u
        v=e.v
        uset=Find(u)
        vset=Find(v)
        if uset!=vset:
            accpeted_e_list.append(e)
            edge_num+=1
            Union(uset,vset)
    return accpeted_e_list     
class Edge(object):
    def __init__(self,u,v,weight):
        self.u=u
        self.v=v
        self.weight=weight
class KruskalQueue(object):
    def __init__(self,elist):
        self.elist=elist
        self.size=len(self.elist)
    def buildHeap(self):
        for i in xrange(self.size/2-1,-1,-1):
            self.perDown(i)
    def delMin(self):
        self.elist[0],self.elist[-1]=self.elist[-1],self.elist[0]
        e=self.elist.pop()
        self.size-=1
        self.perDown(0)
        return e
    def perDown(self,i):
        left=2*i+1
        right=2*i+2
        little=i
        if left<=self.size-1 and self.elist[i].weight>self.elist[left].weight:
            little=left
        if right<=self.size-1 and self.elist[little].weight>self.elist[right].weight:
            little=right
        if little!=i:
            self.elist[i],self.elist[little]=self.elist[little],self.elist[i]
            self.perDown(little)
    def perUp(self,i):
        if i>0 and self.elist[i].weight<self.elist[(i-1)/2].weight:
            self.elist[i],self.elist[(i-1)/2]=self.elist[(i-1)/2],self.elist[i]
            self.perUp((i-1)/2)       
def Find(v):
    if v.parent is None:
        return v
    else:
        v.parent=Find(v.parent)
        return v.parent
def Union(u,v):
    if u.rank<=v.rank:
        u.parent=v
        if u.rank==v.rank:
            v.rank+=1
    else:
        v.parent=u   
if __name__=='__main__':
    g= Graph()
    g.addEdge('a','b',2)
    g.addEdge('a','c',3)
    g.addEdge('b','c',1)
    g.addEdge('b','d',1)
    g.addEdge('d','e',1)
    g.addEdge('b','e',4)
    g.addEdge('f','c',5)
    g.addEdge('f','e',1)
    g.addEdge('g','f',1)
    elist=Kruskal(g)
    for e in elist:
        print 'edge(%s,%s)'%(e.u.id,e.v.id)