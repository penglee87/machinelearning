#!/usr/bin/env python3
''''' 
https://blog.csdn.net/qq_34731703/article/details/53965684
https://stackoverflow.com/questions/23988236/chu-liu-edmonds-algorithm-for-minimum-spanning-tree-on-directed-graphs
'''  
from collections import defaultdict, namedtuple


Arc = namedtuple('Arc', ('tail', 'weight', 'head'))

#最小生成树，参数为有向图和起始点(入度为零)
def min_spanning_arborescence(arcs, sink):
    print('arcs',len(arcs),arcs)
    good_arcs = []
    quotient_map = {arc.tail: arc.tail for arc in arcs}
    #quotient_map = {arc.head: arc.head for arc in arcs}
    print('quotient_map1',quotient_map)
    quotient_map[sink] = sink
    print('quotient_map2',quotient_map)
    #return 前一直循环
    while True:
        min_arc_by_tail_rep = {}
        successor_rep = {}
        for arc in arcs:
            #print('arc',arc)
            if arc.tail == sink:
                continue
            tail_rep = quotient_map[arc.tail]
            head_rep = quotient_map[arc.head]
            if tail_rep == head_rep:
                continue
            #保存每个点入边权重最小的边
            if tail_rep not in min_arc_by_tail_rep or min_arc_by_tail_rep[tail_rep].weight > arc.weight:
                min_arc_by_tail_rep[tail_rep] = arc
                successor_rep[tail_rep] = head_rep
        print('min_arc_by_tail_rep',min_arc_by_tail_rep)
        print('successor_rep',successor_rep)
        print('sink',sink)
        cycle_reps = find_cycle(successor_rep, sink)
        print('cycle_reps',cycle_reps)
        #如果不存在环
        if cycle_reps is None:
            good_arcs.extend(min_arc_by_tail_rep.values())
            return spanning_arborescence(good_arcs, sink)
        good_arcs.extend(min_arc_by_tail_rep[cycle_rep] for cycle_rep in cycle_reps)
        print('good_arcs',good_arcs)
        cycle_rep_set = set(cycle_reps)
        cycle_rep = cycle_rep_set.pop()
        #将环收缩成一个点
        quotient_map = {node: cycle_rep if node_rep in cycle_rep_set else node_rep for node, node_rep in quotient_map.items()}
        print('quotient_map3',quotient_map)

#查找环
def find_cycle(successor, sink):
    print('successor',successor)
    visited = {sink}
    for node in successor:
        cycle = []
        while node not in visited:
            visited.add(node)
            cycle.append(node)
            node = successor[node]
        if node in cycle:
            print('return1')
            return cycle[cycle.index(node):]
    print('return2')
    return None

#展开收缩点
def spanning_arborescence(arcs, sink):
    arcs_by_head = defaultdict(list)
    for arc in arcs:
        if arc.tail == sink:
            continue
        arcs_by_head[arc.head].append(arc)
    solution_arc_by_tail = {}
    stack = arcs_by_head[sink]
    while stack:
        arc = stack.pop()
        if arc.tail in solution_arc_by_tail:
            continue
        solution_arc_by_tail[arc.tail] = arc
        stack.extend(arcs_by_head[arc.tail])
    return solution_arc_by_tail

arcs = [Arc(1, 17, 0), Arc(2, 16, 0), Arc(3, 19, 0), Arc(4, 16, 0), Arc(5, 16, 0), Arc(6, 18, 0), Arc(2, 3, 1), Arc(3, 3, 1), Arc(4, 11, 1), Arc(5, 10, 1), Arc(6, 12, 1), Arc(1, 3, 2), Arc(3, 4, 2), Arc(4, 8, 2), Arc(5, 8, 2), Arc(6, 11, 2), Arc(1, 3, 3), Arc(2, 4, 3), Arc(4, 12, 3), Arc(5, 11, 3), Arc(6, 14, 3), Arc(1, 11, 4), Arc(2, 8, 4), Arc(3, 12, 4), Arc(5, 6, 4), Arc(6, 10, 4), Arc(1, 10, 5), Arc(2, 8, 5), Arc(3, 11, 5), Arc(4, 6, 5), Arc(6, 4, 5), Arc(1, 12, 6), Arc(2, 11, 6), Arc(3, 14, 6), Arc(4, 10, 6), Arc(5, 4, 6)]
arcs = [Arc(1, 17, 0), Arc(2, 16, 0), Arc(3, 19, 0), Arc(4, 16, 0), Arc(5, 16, 0), Arc(6, 18, 0), Arc(2, 3, 1), Arc(3, 3, 2), Arc(1, 11, 3)]
#print(min_spanning_arborescence(arcs, 0))
result = min_spanning_arborescence(arcs, 0)
print('result',type(result),result)

import networkx as nx
from networkx.algorithms.tree import branchings
import matplotlib.pyplot as plt

T1 = []
T2 = []
for arc in arcs:
    T1.append((arc.head,arc.tail,arc.weight))
    
for k,arc in result.items():
    T2.append((arc.head,arc.tail,arc.weight))

G1=nx.DiGraph()
G2=nx.DiGraph()
G1.add_weighted_edges_from(T1)
G2.add_weighted_edges_from(T2)

pos=nx.spring_layout(G1)

plt.subplot(211)
nx.draw_networkx(G1,pos)
labels = nx.get_edge_attributes(G1,'weight')
nx.draw_networkx_edge_labels(G1,pos,edge_labels=labels)

plt.subplot(212)
nx.draw_networkx(G2,pos)
labels = nx.get_edge_attributes(G2,'weight')
nx.draw_networkx_edge_labels(G2,pos,edge_labels=labels)

plt.show()

