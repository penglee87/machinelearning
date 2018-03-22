import networkx as nx
import matplotlib.pyplot as plt

G=nx.DiGraph()

#G.add_node(1,pos=(1,1))
#G.add_node(2,pos=(2,2))
#G.add_node(3,pos=(1,0))
#G.add_edge(1,2,weight=0.5)
#G.add_edge(1,3,weight=9.8)
#pos=nx.get_node_attributes(G,'pos')
#
#nx.draw_networkx(G,pos)
#labels = nx.get_edge_attributes(G,'weight')
#nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
#plt.show()


T = [('a','b',3),\
('a','c',2),\
('a','d',2),\
('b','c',1),\
('b','d',2),\
('c','d',1),\
('c','e',2),\
('d','f',3),\
('e','a',4)]

G=nx.DiGraph()
G.add_weighted_edges_from(T)
pos=nx.spring_layout(G)

nx.draw_networkx(G,pos)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.show()