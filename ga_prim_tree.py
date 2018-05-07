from collections import defaultdict
from heapq import heapify, heappop, heappush
   
def prim( nodes, edges ):
    conn = defaultdict(list)
    
    for n1,n2,c in edges:
        conn[n1].append((c, n1, n2))
        conn[n2].append((c, n2, n1))
    
    print('coon',type(conn.items()),'list',list(conn.items()))
    mst = []  #保存最终的最小生成树
    used = set(nodes[0])  #保存已遍历的点
    usable_edges = conn[nodes[0]][:]
    print('usable_edges',usable_edges)
    heapify(usable_edges)  #堆规则排序
    print('usable_edges2',usable_edges)
  
    while usable_edges:
        cost, n1, n2 = heappop( usable_edges )  #将最小边推出
        if n2 not in used:
            used.add( n2 )
            mst.append( ( n1, n2, cost ) )
   
            for e in conn[ n2 ]:
                if e[ 2 ] not in used:
                    heappush( usable_edges, e )
    return mst
   
nodes = list("ABCDEFGHI")
#edges = [ ("A", "B", 7), ("A", "D", 5),  
#               ("B", "C", 8), ("B", "D", 9),  
#               ("B", "E", 7), ("C", "E", 5),  
#               ("D", "E", 15), ("D", "F", 6),  
#               ("E", "F", 8), ("E", "G", 9),  
#               ("F", "G", 11),("H", "I", 11)]
               
edges = [ ("A", "B", -7), ("A", "D", -5),  
               ("B", "C", -8), ("B", "D", -9),  
               ("B", "E", -7), ("C", "E", -5),  
               ("D", "E", -15), ("D", "F", -6),  
               ("E", "F", -8), ("E", "G", -9),  
               ("F", "G", -11),("H", "I", -11)]
   
print("prim:", prim( nodes, edges ))