
from typing import List, Union
import numpy as np
from node import Node
from scipy.linalg import null_space
from asyncio.windows_events import NULL
import random





class TukeyMedian:
            
    def __init__(self, d:int, points, n_levels:int) -> None:
        self.d = d
        self.points = points
        self.n_levels = n_levels
        self.n_leaves = d**(n_levels-1)
        self.root, self.leaves = self.initialize_Tree()
        self.sample_point_for_leaves()
        self.filling_tree(self.root)

    def initialize_Tree(self) -> Union[Node, List[Node]]:
        root = Node(ID=0, leaf=False)
        nodes = [root]
        leaf = False
        id = 1
        for i in range(1, self.n_levels):
            if i == self.n_levels-1:
                leaf = True
            next_level = []
            for n in nodes:         
                childeren = []

                for j in range(self.d):
                    c = Node(id, leaf, parent=n) 
                    childeren.append(c)
                    id += 1 
                    
                n.add_childeren(childeren)
                next_level.extend(childeren)
            nodes = next_level
        
        return root, nodes

    def sample_point_for_leaves(self):
        for leaf, p in zip(self.leaves, self.points):
            leaf.set_point(p)

    def find_radon_point(self, points):
        if len(points) == 0:
            return np.array()

        points = [x.reshape(1,-1) for x in points]

        M = points[0].T
        for p in points[1:]:
            M = np.concatenate((M, p.T), axis=1)
        
        ones = np.ones(shape=(1,len(points)))
        M = np.concatenate((M, ones), axis=0)

        N = null_space(M)

        convex_hull_mask = N < 0
        radon_point = M[:-1, :].dot((N * convex_hull_mask)).T
        return radon_point

    def set_radon_point_for_node(self, n:Node):
        points = [c.point for c in n.childeren]
        r = self.find_radon_point(points)
        n.set_point(r)

    def filling_tree(self, node):

        if node.childeren[0].has_point:
            self.set_radon_point_for_node(node)
            return

        for c in node.childeren:
            self.filling_tree(c)
        
        self.set_radon_point_for_node(node)
        

if __name__ == "__main__":
    fem_vecs = np.load("fem_vecs.npy")
    masc_vecs = np.load("masc_vecs.npy")

    ps = [x for x in fem_vecs]
    random.shuffle(ps)

    d , L = 302, 3
    while len(ps) < d**(L-1):
        ps.extend(ps)
    
    f_med = TukeyMedian(d=d, points =ps[:d**(L-1)] , n_levels=L)
    print(f_med.root.point)

    
