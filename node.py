from asyncio.windows_events import NULL
from distutils.log import error
from lib2to3.pytree import Node
from this import d
from typing import List
from xmlrpc.client import Boolean
import numpy as np
from pkg_resources import parse_requirements


class Node:
        def __init__(self, ID:int, leaf:Boolean, parent=NULL) -> None:
            self.leaf = leaf
            self.ID = ID
            if not leaf:
                self.childeren = []
            
            self.has_point = False
            self.point = np.empty((0))
            self.parent = parent
            
        def set_point(self, p_array):
            self.point = p_array
            self.has_point = True

        def add_childeren(self, childeren:List[Node]) -> None:
            if self.leaf:
                error("Trying to assign childeren to a leaf node")
            self.childeren.extend(childeren)