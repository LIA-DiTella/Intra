import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import p
import meshplot as mp


class Node:
    """
    Class Node
    """
    def __init__(self, value, radius, left = None, right = None, position = None, cl_prob= None, combos = None):
        self.left = left
        self.data = value
        self.radius = radius
        self.position = position
        self.right = right
        self.prob = cl_prob
        self.combos = combos
        self.children = [self.left, self.right]

    def all_subtrees(self, max_depth):
        yield Node(self.data, 1)

        if max_depth > 0:
            # For each child, get all of its possible sub-trees.
            self.child()
            child_subtrees = [list(self.children[i].all_subtrees(max_depth - 1)) for i in range(len(self.children)) if self.children[i] is not None]

            # Now for the n children iterate through the 2^n possibilities where
            # each child's subtree is independently present or not present. The
            # i-th child is present if the i-th bit in "bits" is a 1.
            for bits in range(1, 2 ** len(self.children)):
                for combos in all_combos([child_subtrees[i] for i in range(len(self.children)) if bits & (1 << i) != 0 and self.children[i] is not None]):
                    yield Node(self.data, 1, combos = combos)

    def child(self):
        self.children = [self.left, self.right]
    def toGraph( self, graph, index, dec, proc=True):
        
        
        radius = self.radius#.cpu().detach().numpy()
        if dec:
            radius= radius[0]
        #print("posicion", self.data, radius)
        #print("right", self.right)
        
        #graph.add_nodes_from( [ (index, {'posicion': radius[0:3], 'radio': radius[3] } ) ])
        graph.add_nodes_from( [ (self.data, {'posicion': radius[0:3], 'radio': radius[3] } ) ])
        

        if self.right is not None:
            #leftIndex = self.right.toGraph( graph, index + 1, dec)#
            self.right.toGraph( graph, index + 1, dec)#
            
            #graph.add_edge( index, index + 1 )
            graph.add_edge( self.data, self.right.data )
            #if proc:
            #    nx.set_edge_attributes( graph, {(index, index+1) : {'procesada':False}})
        
            if self.left is not None:
                #retIndex = self.left.toGraph( graph, leftIndex, dec )#
                self.left.toGraph( graph, 0, dec )#

                #graph.add_edge( index, leftIndex)
                graph.add_edge( self.data, self.left.data)
                #if proc:
                #    nx.set_edge_attributes( graph, {(index, leftIndex) : {'procesada':False}})
            
            else:
                #return leftIndex
                return

        else:
            #return index + 1
            return

    

def createNode(data, radius, position = None, left = None, right = None, cl_prob = None):
        """
        Utility function to create a node.
        """
        return Node(data, radius, position, left, right, cl_prob)

def height(root):
    # Check if the binary tree is empty
    if root is None:
        return 0 
    # Recursively call height of each node
    leftAns = height(root.left)
    rightAns = height(root.right)
    
    # Return max(leftHeight, rightHeight) at each iteration
    return max(leftAns, rightAns) + 1

def printCurrentLevel(root, level):
    if root is None:
        return
    if level == 1:
        print(root.data, end=" ")
    elif level > 1:
        printCurrentLevel(root.left, level-1)
        printCurrentLevel(root.right, level-1)

def printLevelOrder(root):
    h = height(root)
    for i in range(1, h+1):
       printCurrentLevel(root, i)

def serialize(root):
        
    def post_order(root):
        if root:
            post_order(root.left)
            post_order(root.right)
            ret[0] += str(root.data)+'_'+ str(root.radius) +';'
                
        else:
            ret[0] += '#;'           

    ret = ['']
    post_order(root)

    return ret[0][:-1]  # remove last 

def plotTree( root, dec ):
    graph = nx.Graph()
    root.toGraph( graph, 0, dec)
    edges=nx.get_edge_attributes(graph,'procesada')

    p = mp.plot( np.array([ graph.nodes[v]['posicion'] for v in graph.nodes]), shading={'point_size':0.1}, return_plot=True)

    for arista in graph.edges:
        p.add_lines( graph.nodes[arista[0]]['posicion'], graph.nodes[arista[1]]['posicion'])

    return 
#radius = [3, 3, 3]

radius = [2.5, 2.5, 2.5, 2.5]
radius2 = [2., 2., 2., 2.]

root = Node(1, [1., 1., 1., 1.])#0
root.left = createNode(3, [3., 3., 3., 3.]) #0.4
root.left.right = createNode(6, [6., 6., 6., 6.])#1 
#root.left.right.right = createNode(9, [6.5, 6., 6., 6.]) 
#root.left.right.left = createNode(10, [6.8, 6., 6., 6.]) 

root.left.left = createNode(7, radius) 
root.right = createNode(2, [10., 2., 2., 2.]) #0.2
root.right.right = createNode(4, [4., 4., 4., 4.]) #0.6
#root.right.right.right = createNode(5, [5., 4., 4., 4.]) #0.6
#root.right.right.right.right = createNode(6, [6., 4., 4., 4.]) #0.6
#root.right.right.right.right.right = createNode(7, [7., 4., 4., 4.]) #0.6
#root.right.right.right.right.right.right = createNode(8, [8., 4., 4., 4.]) #0.6

#root.right.right.right = createNode(8, radius2) 
#root.right.right.left = createNode(10, radius2) 

root.right.left = createNode(5, [5., 5., 5., 5.]) #0.8

#root.right.left.right = createNode(8, [10., 5., 5., 5.]) #0.8



#root.right.right = createNode(4, radius)


      
print("arbol")
printLevelOrder(root)


serial = serialize(root)
print("serialized", serial)
#write serialized string to file
file = open("./Trees/test2.dat", "w")
file.write(serial)
file.close() 

'''
root = Node(1, [ 23.0585, -69.1120, -49.2585,   1.9735])
#root.right = Node(2, [ 1.,1.,1.,1.])
#root.right.right = Node(4, [ 1.,1.,1.,1.])
#root.left = Node(3, [ 1.,1.,1.,1.])
serial = serialize(root)
file = open("./Trees/test1.dat", "w")
file.write(serial)
file.close() 
'''

def all_combos(choices):
    """
    Given a list of items (a,b,c,...), generates all possible combinations of
    items where one item is taken from a, one from b, one from c, and so on.

    For example, all_combos([[1, 2], ["a", "b", "c"]]) yields:

        [1, "a"]
        [1, "b"]
        [1, "c"]
        [2, "a"]
        [2, "b"]
        [2, "c"]
    """
    if not choices:
        yield []
        return

    for left_choice in choices[0]:
        for right_choices in all_combos(choices[1:]):
            yield [left_choice] + right_choices


    
'''
tree = Node(1,
[
    Node(2, [
        Node(4),
        Node(5)
    ]),
    Node(3,
    [
        Node(6),
        Node(7)
    ])
])
'''
tree = root

for subtree in tree.all_subtrees(3):
    print("///")
    print (subtree)
    breakpoint()
    plotTree(subtree, False)