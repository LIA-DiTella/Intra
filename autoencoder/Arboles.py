import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import p



class Node:
    """
    Class Node
    """
    def __init__(self, value, radius, left = None, right = None, position = None, cl_prob= None):
        self.left = left
        self.data = value
        self.radius = radius
        self.position = position
        self.right = right
        self.prob = cl_prob

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

#radius = [3, 3, 3]

radius = [2.5, 2.5, 2.5, 2.5]
radius2 = [2., 2., 2., 2.]

root = Node(0, [1., 1., 1., 1.])#0
root.left = createNode(2, [3., 3., 3., 3.]) #0.4
#root.left.right = createNode(6, [6., 6., 6., 6.])#1 
#root.left.right.right = createNode(9, [6.5, 6., 6., 6.]) 
#root.left.right.left = createNode(10, [6.8, 6., 6., 6.]) 

#root.left.left = createNode(7, radius) 
root.right = createNode(1, [10., 2., 2., 2.]) #0.2
root.right.right = createNode(3, [4., 4., 4., 4.]) #0.6
#root.right.right.right = createNode(5, [5., 4., 4., 4.]) #0.6
#root.right.right.right.right = createNode(6, [6., 4., 4., 4.]) #0.6
#root.right.right.right.right.right = createNode(7, [7., 4., 4., 4.]) #0.6
#root.right.right.right.right.right.right = createNode(8, [8., 4., 4., 4.]) #0.6

#root.right.right.right = createNode(8, radius2) 
#root.right.right.left = createNode(10, radius2) 

root.right.left = createNode(4, [5., 5., 5., 5.]) #0.8

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

