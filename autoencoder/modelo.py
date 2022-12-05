from logging import raiseExceptions
from tokenize import Double
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from vec3 import Vec3
import meshplot as mp
import torch
torch.manual_seed(125)
import random
random.seed(125)
import torch_f as torch_f

class Node:
    """
    Class Node
    """
    def __init__(self, value, radius, left = None, right = None, position = None, cl_prob= None, ce = None, mse = None, level = None, treelevel = None):
        self.left = left
        self.data = value
        self.radius = radius
        self.position = position
        self.right = right
        self.prob = cl_prob
        self.mse = mse
        self.ce = ce
        self.children = [self.left, self.right]
        self.level = level
        self.treelevel = treelevel
    
    def agregarHijo(self, children):

        if self.right is None:
            self.right = children
        elif self.left is None:
            self.left = children

        else:
            raise ValueError ("solo arbol binario ")


    def is_leaf(self):
        if self.right is None:
            return True
        else:
            return False

    def is_two_child(self):
        if self.right is not None and self.left is not None:
            return True
        else:
            return False

    def is_one_child(self):
        if self.is_two_child():
            return False
        elif self.is_leaf():
            return False
        else:
            return True

    def childs(self):
        if self.is_leaf():
            return 0
        if self.is_one_child():
            return 1
        else:
            return 2
    
    
    def traverseInorder(self, root):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorder(root.left)
            print (root.data, root.radius)
            self.traverseInorder(root.right)

    def traverseInorderwl(self, root):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorderwl(root.left)
            print (root.data, root.radius, root.level, root.treelevel)
            self.traverseInorderwl(root.right)

    def get_tree_level(self, root, c):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.get_tree_level(root.left, c)
            c.append(root.level)
            self.get_tree_level(root.right, c)

    def set_tree_level(self, root, c):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.set_tree_level(root.left, c)
            root.treelevel = c
            self.set_tree_level(root.right, c)

    def traverseInorderLoss(self, root, loss):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorderLoss(root.left, loss)
            loss.append(root.prob)
            self.traverseInorderLoss(root.right, loss)
            return loss

    def traverseInorderMSE(self, root, loss):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorderMSE(root.left, loss)
            loss.append(root.mse)
            self.traverseInorderMSE(root.right, loss)
            return loss

    def traverseInorderCE(self, root, loss):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorderCE(root.left, loss)
            loss.append(root.ce)
            self.traverseInorderCE(root.right, loss)
            return loss

    def traverseInorderChilds(self, root, l):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorderChilds(root.left, l)
            l.append(root.childs())
            self.traverseInorderChilds(root.right, l)
            return l

    def preorder(self, root):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            print (root.data, root.radius)
            self.preorder(root.left)
            self.preorder(root.right)

    def cloneBinaryTree(self, root):
     
        # base case
        if root is None:
            return None
    
        # create a new node with the same data as the root node
        root_copy = Node(root.data, root.radius)
    
        # clone the left and right subtree
        root_copy.left = self.cloneBinaryTree(root.left)
        root_copy.right = self.cloneBinaryTree(root.right)
    
        # return cloned root node
        return root_copy

    def height(self, root):
    # Check if the binary tree is empty
        if root is None:
            return 0 
        # Recursively call height of each node
        leftAns = self.height(root.left)
        rightAns = self.height(root.right)
    
        # Return max(leftHeight, rightHeight) at each iteration
        return max(leftAns, rightAns) + 1

    # Print nodes at a current level
    def printCurrentLevel(self, root, level):
        if root is None:
            return
        if level == 1:
            print(root.data, end=" ")
        elif level > 1:
            self.printCurrentLevel(root.left, level-1)
            self.printCurrentLevel(root.right, level-1)

    def printLevelOrder(self, root):
        h = self.height(root)
        for i in range(1, h+1):
            self.printCurrentLevel(root, i)


    
    def count_nodes(self, root, counter):
        if   root is not None:
            self.count_nodes(root.left, counter)
            counter.append(root.data)
            self.count_nodes(root.right, counter)
            return counter

    
    def serialize(self, root):
        def post_order(root):
            if root:
                post_order(root.left)
                post_order(root.right)
                ret[0] += str(root.data)+'_'+ str(root.radius) +';'
                
            else:
                ret[0] += '#;'           

        ret = ['']
        post_order(root)
        return ret[0][:-1]  # remove last ,

    def toGraph( self, graph, index, dec, proc=True):
        
        
        radius = self.radius.cpu().detach().numpy()
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
            #return index + 1
            return

class InternalEncoder(nn.Module):
    
    def __init__(self, input_size: int, feature_size: int, hidden_size: int):
        super(InternalEncoder, self).__init__()

        #print("init")
        # Encoders atributos
        self.attribute_lin_encoder_1 = nn.Linear(input_size,feature_size)
        self.attribute_lin_encoder_2 = nn.Linear(feature_size,hidden_size)
        self.attribute_lin_encoder_3 = nn.Linear(hidden_size,feature_size)

        # Encoders derecho e izquierdo
        self.right_lin_encoder_1 = nn.Linear(feature_size,hidden_size)
        self.right_lin_encoder_2 = nn.Linear(hidden_size,feature_size)
        #self.right_lin_encoder_3 = nn.Linear(feature_size,feature_size)

        self.left_lin_encoder_1  = nn.Linear(feature_size,hidden_size)
        self.left_lin_encoder_2  = nn.Linear(hidden_size,feature_size)
        #self.left_lin_encoder_3  = nn.Linear(feature_size,feature_size)


        # Encoder final
        self.final_lin_encoder_1 = nn.Linear(2*feature_size, feature_size)

        # Funciones / Parametros utiles
        self.tanh = nn.Tanh()
        self.feature_size = feature_size


    def forward(self, input, right_input, left_input):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Encodeo los atributos
        attributes = self.attribute_lin_encoder_1(input)
        attributes = self.tanh(attributes)
        attributes = self.attribute_lin_encoder_2(attributes)
        attributes = self.tanh(attributes)
        attributes = self.attribute_lin_encoder_3(attributes)
        attributes = self.tanh(attributes)
        #print("attributes", attributes)

        # Encodeo el derecho
        if right_input is not None:
            #print("right input", right_input)
            context = self.right_lin_encoder_1(right_input)
            context = self.tanh(context)
            context = self.right_lin_encoder_2(context)
            #context = self.tanh(context)
            #context = self.right_lin_encoder_3(context)
            
            # Encodeo el izquierdo
            #print("left input", left_input)
            if left_input is not None:
                left = self.left_lin_encoder_1(left_input)
                #print("izquierdo", left.shape)
                left = self.tanh(left)
                #left = self.left_lin_encoder_2(left)
                #left = self.tanh(left)
                context += self.left_lin_encoder_2(left)
                #print("context izquierdo", context.shape)
        else:
            context = torch.zeros(input.shape[0],self.feature_size, requires_grad=True, device=device)
        

        context = self.tanh(context)
        feature = torch.cat((attributes,context), 1)
        feature = self.final_lin_encoder_1(feature)
        feature = self.tanh(feature)
        #print("output", feature)
        return feature

       
    

class GRASSEncoder(nn.Module):
    
    def __init__(self, input_size: int, feature_size : int, hidden_size: int):
        super(GRASSEncoder, self).__init__()
        self.leaf_encoder = InternalEncoder(input_size,feature_size, hidden_size)
        self.internal_encoder = InternalEncoder(input_size,feature_size, hidden_size)
        self.bifurcation_encoder = InternalEncoder(input_size,feature_size, hidden_size)
        
    def leafEncoder(self, node, right=None, left = None):
        return self.internal_encoder(node, right, left)
    def internalEncoder(self, node, right, left = None):
        return self.internal_encoder(node, right, left)
    def bifurcationEncoder(self, node, right, left):
        
        return self.bifurcation_encoder(node, right, left)

class NodeClassifier(nn.Module):
    
    def __init__(self, latent_size : int, hidden_size : int):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(latent_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, 3)
        self.tanh = nn.Tanh()

    def forward(self, input_feature):
        #print("classifier input", input_feature)
        output = self.mlp1(input_feature)
        output = self.tanh(output)
        output = self.mlp2(output)
        output = self.tanh(output)
        output = self.mlp3(output)
        #print("classifier output", output)

        return output


class Decoder(nn.Module):
    
    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self, latent_size : int, hidden_size : int):
        super(Decoder, self).__init__()
        #self.mlp = nn.Linear(latent_size,hidden_size)
        self.mlp = nn.Linear(latent_size,hidden_size)
        self.lp2 = nn.Linear(hidden_size, hidden_size)
        self.lp3 = nn.Linear(hidden_size, latent_size)

        self.mlp_left = nn.Linear(latent_size, hidden_size)
        self.mlp_left2 = nn.Linear(hidden_size, latent_size)
        #self.mlp_left3 = nn.Linear(latent_size, latent_size)
        self.mlp_right = nn.Linear(latent_size, hidden_size)
        self.mlp_right2 = nn.Linear(hidden_size, latent_size)
        #self.mlp_right3 = nn.Linear(latent_size, latent_size)


        self.mlp2 = nn.Linear(latent_size,latent_size)
        self.mlp3 = nn.Linear(latent_size,4)
        self.tanh = nn.Tanh()

    def common_branch(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        vector = self.lp2(vector)
        vector = self.tanh(vector)
        vector = self.lp3(vector)
        vector = self.tanh(vector)
        return vector

    def attr_branch(self, vector):
        vector = self.mlp2(vector)
        vector = self.tanh(vector)
        vector = self.mlp3(vector)        
        return vector

    def right_branch(self, vector):
        right_feature = self.mlp_right(vector)
        right_feature = self.tanh(right_feature)
        right_feature = self.mlp_right2(right_feature)
        right_feature = self.tanh(right_feature)
        #right_feature = self.mlp_right3(right_feature)
        #right_feature = self.tanh(right_feature)
        return right_feature

    def left_branch(self, vector):
        left_feature = self.mlp_left(vector)
        left_feature = self.tanh(left_feature)
        left_feature = self.mlp_left2(left_feature)
        left_feature = self.tanh(left_feature)
        #left_feature = self.mlp_left3(left_feature)
        #left_feature = self.tanh(left_feature)
        return left_feature

    def forward(self, parent_feature):
      
        vector      = self.common_branch(parent_feature)
        attr_vector = self.attr_branch(vector)
        return attr_vector 

    def forward1(self, parent_feature):
    

        vector       = self.common_branch(parent_feature)
        attr_vector  = self.attr_branch(vector)
        right_vector = self.right_branch(vector)
        
        #print("right vector", right_vector)
        #print("radius", attr_vector)
        return right_vector, attr_vector

    def forward2(self, parent_feature):
       

        vector       = self.common_branch(parent_feature)
        attr_vector  = self.attr_branch(vector)
        right_vector = self.right_branch(vector)
        left_vector  = self.left_branch(vector)
        #print("left vector", left_vector)
        #print("right vector", right_vector)
        #print("radius", attr_vector)
        return left_vector, right_vector, attr_vector



class GRASSDecoder(nn.Module):
    def __init__(self, latent_size : int, hidden_size: int, mult: torch.Tensor):
        super(GRASSDecoder, self).__init__()
        self.decoder = Decoder(latent_size, hidden_size)
        self.node_classifier = NodeClassifier(latent_size, hidden_size)
        self.mseLoss = nn.MSELoss()  # pytorch's mean squared error loss
        self.ceLoss = nn.CrossEntropyLoss(weight = mult)  # pytorch's cross entropy loss (NOTE: no softmax is needed before)
        


    def featureDecoder(self, feature):
        return self.decoder.forward(feature)

    def internalDecoder(self, feature):
        return self.decoder.forward1(feature)

    def bifurcationDecoder(self, feature):
        return self.decoder.forward2(feature)

    def nodeClassifier(self, feature):
        return self.node_classifier(feature)

    def calcularLossAtributo(self, nodo, radio):
        #print("nodo", nodo)
        #print("radio", radio)
        a, b = list(zip(*nodo))# a son los atributos, b los pesos
        if nodo is None:
            return
        else:
            nodo = torch.stack(list(a))
        
            l = [self.mseLoss(b.reshape(1,4), gt.reshape(1,4)) for b, gt in zip(radio.reshape(-1,4), nodo.reshape(-1,4))]
            #print("mse", l)
            return l


    def classifyLossEstimator(self, label_vector, original):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if original is None:
            return
        else:
           
            v = []
            for o in original:
                if o == 0:
                    vector = torch.tensor([1, 0, 0], device = device, dtype = torch.float)
                if o == 1:
                    vector = torch.tensor([0, 1, 0], device = device, dtype = torch.float)
                if o == 2:
                    vector = torch.tensor([0, 0, 1], device = device, dtype = torch.float)
                v.append(vector)
            

            v = torch.stack(v)
            
            l = [self.ceLoss(b.reshape(1,3), gt.reshape(1,3)).mul(0.4) for b, gt in zip(label_vector.reshape(-1,3), v.reshape(-1,3))]
         

            return l
            
    def vectorAdder(self, v1, v2):
        v = v1.add(v2)
        return v

    def vectorMult(self, m, v):
        #print("v", v)
        #print("m", m)
        z = zip(v, m)
        r = []
        for c, d in z:
            #print("v", c)
            #print("m", d)
            r.append(torch.mul(c, d))
        #res = [torch.mul(v, m) for v, m in zip(v, m)]
        #print("res", r)
        return r