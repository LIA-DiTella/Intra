import networkx as nx
import pickle

class Node:
    """
    Class Node
    """
    def __init__(self, value, radius, position):
        self.left = None
        self.data = value
        self.radius = radius
        self.position = position
        self.right = None

    def is_leaf(self):
        if self.right is None:
            return True
        else:
            return False

node_list = []
MARKER = -1
class Tree:
    """
    Class tree will provide a tree as well as utility functions.
    """

    def createNode(self, data, radius, position):
        """
        Utility function to create a node.
        """
        return Node(data, radius, position)

    def insert(self, node , data, radius, position):
        """
        Insert function will insert a node into tree.
        Duplicate keys are not allowed.
        """
        #if tree is empty , return a root node
        if node is None:
            return self.createNode(data, radius, position)
        # if data is smaller than parent , insert it into left side
        if data < node.data:
            node.left = self.insert(node.left, data, radius, position)
        elif data > node.data:
            node.right = self.insert(node.right, data, radius, position)

        return node


    def search(self, node, data):
        """
        Search function will search a node into tree.
        """
        # if root is None or root is the search data.
        if node is None or node.data == data:
            return node

        if node.data < data:
            return self.search(node.right, data)
        else:
            return self.search(node.left, data)


    

    def traverseInorder(self, root):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorder(root.left)
            print (root.data, root.radius, root.position)
            self.traverseInorder(root.right)

    def serialize(self, root):
        if not root:
            return ''

        def post_order(root):
            if root:
                post_order(root.left)
                post_order(root.right)
                ret[0] += str(root.data) + ',' + str(root.radius) + ',' + str(root.position) + ';'
            #else:
            #    ret[0] += '#,'

        ret = ['']
        post_order(root)

        return ret[0][:-1]  # remove last ,

def deserialize(data):
        root = None
        tree2 = Tree()
        nodes = data.split(';')
        node = nodes.pop().split(',')
        data = node[0]
        radius = node[1]
        position = node[2]
        r = tree2.insert(root, data, radius, position) 


        def post_order(nodes):
            
            if  not nodes:
                return 
            node = nodes.pop().split(',')
            data = node[0]
            print("data", data)
            radius = node[1]
            position = node[2]
            tree2.insert(r, data, radius, position)
            root = Node(data, radius, position)
            root.right = post_order(nodes)
            root.left = post_order(nodes)
            return r, tree2
        return post_order(nodes)    


def main():
    

    filename = "ArteryObjAN1-6"

    grafo = pickle.load(open('grafos/' +filename + '-grafo.gpickle', 'rb'))

    grafo = grafo.to_undirected()


    l_nodes = [(n, nbrdict) for n, nbrdict in grafo.adjacency()]
    root = None
    tree = Tree()

    #agrego el primer nodo
    node = l_nodes[0]
    tag = node[0]
    print("tag", tag)
    radius = grafo.nodes[tag]['radio']
    posicion = grafo.nodes[tag]['posicion'].toNumpy()
    root = tree.insert(root, tag, radius, posicion)


    for node in l_nodes[1:]:
        tag = node[0]
        radius = grafo.nodes[tag]['radio']
        posicion = grafo.nodes[tag]['posicion'].toNumpy()
        tree.insert(root, tag, radius, posicion)


    print("arbol")
    tree.traverseInorder(root)
    #print(tree.search(root,61).right)
    #p = tree.search(root,61).is_leaf()
    #print("p", p)
    
    serial = tree.serialize(root)
    print("serialized", serial)


    #write serialized string to file
    file = open("ArteryObjAN1-7.dat", "w")
    file.write(serial)
    file.close() 

    deserial, tree3 = deserialize(serial)
    print(deserial)
    tree3.traverseInorder(deserial)
    
if __name__ == "__main__":
    main()