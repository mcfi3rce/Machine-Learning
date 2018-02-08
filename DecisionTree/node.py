class Node():
    def __init__(self, name = "", children = {}):
        self.name = name
        self.children = children

    def isLeaf(self):
        return self.name == self.children

    def appendChild(self, attribute, value):
        self.children[attribute] = value

    def __repr__(self):
        if self.isLeaf():
            print self.name
        else:
            print self.children
