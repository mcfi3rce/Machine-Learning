class Node():
    def __init__(self, name = "", children = {}):
        self.name = name
        self.children = children

    def isLeaf(self):
        return self.children == type(dict)

    def appendChild(self, attribute, value):
        self.children[attribute] = value
