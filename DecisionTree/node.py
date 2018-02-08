class Node():
    def __init__(self, name = "", children = {}):
        self.name = name
        self.children = children

    def isLeaf(self):
        return len(self.children) == 0

    def appendChild(self, value):
        self.children.append(value)

    def getNextChild(self, node_value):
        if self.isLeaf():
            return self
        else:
            for child in self.children:
                if (child == node_value):
                    return self.name

    def display(self):
        if self.isLeaf():
            print("{}".format(self.name))
        else:
            print self.name,
            for child in self.children:
                print " -> ", child.name,

