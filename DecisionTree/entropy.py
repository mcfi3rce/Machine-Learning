import numpy as np
from node import Node
import operator

def entropy(p):
    if p!=0:
        return -p * np.log2(p)
    else:
        return 0

def most_common(train_data):
    count = {}
    unique, counts = np.unique(train_data, return_counts=True)
    count = dict(zip(unique, counts))
    most_common =max(count.iteritems(), key=operator.itemgetter(1))[0]
    return most_common

def print_tree(node):
    print node.name,
    if not node.isLeaf():
        for (key, value) in node.children.iteritems():
            print key, "{",
            print_tree(value),
            print "}",

def build_tree(train_data, attributes, removed = []):
    # make an empty node
    cur_node = Node()
    
    # remove any used attributes
    remaining = set(attributes) - set(removed)

    # test for same targets
    remaining_targets = train_data.iloc[:,-1].unique()

    # if no more rows SOMETHING IS WRONG
    if len(train_data) == 0:
        cur_node.name = "INVALID DATA"
        return cur_node

    if len(remaining_targets) == 1:
        # print "LEAF: ", remaining_targets[0]
        cur_node.name = remaining_targets[0]
        return cur_node
    # else if no more options
    elif len(remaining) == 0:
        #count the number of each class and return the most common
        leaf_val = Node(most_common(train_data.iloc[:,-1]))
        cur_node.appendChild(leaf_val, leaf_val)
        return cur_node
    else:
        # calculate the best value and set it to the node
        entropies = {}
        for attribute in remaining:
            entropies[attribute] = calculate_entropy(train_data, attribute)

        # find the lowest value in our list of entropies
        best_val = min(entropies, key=entropies.get)
        cur_node.name = best_val

        # get all possible values of root
        poss_values = train_data[best_val].unique()

        # build the tree
        child_nodes = {}
        for poss_value in poss_values:
            data_subset = train_data[train_data[best_val] == poss_value]
            data_subset.reset_index(inplace=True, drop=True)
            # remove this attribute
            removed.append(best_val)
            node = build_tree(data_subset, attributes, removed)
            # print "POS: ", poss_value
            child_nodes[poss_value] = node
            cur_node.children = child_nodes
            removed = []

    return cur_node

def calculate_entropy(train_data, attribute): 
    the_set = train_data[attribute].unique()
    no_bin = 0.0
    yes_bin = 0.0
    total_entropy = 0.0
    size = len(train_data)
    for attr in the_set:
        for x in range(0, size):
            if attr == train_data[attribute][x]:
                if train_data.iloc[:,-1][x] == 0:
                    no_bin += 1
                else:
                    yes_bin += 1

        total = no_bin + yes_bin
        no_bin_entropy = entropy(no_bin/total)
        yes_bin_entropy = entropy(yes_bin/total)
        single_entropy = no_bin_entropy + yes_bin_entropy
        weighted_entropy = single_entropy * (total / size)
        #print weighted_entropy
        total_entropy += weighted_entropy
        no_bin = yes_bin = 0.0
    return total_entropy
