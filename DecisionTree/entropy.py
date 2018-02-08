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

def build_tree(train_data, test_data, attributes, removed = []):
    # make an empty node
    cur_node = Node()

    # remove any used attributes
    remaining = set(attributes) - set(removed)

    # if no more rows SOMETHING IS WRONG
    if len(train_data) == 0:
        cur_node.name = "INVALID DATA"
        return cur_node

    # else if no more options
    elif len(remaining) == 1:
        options_set = train_data[remaining.pop()].unique()
        if len(options_set) == 1:
            cur_node.appendChild(options_set, options_set)
            return cur_node
        else:
            #count the number of each class and return the most common
            leaf_val = Node(most_common(train_data))
            cur_node.appendChild(leaf_val, leaf_val)
            return cur_node
    else:
        # calculate the best value and set it to the node
        entropies = {}
        print remaining
        for attribute in remaining:
            entropies[attribute] = calculate_entropy(train_data[attribute], test_data["should_loan"])

        for k, v in entropies.iteritems():
            print k, v
        
        # find the lowest value in our list of entropies
        best_val = min(entropies, key=entropies.get)
        cur_node.name = best_val
        # get all possible values of root
        poss_values = train_data[best_val].unique()
        
        # build the tree
        child_nodes = {}
        for poss_value in poss_values:
            data_subset = train_data[train_data[best_val] == poss_value]
            # remove this attribute
            removed.append(best_val)
            node = build_tree(data_subset, test_data, attributes, removed)
            cur_node.children = child_nodes[poss_value] = node
            removed = []

    return cur_node

def calculate_entropy(classes, target): 
    the_set = classes.unique()
    no_bin = 0.0
    yes_bin = 0.0
    total_entropy = 0.0

    
    for attribute in the_set:
        # DON'T FORGET TO RESET THE ITERATOR
        it = np.nditer(classes, flags=['f_index'])
        while not it.finished:
            class_type = it[0]
            if attribute == class_type:
                if target[it.index] == 0:
                    no_bin += 1
                else:
                    yes_bin += 1
            it.iternext()
        total = no_bin + yes_bin
        no_bin_entropy = entropy(no_bin/total)
        yes_bin_entropy = entropy(yes_bin/total)
        single_entropy = no_bin_entropy + yes_bin_entropy
        weighted_entropy = single_entropy * (total / len(classes))
        print weighted_entropy
        total_entropy += weighted_entropy
        no_bin = yes_bin = 0.0
    return total_entropy
