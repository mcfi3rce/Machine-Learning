import numpy as np
from node import Node

def entropy(p):
    if p!=0:
        return -p * np.log2(p)
    else:
        return 0

def build_tree(train_data, test_data, attributes, removed = []):
    
    cur_node = Node()
    remaining = set(attributes) - set(removed)
    print remaining
    # if no more rows
    if len(train_data) == 0:
        return cur_node
    # else if no more options
    elif len(remaining) == 1:
        options = train_data[remaining.pop()].unique()
        if len(options) == 1:
            cur_node.children = options
            return cur_node
        else:
            cur_node.children = options[0]
            return cur_node
    else:
        # calculate the best value and set it to the node
        entropies = {}
        print "Remaining: ", remaining
        for attribute in remaining:
            entropies[attribute] = calculate_entropy(train_data[attribute], test_data["should_loan"])

        # find the lowest value in our list of entropies
        best_val = min(entropies, key=entropies.get)
        
        cur_node.name = best_val
        # get all possible values of root
        poss_values = train_data[best_val].unique()
        
        # build the tree
        for poss_value in poss_values:
            child_nodes = {}
            data_subset = train_data[train_data[best_val] == poss_value]
            # remove this attribute
            removed.append(best_val)
            node = build_tree(data_subset, test_data, attributes, removed)
            cur_node.children = child_nodes[poss_value] = node

    return cur_node

def calculate_entropy(classes, target): 
    the_set = classes.unique()
    print "Set: ", the_set, "Classes: ", classes
    no_bin = 0.0
    yes_bin = 0.0
    total_entropy = 0.0

    for x in the_set:
        for y in range(len(classes)):
            print "Set_Value: ", x , "Classes_val: ", classes[y]
            if x == classes[y]:
                if target[y] == 0:
                    no_bin += 1
                else:
                    yes_bin += 1

        total = no_bin + yes_bin
        no_bin_entropy = entropy(no_bin/total)
        yes_bin_entropy = entropy(yes_bin/total)
        single_entropy = no_bin_entropy + yes_bin_entropy
        weighted_entropy = single_entropy * (total / len(classes))
        total_entropy += weighted_entropy
        no_bin = yes_bin = 0.0
    return total_entropy
