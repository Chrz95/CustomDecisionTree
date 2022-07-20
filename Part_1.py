######################## IMPORTS ################################

import numpy as np
import pydotplus
import pandas as pd
import csv
import random
import numpy.random as rd
import math
import copy
import matplotlib.pyplot as plt

######################## CLASSES ##############################

class Node:
    ID = 0 # Unique Integer ID
    Father = 0 # Pointer to another node
    LeftChild = 0
    RightChild = 0
    # The question that will split the node
    Feature = 0
    Value = 0
    isLeaf = 0
    Data = []
    Labels = []
    DataSize = 0
    Avg_Impurity = 0
    Information_Gain = 0
    Depth = 0
    Test_Data = []
    Majority_Class = 0
    Class_Errors = 0
    TestDataSize = 0
    Error_Rate = 0
    Relative_Error_Rate = 0
    Relative_Class_Errors = 0
    Subtree_Error = 0
    Error_Complexity = 0
    Subtree_Leaves = 0
    Error_Without_Pruning = 0
    Expected_Error_With_Pruning = 0

    def __init__(self,ID = 0,Father = 0,LeftChild = 0,RightChild = 0,  Value = 0, Feature = 0 , isLeaf = 0 , Data = []):
        self.ID = ID
        self.Father = Father
        self.LeftChild = LeftChild
        self.RightChild = RightChild
        self.Feature = Feature
        self.isLeaf = isLeaf
        self.Value = Value
        self.Data = Data
        self.Test_Data = []

        self.Labels = [row [13] for row in Data]
        count_dict = dict((x,self.Labels.count(x)) for x in self.Labels)
        self.Majority_Class = max(zip(count_dict.keys() ,count_dict.values()))

        if (Father != 0) :
            self.Depth = self.Father.Depth + 1
        else:
            self.Depth = 0

        self.DataSize = len(self.Data)
        self.DataImpurity = Impurity(self.Data,impurity_type)

    def BecomeLeaf(self):
        self.isLeaf = 1
        self.LeftChild = 0
        self.RightChild = 0
        self.Feature = 0
        self.Value = 0
        self.Subtree_Error = 0
        self.Subtree_Leaves = 0
        self.Error_Complexity = math.inf

    # Creates a dictionary that contains information about all the nodes in the tree
    def create_info(self):

            tree_info[self.ID] = []

            if (self.Father != 0):
                tree_info[self.ID].append(self.Father.ID)
            else :
                tree_info[self.ID].append("None")

            if (self.Feature != 0):
                tree_info[self.ID].append(self.Feature)
            else :
                tree_info[self.ID].append("None")

            tree_info[self.ID].append(self.Value)

            if (self.isLeaf == 0):
                tree_info[self.ID].append("NO")
            else :
                tree_info[self.ID].append("YES")

            tree_info[self.ID].append(self.DataSize)
            tree_info[self.ID].append(self.DataImpurity)

            tree_info[self.ID].append(self.Depth)
            tree_info[self.ID].append(self.Avg_Impurity)
            tree_info[self.ID].append(self.Information_Gain)

            tree_info[self.ID].append(self.Majority_Class)
            tree_info[self.ID].append(self.TestDataSize)
            tree_info[self.ID].append(self.Class_Errors)

            tree_info[self.ID].append(self.Error_Rate)
            tree_info[self.ID].append(self.Subtree_Error)
            tree_info[self.ID].append(self.Error_Complexity)
            tree_info[self.ID].append(self.Subtree_Leaves)

            tree_info[self.ID].append(self.Data)
            tree_info[self.ID].append(self.Test_Data)

            if (self.LeftChild != 0):
                self.LeftChild.create_info()

            if (self.RightChild != 0):
                self.RightChild.create_info()

######################## FUNCTIONS ##############################

# Calculates impurity
def Impurity(node,type):

    purityvar = 0
    impurityvar = 1

    if(type == 1):   #Gini Purity
        for x in labels:
            if (len(node) == 0) :
                purityvar = 1
                break
            else :
                purityvar = (sum(row [13] == x for row in node)/(len(node)))**2 + purityvar

        impurityvar = 1 - purityvar
    elif(type == 2): #Entropy Purity
        for x in labels:
            if len(node) == 0 :
                purityvar = 1
                break
            else:
                if (sum(row [13] == x for row in node)/(len(node)) != 0) :
                    purityvar = ((sum(row [13] == x for row in node)/(len(node))) * math.log(sum(row [13] == x for row in node)/(len(node)))) + purityvar
                else :
                    purityvar = 0  + purityvar

        impurityvar = -purityvar

    return impurityvar

# Find the question (k,tk) that minimizes the cost function
def find_least_cost(node_train_data):

    minJ = 2

    for item in unique_values_dict.items() : # Choose the best pair (k = item,tk = x)
        for x in item[1] :

            leftpart , rightpart =  question(item[0],x,node_train_data)
            Nleft = len (leftpart)
            Nright = len (rightpart)

            Ileft = Impurity(leftpart, impurity_type)
            Iright = Impurity(rightpart, impurity_type)
            J = ((Nleft/N) * Ileft) + ((Nright/N) * Iright)

            if (J < minJ) :
                minJ = J
                minFeat = item[0]
                minPart = x
                final_leftpart = leftpart
                final_rightpart = rightpart

    return final_leftpart , final_rightpart , minFeat , minPart

# Creates and grows the tree
def partition_database(node_train_data, node):

    threshold = imp_threshold # Limit of tolerable impurity
    global id
    global start_node

    final_leftpart ,final_rightpart , minFeat , minPart = find_least_cost (node_train_data)

    if (node == 0): # Create the start node and its children
        cur_node = Node(id,0,0,0,minPart,minFeat,0,node_train_data)
        id = id + 1
        start_node = cur_node
        left_node = Node(id,cur_node,0,0,0,0,0,final_leftpart)
        id = id + 1
        right_node = Node(id,cur_node,0,0,0,0,0,final_rightpart)
        id = id + 1
        cur_node.LeftChild = left_node
        cur_node.RightChild = right_node
        cur_node.Avg_Impurity = (cur_node.LeftChild.DataSize / cur_node.DataSize) * cur_node.LeftChild.DataImpurity + (cur_node.RightChild.DataSize / cur_node.DataSize) *  cur_node.RightChild.DataImpurity
        cur_node.Information_Gain = cur_node.DataImpurity - cur_node.Avg_Impurity
    else : # Create the other nodes
        cur_node = node
        cur_node.Feature = minFeat
        cur_node.Value = minPart

        if ((not final_leftpart) or (not final_rightpart)): # If any of the children of the node has no children, make the current node a leaf
            cur_node.BecomeLeaf()
            return
        else:
            left_node = Node(id,cur_node,0,0,0,0,0,final_leftpart)
            id = id + 1
            right_node = Node(id,cur_node,0,0,0,0,0,final_rightpart)
            id = id + 1
            cur_node.LeftChild = left_node
            cur_node.RightChild = right_node
            cur_node.Avg_Impurity = (cur_node.LeftChild.DataSize / cur_node.DataSize) * cur_node.LeftChild.DataImpurity + (cur_node.RightChild.DataSize / cur_node.DataSize) *  cur_node.RightChild.DataImpurity
            cur_node.Information_Gain = cur_node.DataImpurity - cur_node.Avg_Impurity

    Ileft = Impurity(final_leftpart,impurity_type)
    Iright = Impurity(final_rightpart,impurity_type)

    # In case there is not prepruning
    if(prepruning(cur_node) == 0):
        if ((not final_leftpart) or (not final_rightpart)): # One of the nodes is empty
            cur_node.BecomeLeaf()
        else:
            if (condition(Ileft, Iright, threshold, condition_type, final_leftpart, final_rightpart,cur_node) == 1):  # both nodes can be split further
                partition_database(final_leftpart,left_node)
                partition_database(final_rightpart,right_node)
            elif (condition(Ileft, Iright, threshold, condition_type,final_leftpart, final_rightpart,cur_node) == 2): # only left node can be split further
                partition_database(final_leftpart,left_node)
                right_node.BecomeLeaf()
            elif (condition(Ileft, Iright, threshold, condition_type,final_leftpart, final_rightpart,cur_node) == 3): # only right node can be split further
                partition_database(final_rightpart,right_node)
                left_node.BecomeLeaf()
            else : # No node can be split further
                left_node.BecomeLeaf()
                right_node.BecomeLeaf()
    else :
        return

# Splits the node dataset into two new datasets according to the best question
def question(feature, value, train_data):

    leftpart = []
    rightpart = []

    Feature_Importance_before_pruning[feature] = Feature_Importance_before_pruning[feature] + 1

    for row in train_data :
        if (row[features_name[feature]] <= value):
            leftpart.append(row)
        else:
            rightpart.append(row)

    return leftpart , rightpart

# Implements stopping criteria
def condition(Ileft, Iright, threshold, type, leftpart, rightpart,cur_node):

    children_tree_depth = cur_node.Depth + 1

    if(type == 1): # Impurity-only condition
        if ((Ileft <= threshold) and (Iright <= threshold)):
            condition = 4
        elif((Ileft <= threshold) and (Iright > threshold)):
            condition = 3
        elif((Ileft > threshold) and (Iright <= threshold)):
            condition = 2
        else:
            condition = 1
    elif(type == 2): # Depth only condition
        if  (children_tree_depth >= tree_depth_limit):
            condition = 4
        else:
            condition = 1
    elif(type == 3): # Node max data only condition
        if (len(leftpart) <= node_limit) and (len(rightpart) <= node_limit):
            condition = 4
        elif (len(leftpart) <= node_limit) and (len(rightpart) > node_limit):
            condition = 3
        elif (len(leftpart) > node_limit) and (len(rightpart) <= node_limit):
            condition = 2
        else:
            condition = 1
    elif(type ==4): # depth condition AND node max data condition
        if ((len(leftpart) <= node_limit) and (len(rightpart) <= node_limit)) or (children_tree_depth >= tree_depth_limit):
            condition = 4
        elif (len(leftpart) <= node_limit)  and (len(rightpart) > node_limit):
            condition = 3
        elif (len(leftpart) > node_limit) and (len(rightpart) <= node_limit):
            condition = 2
        else:
            condition = 1
    elif(type ==5): # Impurity AND depth condition
        if ((Ileft <= threshold) and (Iright <= threshold))  or (children_tree_depth >= tree_depth_limit):
            condition = 4
        elif (Ileft <= threshold) and (Iright > threshold):
            condition = 3
        elif (Ileft > threshold) and (Iright <= threshold):
            condition = 2
        else:
            condition = 1
    elif(type ==6): # Impurity AND node max data condition
        if ((Ileft <= threshold) or (len(leftpart) <= node_limit)) and ((Iright <= threshold) or (len(rightpart) <= node_limit)):
            condition = 4
        elif ((Ileft <= threshold) or (len(leftpart) <= node_limit)) and ((Iright > threshold) and (len(rightpart) > node_limit)):
            condition = 3
        elif ((Ileft > threshold) or (len(leftpart) > node_limit)) and ((Iright <= threshold) and (len(rightpart) <= node_limit)):
            condition = 2
        else:
            condition = 1
    elif (type == 7): # Impurity AND depth condition AND node max data condition
        if (((Ileft <= threshold) or (len(leftpart) <= node_limit)) and ((Iright <= threshold) or (len(rightpart) <= node_limit))) or (children_tree_depth >= tree_depth_limit):
            condition = 4
        elif ((Ileft <= threshold) or (len(leftpart) <= node_limit)) and ((Iright > threshold) and (len(rightpart) > node_limit)):
            condition = 3
        elif ((Ileft > threshold) or (len(leftpart) > node_limit)) and ((Iright <= threshold) and (len(rightpart) <= node_limit)):
            condition = 2
        else:
            condition = 1

    return condition


def print_tree(LeavesOnly,PrintTrainSet,PrintTestSet):
    #  Create tree info and print it
    names = ['FatherID','Feature','Feature_Value','isLeaf','Data_Size','Impurity','Depth','Avg_Impurity','Info_Gain','Majority_Class' ,'TestDataSize','Class_Errors','Error_Rate','Subtree_Error','Error_Complexity','Subtree_Leaves','Train_Dataset','Test_Dataset']
    for i in sorted(tree_info) :
        if (LeavesOnly): # Print only the leaves
            if (tree_info[i][3] == 'YES') or (i == 0):
                print ('( ID =',i,')',end = ',')
                for j in range(0,len(tree_info[i]) - 2):
                        print ('(',names[j],'=',tree_info[i][j],')',end = ',')
                if (PrintTrainSet) :
                            print ('(',names[len(tree_info[i]) - 2],'=',tree_info[i][len(tree_info[i]) - 2],')',end = ',')
                if (PrintTestSet) :
                            print ('(',names[len(tree_info[i]) - 1],'=',tree_info[i][len(tree_info[i]) - 1],')',end = ',')
                print ("")
        else:
                print ('( ID =',i,')',end = ',')
                for j in range(0,len(tree_info[i]) - 2):
                    print ('(',names[j],'=',tree_info[i][j],')',end = ',')
                if (PrintTrainSet) :
                            print ('(',names[len(tree_info[i]) - 2],'=',tree_info[i][len(tree_info[i]) - 2],')',end = ',')
                if (PrintTestSet) :
                            print ('(',names[len(tree_info[i]) - 1],'=',tree_info[i][len(tree_info[i]) - 1],')',end = ',')
                print ("")

def prepruning(cur_node):

    if(prepruning_type != 0): # Info Gain Pruning

        if (prepruning_type == 1) :
            information_gain = cur_node.Information_Gain
            if (information_gain <= gain_threshold): # Pruning
                cur_node.BecomeLeaf()
                #print ("Pruned Node :",cur_node.ID)
                #print ("The cur_node is ",cur_node.ID,"with Information_gain : ",information_gain , "and inpurity of ", cur_node.DataImpurity,"\n")
                return 1
            else:
                return 0
        elif (prepruning_type == 2) : # Chi Square Pruning
            e_value = {}
            obs_value1 = {}
            obs_value2 = {}
            chi_squared = 0
            for x in range(0,NumOfClasses):
                e_value[x] = sum(row [13] == x for row in cur_node.Data)/(len(cur_node.Data))
                if (e_value[x] != 0):
                    obs_value1[x] = sum(row [13] == x for row in cur_node.LeftChild.Data)/(len(cur_node.LeftChild.Data))
                    obs_value2[x] = sum(row [13] == x for row in cur_node.RightChild.Data)/(len(cur_node.RightChild.Data))
                    chi_squared = ((obs_value1[x]-e_value[x])**2)/e_value[x] + ((obs_value2[x]-e_value[x])**2)/e_value[x] + chi_squared
            if (chi_squared <= chi_threshold): # Pruning
                #print ("Pruned Node :",cur_node.ID)
                cur_node.BecomeLeaf()
            else:
                return 0
    else:
        return 0

# Runs a test set on the fully grown tree and calculates the error and majority class for each node
def run_test_set(cur_node):

    global tree_errors
    # Calculate current node error
    Elements_in_maj_class = [row for row in cur_node.Test_Data if (row [13] == cur_node.Majority_Class[0])]
    cur_node.Class_Errors = len(cur_node.Test_Data) - len(Elements_in_maj_class)
    cur_node.TestDataSize = len(cur_node.Test_Data)
    cur_node.Error_Rate = cur_node.Class_Errors / len(start_node.Test_Data) # R(i)

    if (cur_node.isLeaf == 0):

        if (cur_node.LeftChild !=0) and (cur_node.RightChild !=0):
            cur_node.LeftChild.Test_Data = []
            cur_node.RightChild.Test_Data = []

            for row in cur_node.Test_Data :
                # print ("Split feature : ",cur_node.Feature,cur_node.Value )
                if (cur_node.Feature != 0):
                    if (row[features_name[cur_node.Feature]] <= cur_node.Value):
                        cur_node.LeftChild.Test_Data.append(row)
                    else:
                        cur_node.RightChild.Test_Data.append(row)

            run_test_set(cur_node.LeftChild)
            run_test_set(cur_node.RightChild)

    else : # Node is leaf
        tree_errors = tree_errors + cur_node.Class_Errors
        return 0 # End the recursion

def reduced_error_pruning(cur_node):

    if(cur_node.isLeaf == 1):
        return cur_node.Class_Errors

    left_error = reduced_error_pruning(cur_node.LeftChild)
    right_error = reduced_error_pruning(cur_node.RightChild)
    total_error = left_error + right_error

    if(cur_node.Class_Errors < total_error):
        cur_node.BecomeLeaf()
        #print("Pruned Node : ", cur_node.ID  )

    return cur_node.Class_Errors

# Used in Cost Complexity Pruning to calculate the needed variables
def calculate_subtree_errors(cur_node) :

    if (cur_node.isLeaf == 0):
        Left_Subtree_Error = calculate_subtree_errors(cur_node.LeftChild)
        Right_Subtree_Error = calculate_subtree_errors(cur_node.RightChild)

    else:
        if (cur_node.Father != 0):
            cur_node.Father.Subtree_Leaves = cur_node.Father.Subtree_Leaves + 1
        return cur_node.Error_Rate

    if (cur_node.Father != 0):
        cur_node.Father.Subtree_Leaves = cur_node.Father.Subtree_Leaves + cur_node.Subtree_Leaves

    cur_node.Subtree_Error = Left_Subtree_Error + Right_Subtree_Error
    if (cur_node.Subtree_Leaves - 1 != 0):
        cur_node.Error_Complexity = (cur_node.Error_Rate - cur_node.Subtree_Error)/(cur_node.Subtree_Leaves - 1)
    else:
        cur_node.Error_Complexity = math.inf

    return cur_node.Subtree_Error

def initialize_subtree_data(cur_node):

    if (cur_node.isLeaf == 0):
        if (cur_node.ID == 0): # If it is the root
            cur_node.Subtree_Leaves = -2
        else:
            cur_node.Subtree_Leaves = 0

        #cur_node.Subtree_Error = 0
        initialize_subtree_data(cur_node.LeftChild)
        initialize_subtree_data(cur_node.RightChild)
    else :
        return

def cost_complexity_pruning(start_node):

    global tree_errors
    global pruned_node
    global pruned_node_leftchild
    global pruned_node_rightchild

    left_errors = 0
    right_errors = 0

    found_best_acc = 0
    pruned_node = 0
    new_accuracy = 0

    tree_errors = 0
    calculate_tree_error (start_node) # Create the errors for each node
    best_accuracy = (1 - (tree_errors/len(start_node.Test_Data)))*100
    #print ("Best accuracy is :",best_accuracy,"%")

    # Prunes tree until the root has the minimum error complexity
    while (pruned_node != start_node) :

        initialize_subtree_data(start_node)
        calculate_subtree_errors(start_node)
        pruned_node = return_min_err_complexity(start_node)

        #print("Pruned Node: ",pruned_node.ID)
        pruned_node_leftchild = pruned_node.LeftChild
        pruned_node_rightchild = pruned_node.RightChild
        pruned_node.BecomeLeaf()

        tree_info.clear()
        start_node.create_info()

        '''
        print ("Tree with errors :\n")
        print_tree(False,False,False)
        '''

        tree_errors = 0
        calculate_tree_error (start_node)
        new_accuracy = (1 - (tree_errors/len(start_node.Test_Data)))*100
        #print ("New accuracy is :",new_accuracy,"%\n")

        if (new_accuracy >= best_accuracy):
            best_accuracy = new_accuracy
            found_best_acc = 1

        if (new_accuracy < best_accuracy):
            if (found_best_acc == 1) : # Resplit the last pruned node
                break

    return 0

def return_min_err_complexity (cur_node):

    if (cur_node.isLeaf == 0):
        left = return_min_err_complexity (cur_node.LeftChild)
        right = return_min_err_complexity (cur_node.RightChild)
    else :
        return cur_node

    if(min(left.Error_Complexity, right.Error_Complexity, cur_node.Error_Complexity) == cur_node.Error_Complexity):
        return cur_node
    elif(min(left.Error_Complexity, right.Error_Complexity, cur_node.Error_Complexity) == left.Error_Complexity):
        return left
    else:
        return right

def minimum_error_pruning (cur_node):

    if (cur_node.isLeaf == 0):
        minimum_error_pruning (cur_node.LeftChild)
        minimum_error_pruning (cur_node.RightChild)

        cur_node.Expected_Error_With_Pruning = (len(cur_node.Test_Data) - (len(cur_node.Test_Data) - cur_node.Class_Errors + NumOfClasses - 1))/(len(cur_node.Test_Data) + NumOfClasses)
        Error_Without_Pruning_Left =  (len(cur_node.LeftChild.Test_Data) - (len(cur_node.LeftChild.Test_Data) - cur_node.LeftChild.Class_Errors + NumOfClasses - 1))/(len(cur_node.LeftChild.Test_Data) + NumOfClasses)
        Error_Without_Pruning_Right =  (len(cur_node.RightChild.Test_Data) - (len(cur_node.RightChild.Test_Data) - cur_node.RightChild.Class_Errors + NumOfClasses - 1))/(len(cur_node.RightChild.Test_Data) + NumOfClasses)

        if len(cur_node.Test_Data) == 0:
            cur_node.Error_Without_Pruning  = 0
        else:
            cur_node.Error_Without_Pruning = (len(cur_node.LeftChild.Test_Data)/len(cur_node.Test_Data))*Error_Without_Pruning_Left + (len(cur_node.RightChild.Test_Data)/len(cur_node.Test_Data))*Error_Without_Pruning_Right

        if (cur_node.Expected_Error_With_Pruning <= cur_node.Error_Without_Pruning):
            cur_node.BecomeLeaf()
            #print (cur_node.ID,cur_node.isLeaf)

    else: return 0

def postpruning(start_node,type):
    if (type == 0): # No post-pruning
        return 0
    elif(type == 1): # Reduced error pruning
        return reduced_error_pruning(start_node)
    elif(type == 2): # Cost complexity pruning
        return cost_complexity_pruning (start_node)
    elif(type == 3): # Cost complexity pruning
        return minimum_error_pruning (start_node)

# Creates a duplicate of the tree
def clone_tree(cur_node):

    left = 0
    right = 0

    cloned_node = copy.deepcopy(cur_node)
    if (cur_node.isLeaf == 0):
        left = clone_tree(cur_node.LeftChild)
        right = clone_tree(cur_node.RightChild)

    cloned_node.LeftChild = left
    cloned_node.RightChild = right

    return cloned_node

def calculate_tree_error (cur_node) :
     global tree_errors
    # Calculate current node error
     if (cur_node != 0):
         Elements_in_maj_class = [row for row in cur_node.Test_Data if (row [13] == cur_node.Majority_Class[0])]
         cur_node.Class_Errors = len(cur_node.Test_Data) - len(Elements_in_maj_class)
         cur_node.TestDataSize = len(cur_node.Test_Data)
         cur_node.Error_Rate = cur_node.Class_Errors / len(start_node.Test_Data) # R(i)

         if (cur_node.isLeaf == 0):
             calculate_tree_error(cur_node.LeftChild)
             calculate_tree_error(cur_node.RightChild)
         else:
             tree_errors = tree_errors + cur_node.Class_Errors
             return 0 # End the recursion

# Calcualates how many elements have been severely misclassified (severity is decided by relative_distance)
def calculate_relative_error (cur_node) :
     global relative_errors
     global relative_distance
    # Calculate current node relative error
     if (cur_node!=0):
         Elements_in_maj_class = [row for row in cur_node.Test_Data if ((row [13] == cur_node.Majority_Class[0]) or (row [13] == cur_node.Majority_Class[0] -relative_distance) or (row [13] == cur_node.Majority_Class[0] + relative_distance)) ]
         cur_node.Relative_Class_Errors = len(cur_node.Test_Data) - len(Elements_in_maj_class)
         cur_node.TestDataSize = len(cur_node.Test_Data)
         cur_node.Relative_Error_Rate = cur_node.Relative_Class_Errors / len(start_node.Test_Data) # R(i)

         if (cur_node.isLeaf == 0):
             calculate_relative_error(cur_node.LeftChild)
             calculate_relative_error(cur_node.RightChild)
         else:
             relative_errors = relative_errors + cur_node.Relative_Class_Errors
             return 0 # End the recursion

# Finds the most important feature of the pruned tree
def find_best_feature (cur_node) :

    global Feature_Importance_after_pruning

    if (cur_node.Feature != 0):
        Feature_Importance_after_pruning[cur_node.Feature] = Feature_Importance_after_pruning[cur_node.Feature] + 1

        if (cur_node.LeftChild.isLeaf == 0):
            find_best_feature (cur_node.LeftChild)

        if (cur_node.RightChild.isLeaf == 0):
            find_best_feature (cur_node.RightChild)

    return 0

def run_classifier () :

        global unique_values_dict
        global id
        global best_accuracy
        global best_relative_accuracy
        global best_tree_num
        global start_node
        global best_tree_start_node
        global tree_errors
        global relative_errors
        global average_tree_accuracy
        global i
        global shuffled_data
        global data
        global train_data
        global test_data
        global pruning_data
        global unique_feature_values
        global names
        global features
        global folds
        global fold_data
        global accuracy_rep
        global relative_accuracy_rep
        global postpruning_type
        global prepruning_type
        global impurity_type
        global Feature_Importance_before_pruning
        global Feature_Importance_after_pruning
        global tree_info

        Feature_Importance_before_pruning =  {i:0 for i in names}
        Feature_Importance_after_pruning =  {i:0 for i in names}

        accuracy_rep.clear()
        relative_accuracy_rep.clear()

        accuracy_rep = []
        relative_accuracy_rep = []

        # Runs the algorithm many times to find more accurate result
        for k in range(0,repetitions):

            #print("{:.2%}".format((k/repetitions)))

            best_accuracy = 0
            best_tree_num = 0
            start_node = 0
            best_tree_start_node = 0
            i = 0
            id = 0
            tree_info = {}
            tree_errors = 0
            relative_errors = 0
            average_tree_accuracy = 0

            rd.seed(k)
            shuffled_data = data[:]
            rd.shuffle(shuffled_data)

            #   Splits the dataset into 5 folds for 5-fold Cross Validation
            fold_data = [[] for i in range (0,folds)]
            fold_data[0] = shuffled_data[0:61]
            fold_data[1] = shuffled_data[62:122]
            fold_data[2] = shuffled_data[123:182]
            fold_data[3] = shuffled_data[183:242]
            fold_data[4] = shuffled_data[243:303]

            # Fold

            for x in range (0, folds):

                # Create tree
                id = 0
                start_node = 0

                if (postpruning_type != 0) :
                    train_data = fold_data[x].tolist() + fold_data[(x+1)%folds].tolist()
                    pruning_data = fold_data[(x+2)%folds].tolist() + fold_data[(x+3)%folds].tolist()
                    test_data = fold_data[(x+4)%folds]

                    partition_database(train_data,0) # Build the tree

                    # Run pruning set on tree to calculate errors

                    start_node.Test_Data = pruning_data
                    tree_errors = 0
                    run_test_set(start_node)

                    accuracy = (1 - (tree_errors/len(start_node.Test_Data)))*100

                    # Post Pruning
                    postpruning(start_node,postpruning_type)

                    # Unprune the last pruned node (Only for Cost Complexity)
                    if (postpruning_type == 2):
                        pruned_node.LeftChild = pruned_node_leftchild
                        pruned_node.RightChild = pruned_node_rightchild
                        pruned_node.isLeaf = 0

                    tree_errors = 0
                    calculate_tree_error(start_node)
                    accuracy = (1 - (tree_errors/len(start_node.Test_Data[:])))*100

                    #Final Accuracy (On test samples)

                    start_node.Test_Data = test_data[0:14]
                    tree_errors = 0
                    run_test_set(start_node)
                    accuracy = (1 - (tree_errors/len(start_node.Test_Data[:])))*100

                    relative_errors = 0
                    calculate_relative_error (start_node)
                    relative_accuracy = (1 - (relative_errors/len(start_node.Test_Data)))*100

                    if (accuracy > best_accuracy):
                        best_tree_num = x
                        best_accuracy = accuracy
                        best_tree_start_node = copy.deepcopy(start_node)
                        best_relative_accuracy = relative_accuracy

                    average_tree_accuracy = average_tree_accuracy + accuracy

                    find_best_feature(start_node)

                else :
                    train_data = fold_data[x].tolist() + fold_data[(x+1)%folds].tolist() + fold_data[(x+2)%folds].tolist() + fold_data[(x+3)%folds].tolist()
                    test_data = fold_data[(x+4)%folds]

                    partition_database(train_data,0) # Build the tree

                   #Final Accuracy (On test samples)

                    # Cheating
                    start_node.Test_Data = test_data
                    tree_errors = 0
                    run_test_set(start_node)
                    accuracy = (1 - (tree_errors/len(start_node.Test_Data)))*100
                    relative_errors = 0
                    calculate_relative_error (start_node)
                    relative_accuracy = (1 - (relative_errors/len(start_node.Test_Data)))*100

                    if (accuracy > best_accuracy):
                        best_tree_num = x
                        best_accuracy = accuracy
                        best_tree_start_node = copy.deepcopy(start_node)
                        best_relative_accuracy = relative_accuracy

                    average_tree_accuracy = average_tree_accuracy + accuracy

                    if (prepruning_type != 0):
                        find_best_feature(start_node)

            # Final Result

            average_tree_accuracy = average_tree_accuracy/folds
            accuracy_rep.append(best_accuracy)
            relative_accuracy_rep.append(best_relative_accuracy)

        # Final Results

        Final_Accuracy = sum(accuracy_rep)/repetitions
        Final_Relative_Accuracy = sum(relative_accuracy_rep)/repetitions

        if ((Final_Accuracy >= accuracy_limit) and (Final_Accuracy <= 80)) or (automatic == 0):

            f.write("================================================================================ \n\n")

            if (impurity_type == 1):
                f.write("Impurity Type = Gini\n")
            else:
                f.write("Impurity Type = Entropy\n")

            if (prepruning_type == 0):
                f.write("Prepruning Type = None\n")
            elif (prepruning_type == 1):
                f.write("Prepruning Type = Information Gain Pruning\n")
            elif (prepruning_type == 2):
                f.write("Prepruning Type = Chi-Squared Error Pruning\n")

            if (postpruning_type == 0):
                f.write("Postpruning Type = None\n")
            elif (postpruning_type == 1):
                f.write("Postpruning Type = Reduced Error Pruning\n")
            elif (postpruning_type == 2):
                f.write("Postpruning Type = Cost Complexity Pruning\n")
            else:
                f.write("Postpruning Type = Minimum Error Pruning\n")

            f.write("Impurity Threshold = " + str(imp_threshold) + '\n')
            f.write("Max Tree Depth = " + str(tree_depth_limit) + '\n')
            f.write("Max Leaf Data = " + str(node_limit) + '\n' )
            f.write("Gain Threshold = " + str(gain_threshold) + "\n")
            f.write("Chi Threshold = " + str(chi_threshold) + "\n")
            f.write("\nFinal Accuracy : {:.2%}\n".format(Final_Accuracy/100))
            f.write("Final Relative Accuracy : {:.2%}\n".format(Final_Relative_Accuracy/100))

            # Feature Histogram

            Feature_Percentages = list(Feature_Importance_before_pruning.values())
            total = sum(Feature_Percentages)
            Feature_Percentages = [(Feature_Percentages[i]/total) for i in range (0,len(Feature_Percentages))]

            f.write("\nFeature Importance (before pruning) :\n")
            for i in range (0,features):
                f.write("{:}".format(names[i]))
                f.write(" = {:.2%}\n".format(Feature_Percentages[i]))
            #plt.pie(Feature_Percentages, labels=names,autopct='%1.1f%%', startangle=140)
            #plt.title('Features without pruning')
            #plt.show()

            if (postpruning_type != 0) or (prepruning_type != 0):
                Feature_Percentages = list(Feature_Importance_after_pruning.values())
                total = sum(Feature_Percentages)
                if (total != 0):
                    Feature_Percentages = [(Feature_Percentages[i]/total) for i in range (0,len(Feature_Percentages))]
                    f.write("\nFeature Importance after pruning :  \n")
                    for i in range (0,features):
                        f.write("{:}".format(names[i]))
                        f.write(" = {:.2%} \n".format(Feature_Percentages[i]))
                else:
                    f.write("\nFeature Importance after pruning : \n")
                    f.write("Only root remained after pruning \n")

                if (automatic == 0):
                    print("Close the pie chart to continue")
                    plt.pie(Feature_Percentages, labels=names,autopct='%1.1f%%', startangle=140)
                    plt.title('Features with pruning')
                    plt.show()

            f.write("\n================================================================================\n")

####################### MAIN ###############################

filename = 'cleveland.data'
names = ['Age', 'Sex', 'Pain', 'Pressure', 'Cholesterol','Sugar', 'Results', 'Max Heart Rate', 'Angina','Old Peak','Slope','Num of Major Vessels','Thal']
features_name = {names[i]:i for i in range(0,len(names))}
labels = [0,1,2,3,4]
Feature_Importance_before_pruning =  {i:0 for i in names}
Feature_Importance_after_pruning =  {i:0 for i in names}
raw_data = open(filename,'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x).astype('float')
folds = 5
features = len(names)
NumOfSamples = len(data)
NumOfClasses = len(labels)
N = NumOfSamples
unique_feature_values = [[] for i in range(0,features)]
tree_depth_limit = None
node_limit = None
imp_threshold = None
gain_threshold = None
chi_threshold = None
best_tree_start_node = 0
id = 0
tree_info = {}
i = 0
tree_errors = 0
relative_errors = 0
average_tree_accuracy = 0
relative_distance = 1
repetitions = 500
automatic = 0
accuracy_rep = []
relative_accuracy_rep = []
f= open("Results_1.txt","w+")

for x in range (0, features):
    unique_feature_values[x] = list(np.unique([item[x] for item in data[0:13]]))

unique_values_dict = {names[i] : unique_feature_values[i] for i in range (0,features)}

automatic = int(input('Enter 1 to run all cases or 0 to run a particular case : '))
while (automatic < 0) or (automatic > 1):
    automatic = int(input('Enter 1 to run all cases or 0 to run a particular case : '))

repetitions = int(input('Enter the number that the experiment will be repeated (Small value is faster but less accurate) : '))
while (repetitions <= 0) :
    repetitions = int(input('Enter the number that the experiment will be repeated (Small value is faster but less accurate) : '))

if (automatic == 0): # Run only for one case

    print ("Enter the necessary parameters : \n")

    condition_type = int(input('Enter the Condition type (1: Leaf Max Impurity Limit, 2: Max Tree Depth Limit, 3: Max Leaf Data Limit, 4: 2 & 3 , 5: 1 & 3 ,6: 1 & 2 ,7: 1 & 2 & 3 ): '))
    while (condition_type < 1) or (condition_type > 7):
        condition_type = int(input('Enter the Condition type (1: Leaf Max Impurity Limit, 2: Max Tree Depth Limit, 3: Max Leaf Data Limit, 4: 2 & 3 , 5: 1 & 3 ,6: 1 & 2 ,7: 1 & 2 & 3 ): '))

    if (condition_type == 1) or (condition_type == 5) or (condition_type == 6) or (condition_type == 7) :
        imp_threshold = float(input('Enter the impurity limit of the leaf nodes (>=0): '))
        while (imp_threshold < 0):
            imp_threshold = float(input('Enter the impurity limit of the leaf nodes (>=0): '))

    if (condition_type == 2) or (condition_type == 4) or (condition_type == 5) or (condition_type == 7) :
        tree_depth_limit = int(input('Enter the maximum tree depth (>1): '))
        while (tree_depth_limit <= 1):
            tree_depth_limit = int(input('Enter the maximum tree depth (>1): '))

    if (condition_type == 3) or (condition_type == 4) or (condition_type == 6) or (condition_type == 7) :
        node_limit = int(input('Enter the maximum samples of a leaf (>=1): '))
        while (node_limit < 1):
            node_limit = int(input('Enter the maximum samples of a leaf (>=1):'))

    impurity_type = int(input('Enter the Impurity type (1: Gini, 2: Entropy) : '))
    while (impurity_type < 1) or (impurity_type > 2):
        impurity_type = int(input('Enter the Impurity type (1: Gini, 2: Entropy) : '))

    prepruning_type = int(input('Enter the Prepruning type (0 : None, 1 : Information Gain, 2 : Chi-Squared): '))
    while (prepruning_type < 0) or (prepruning_type > 2):
        prepruning_type = int(input('Enter the Prepruning type (0 : None, 1 : Information Gain, 2 : Chi-Squared): '))

    postpruning_type = int(input('Enter the Postpruning type (0: None, 1: Reduced Error, 2: Cost Complexity, 3: Minimum Error): '))
    while (postpruning_type < 0) or (postpruning_type > 3):
        ostpruning_type = int(input('Enter the Postpruning type (0: None, 1: Reduced Error, 2: Cost Complexity, 3: Minimum Error): '))

    print('')

    if (prepruning_type == 1):
        gain_threshold = float(input('Enter the gain threshold (>= 0) : '))
        while (gain_threshold < 0) or (postpruning_type > 3):
            gain_threshold = float(input('Enter the gain threshold (>= 0) : '))
        print('')
    elif (prepruning_type == 2):
        chi_threshold = float(input('Enter the chi threshold (>= 0) : '))
        while (chi_threshold < 0) :
            chi_threshold = float(input('Enter the chi threshold (>= 0) : '))
        print('')

    accuracy_limit = 0

    print ("Please wait. This may take a few minutes.....")

    run_classifier ()

else : # Run for all cases
    accuracy_limit = float(input('Choose the lower bound of the accuracy of the classifiers you wish to see (0 - 100) : '))
    while (accuracy_limit < 0) or (accuracy_limit > 100):
        accuracy_limit = float(input('Choose the lower bound of the accuracy of the classifiers you wish to see (0 - 100) : '))

    print ("Please wait. This may take a few minutes.....")
    for cond in range (1,8):
        for postpr in range (0,4):
            for prepr in range (0,3):
                if (prepr == 0):
                    chi_threshold = None
                    gain_threshold = None
                    for impurity in range (1,3):
                        if (cond == 1) :
                            for imp_thr in np.arange (0.01,0.11,0.03):
                                imp_threshold = imp_thr
                                tree_depth_limit = None
                                node_limit = None
                                impurity_type = impurity
                                postpruning_type = postpr
                                prepruning_type = prepr
                                condition_type = cond
                                run_classifier ()
                        if (cond == 2) :
                            for depth in range (6,10):
                                tree_depth_limit = depth
                                imp_threshold = None
                                node_limit = None
                                impurity_type = impurity
                                postpruning_type = postpr
                                prepruning_type = prepr
                                condition_type = cond
                                run_classifier ()
                        if (cond == 3) :
                            for n_lim in range (2,5):
                                imp_threshold = None
                                tree_depth_limit = None
                                node_limit = n_lim
                                impurity_type = impurity
                                postpruning_type = postpr
                                prepruning_type = prepr
                                condition_type = cond
                                run_classifier ()
                        if (cond == 4) :
                            for depth in range (6,10):
                                for n_lim in range (2,5):
                                    node_limit = n_lim
                                    tree_depth_limit = depth
                                    impurity_type = impurity
                                    postpruning_type = postpr
                                    prepruning_type = prepr
                                    condition_type = cond
                                    run_classifier ()
                        if (cond == 5) :
                            for imp_thr in np.arange (0.01,0.11,0.03):
                                for depth in range (6,10):
                                    imp_threshold = imp_thr
                                    tree_depth_limit = depth
                                    impurity_type = impurity
                                    postpruning_type = postpr
                                    prepruning_type = prepr
                                    condition_type = cond
                                    run_classifier ()
                        if (cond == 6) :
                            for imp_thr in np.arange (0.01,0.11,0.03):
                                for n_lim in range (2,5):
                                    node_limit = n_lim
                                    imp_threshold = imp_thr
                                    impurity_type = impurity
                                    postpruning_type = postpr
                                    prepruning_type = prepr
                                    condition_type = cond
                                    run_classifier ()
                        if (cond == 7) :
                            for imp_thr in np.arange (0.01,0.11,0.03):
                                for n_lim in range (2,5):
                                    for depth in range (6,10):
                                        imp_threshold = imp_thr
                                        node_limit = n_lim
                                        tree_depth_limit = depth
                                        impurity_type = impurity
                                        postpruning_type = postpr
                                        prepruning_type = prepr
                                        condition_type = cond
                                        run_classifier ()
                else:
                    for thres in np.arange (0,0.2,0.03):
                        for impurity in range (1,3):
                            if (cond == 1) :
                                for imp_thr in np.arange (0.01,0.11,0.03):
                                    imp_threshold = imp_thr
                                    tree_depth_limit = None
                                    node_limit = None
                                    impurity_type = impurity
                                    if (prepr == 1):
                                        gain_threshold = thres
                                        chi_threshold = None
                                    else:
                                        chi_threshold = thres
                                        gain_threshold = None
                                    postpruning_type = postpr
                                    prepruning_type = prepr
                                    condition_type = cond
                                    run_classifier ()
                            if (cond == 2) :
                                for depth in range (6,10):
                                    tree_depth_limit = depth
                                    imp_threshold = None
                                    node_limit = None
                                    impurity_type = impurity
                                    if (prepr == 1):
                                        gain_threshold = thres
                                        chi_threshold = None
                                    else:
                                        chi_threshold = thres
                                        gain_threshold = None
                                    postpruning_type = postpr
                                    prepruning_type = prepr
                                    condition_type = cond
                                    run_classifier ()
                            if (cond == 3) :
                                for n_lim in range (2,5):
                                    imp_threshold = None
                                    tree_depth_limit = None
                                    node_limit = n_lim
                                    if (prepr == 1):
                                        gain_threshold = thres
                                        chi_threshold = None
                                    else:
                                        chi_threshold = thres
                                        gain_threshold = None
                                    impurity_type = impurity
                                    postpruning_type = postpr
                                    prepruning_type = prepr
                                    condition_type = cond
                                    run_classifier ()
                            if (cond == 4) :
                                for depth in range (6,10):
                                    for n_lim in range (2,5):
                                        node_limit = n_lim
                                        gain_threshold = thres
                                        chi_threshold = thres
                                        tree_depth_limit = depth
                                        impurity_type = impurity
                                        postpruning_type = postpr
                                        prepruning_type = prepr
                                        condition_type = cond
                                        run_classifier ()
                            if (cond == 5) :
                                for imp_thr in np.arange (0.01,0.11,0.03):
                                    for depth in range (6,10):
                                        imp_threshold = imp_thr
                                        tree_depth_limit = depth
                                        gain_threshold = thres
                                        chi_threshold = thres
                                        impurity_type = impurity
                                        postpruning_type = postpr
                                        prepruning_type = prepr
                                        condition_type = cond
                                        run_classifier ()
                            if (cond == 6) :
                                for imp_thr in np.arange (0.01,0.11,0.03):
                                    for n_lim in range (2,5):
                                        node_limit = n_lim
                                        imp_threshold = imp_thr
                                        gain_threshold = thres
                                        chi_threshold = thres
                                        impurity_type = impurity
                                        postpruning_type = postpr
                                        prepruning_type = prepr
                                        condition_type = cond
                                        run_classifier ()
                            if (cond == 7) :
                                        for imp_thr in np.arange (0.01,0.11,0.03):
                                            for n_lim in range (2,5):
                                                for depth in range (6,10):
                                                    node_limit = n_lim
                                                    tree_depth_limit = depth
                                                    imp_threshold = imp_thr
                                                    if (prepr == 1):
                                                        gain_threshold = thres
                                                        chi_threshold = None
                                                    else:
                                                        chi_threshold = thres
                                                        gain_threshold = None
                                                    impurity_type = impurity
                                                    postpruning_type = postpr
                                                    prepruning_type = prepr
                                                    condition_type = cond
                                                    run_classifier ()

f.close()
print ("The program is finished!")
