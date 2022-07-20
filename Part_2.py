import numpy as np
from sklearn import tree
import os
import pydotplus
import csv
from six import StringIO
import sklearn as skl
from sklearn.metrics import accuracy_score

##################### FUNCTIONS ########################

def relative_accuracy_score(Test_Classes, Actual_Classes):
    global relative_distance
    error = 0
    for x in range (0,len(Actual_Classes)):
        if (Test_Classes[x] != Actual_Classes[x]) and (Test_Classes[x] != (Actual_Classes[x]-relative_distance)) and (Test_Classes[x] != (Actual_Classes[x] + relative_distance)):
            error = error + 1

    relative_accuracy = (1 - (error/len(Actual_Classes)))
    return relative_accuracy

################### MAIN #################################

names = ['Age', 'Sex', 'Pain', 'Pressure', 'Cholesterol', 'Sugar', 'Results', 'Max Heart Rate', 'Angina','Old Peak','Slope','Num of Major Vessels','Thal']
NumOfFeatures = len(names)
labels = ['0', '1', '2', '3', '4', '5']
filename = 'cleveland.data'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x).astype('float')
relative_distance = 1
repetitions = 100
automatic = 0
accuracy_rep = []
relative_accuracy_rep = []
f= open("Results_2.txt","w+")

data_labels = data[:,-1]
data = data[:,0:13]

automatic = int(input('Enter 1 to run all cases or 0 to run a particular case : '))
while (automatic < 0) or (automatic > 1):
    automatic = int(input('Enter 1 to run all cases or 0 to run a particular case : '))

repetitions = int(input('Enter the number that the experiment will be repeated (Small value is faster but less accurate) : '))
while (repetitions <= 0) :
    repetitions = int(input('Enter the number that the experiment will be repeated (Small value is faster but less accurate) : '))

if (automatic == 0):

    accuracy_rep = []
    relative_accuracy_rep = []

    print ("Enter the necessary parameters : \n")

    critirion = int(input('Choose the quality criterion (0 for gini and 1 for entropy) (default : Gini) : '))
    while (critirion != 0) and (critirion != 1):
        critirion = int(input('Choose the quality criterion (0 for gini and 1 for entropy) (default : Gini) : '))

    if (critirion == 0):
        critirion = "gini"
    else :
        critirion = "entropy"

    max_depth = int(input('Enter the max depth of the tree (>=1 or -1 for no limit) (default : No limit) : '))
    while (max_depth <= 0) and (max_depth != -1):
        max_depth = int(input('Enter the max depth of the tree (>=1 or -1 for no limit) (default : No limit) : '))

    if (max_depth == -1):
        max_depth = None

    splitter = int(input('Enter the spliting strategy of the tree (0 for best and 1 for random) (default : best) : '))
    while (splitter != 0) and (splitter != 1):
        splitter = int(input('Enter the spliting strategy of the tree (0 for best and 1 for random) (default : best) : '))

    if (splitter == 0):
        splitter = "best"
    else :
        splitter = "random"

    min_samples_split  = int(input('Enter the minimum number of samples required to split an internal node (>1) (default : 2) : '))
    while (min_samples_split <= 1):
        min_samples_split  = int(input('Enter the minimum number of samples required to split an internal node (>1) (default : 2) : '))

    min_samples_leaf = int(input('Enter the minimum number of samples required to be at a leaf node (>=1) (default : 1): '))
    while (min_samples_leaf < 1):
        min_samples_leaf   = int(input('Enter the minimum number of samples required to be at a leaf node (>=1) (default : 1): '))

    max_leaf_nodes  = int(input('Enter the maximum number of leaves allowed (>=2 or -1 for no limit) (default : No limit): '))
    while (max_leaf_nodes  < 2) and (max_leaf_nodes != -1):
        max_leaf_nodes  = int(input('Enter the maximum number of leaves allowed (>=2 or -1 for no limit) : (default : No limit) '))

    if (max_leaf_nodes == -1):
        max_leaf_nodes = None

    min_impurity_decrease   = float(input('Enter the minimum decrease of the impurity (>=0) (default : 0) : '))
    while (min_impurity_decrease  < 0):
        min_impurity_decrease  = float(input('Enter the minimum decrease of the impurity (>=0) (default : 0): '))

    presort  = int(input('Choose whether to pre-sort the data (1) or not (0) (default : 0) : '))
    while (presort != 0) and (presort != 1):
        presort  = int(input('Choose whether to pre-sort the data (1) or not (0) (default : 0) : '))

    if (presort == 0):
        presort = False
    else :
        presort = True

    max_features = int(input('Enter the number of features to consider when looking for the best split (>=1 & <=12 or -1 for all available features) : (default : -1) : '))
    while ((max_features <= 0) and (max_features != -1)) and (max_features <= NumOfFeatures) :
        max_features = int(input('Enter the number of features to consider when looking for the best split (>=1 & <=12 or -1 for all available features) (default : -1) :'))

    if (max_features == -1):
        max_features = None

    test_size = float(input('Enter the percentage of the dataset, that will be used for testing (0 < test_size < 1) : '))
    while (test_size <= 0) or (test_size >= 1) :
        test_size = float(input('Enter the percentage of the dataset, that will be used for testing (0 < test_size < 1) : '))

    print_tree = int(input('Do you want to print the tree (Data_*.pdf) for each repetition ? (0 for No, 1 for Yes) : '))
    while (print_tree < 0) or (print_tree > 1) :
        print_tree = int(input('Do you want to print the tree (Data_*.pdf) for each repetition ? (0 for No, 1 for Yes) : '))

    print('')

    print ("Please wait. This may take a few minutes.....")

    for k in range(0,repetitions):

        if (print_tree == 1):
            os.system("del Data_" + str(k) + ".pdf")

        X_train, X_test, y_train, y_test = skl.model_selection.train_test_split (data, data_labels, test_size = 0.1, random_state = 100)

        clf = tree.DecisionTreeClassifier(criterion = critirion,splitter= splitter,max_depth = max_depth,min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf,max_leaf_nodes = max_leaf_nodes,min_impurity_decrease = min_impurity_decrease,presort = presort,max_features = max_features)
        clf.fit(X_train, y_train)

        # Cheating

        '''
        y_pred = clf.predict(X_train)
        print ("Accuracy is ", accuracy_score(y_train,y_pred)*100,"%")
        print ("Relative Accuracy is ", relative_accuracy_score(y_train,y_pred)*100,"%")
        '''

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)*100
        relative_accuracy = relative_accuracy_score(y_test,y_pred)*100
        #print ("Accuracy is ",accuracy,"%")
        #print ("Relative Accuracy is ",relative_accuracy,"%")
        accuracy_rep.append(accuracy)
        relative_accuracy_rep.append(relative_accuracy)

        if (print_tree == 1):
            if (k <= 10):
                dot_data = StringIO()
                tree.export_graphviz(clf,out_file=dot_data,feature_names = names , class_names = labels , filled = True , rounded = True , impurity = False )
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                graph.write_pdf("Data_" + str(k) + ".pdf")
                os.system("start Data_" + str(k) + ".pdf")

    # Final Results

    Final_Accuracy = sum(accuracy_rep)/repetitions
    Final_Relative_Accuracy = sum(relative_accuracy_rep)/repetitions

    f.write("================================================================================ \n")
    f.write ("(Critirion,Max Depth,Splitter,Min Internal Node Data,Min Leaf Node Data,Max Leaf Nodes,Min Impurity Decrease,Presort,Max Features to use,Test Size) = (" + str(critirion) + "," + str(max_depth) + ',' + str(splitter) + ',' + str(min_samples_split) +  ',' + str(min_samples_leaf) + ',' + str(max_leaf_nodes) +  ',' + str(min_impurity_decrease) + ',' + str(presort) + ',' + str(max_features) + ',' +  str(test_size) +")")
    f.write ("\nFinal Accuracy : {:.2%}".format(Final_Accuracy/100) + "\nFinal Relative Accuracy : {:.2%}".format(Final_Relative_Accuracy/100))
    f.write("\n================================================================================\n")

else:
    accuracy_limit = float(input('Choose the lower bound of the accuracy of the classifiers you wish to see (0 - 100) : '))
    while (accuracy_limit < 0) or (accuracy_limit > 100):
        accuracy_limit = float(input('Choose the lower bound of the accuracy of the classifiers you wish to see (0 - 100) : '))

    print ("Please wait. This may take a few minutes.....")

    for critirion in range(0,2):
        for splitter in range(0,2):
            for max_depth in range(5,20):
                for min_samples_split in range(2,5):
                    for min_samples_leaf in range(1,3):
                        for max_leaf_nodes in range (30,50,4):
                            for min_impurity_decrease in np.arange (0.01,0.02,0.01):
                                for presort in range(0,2):
                                    for max_features in range(1,NumOfFeatures+1):
                                        for test_size in np.arange (0.1,0.4,0.1):

                                            if (critirion == 0):
                                                critirion = "gini"
                                            else :
                                                critirion = "entropy"

                                            if (max_depth == -1):
                                                max_depth = None

                                            if (splitter == 0):
                                                splitter = "best"
                                            else :
                                                splitter = "random"

                                            if (max_leaf_nodes == -1):
                                                max_leaf_nodes = None

                                            if (presort == 0):
                                                presort = False
                                            else :
                                                presort = True

                                            if (max_features == -1):
                                                max_features = None

                                            accuracy_rep = []
                                            relative_accuracy_rep = []

                                            for k in range(0,repetitions):

                                                    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split (data, data_labels, test_size = test_size, random_state = 100)

                                                    clf = tree.DecisionTreeClassifier(criterion = critirion,splitter= splitter,max_depth = max_depth,min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf,max_leaf_nodes = max_leaf_nodes,min_impurity_decrease = min_impurity_decrease,presort = presort,max_features = max_features)
                                                    clf.fit(X_train, y_train)

                                                    # Cheating

                                                    y_pred = clf.predict(X_test)
                                                    accuracy = accuracy_score(y_test,y_pred)*100
                                                    relative_accuracy = relative_accuracy_score(y_test,y_pred)*100
                                                    #print ("Accuracy is ",accuracy,"%")
                                                    #print ("Relative Accuracy is ",relative_accuracy,"%")
                                                    accuracy_rep.append(accuracy)
                                                    relative_accuracy_rep.append(relative_accuracy)

                                            #Final Results

                                            Final_Accuracy = sum(accuracy_rep)/repetitions
                                            Final_Relative_Accuracy = sum(relative_accuracy_rep)/repetitions

                                            if (Final_Accuracy >= accuracy_limit):

                                                f.write("================================================================================ \n")
                                                f.write ("(Critirion,Max Depth,Splitter,Min Internal Node Data,Min Leaf Node Data,Max Leaf Nodes,Min Impurity Decrease,Presort,Max Features to use,Test Size) = (" + str(critirion) + "," + str(max_depth) + ',' + str(splitter) + ',' + str(min_samples_split) +  ',' + str(min_samples_leaf) + ',' + str(max_leaf_nodes) +  ',' + str(min_impurity_decrease) + ',' + str(presort) + ',' + str(max_features) + ',' +  str(test_size) +")")
                                                f.write ("\nFinal Accuracy : {:.2%}".format(Final_Accuracy/100) + "\nFinal Relative Accuracy : {:.2%}".format(Final_Relative_Accuracy/100))
                                                f.write("\n================================================================================\n")


f.close()
print ("The program is finished!")
