import warnings
warnings.filterwarnings("ignore")
import numpy as np
import csv
import xgboost as xgb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn import linear_model
import sys

################ FUNCTIONS ###############

def NeuralNetworkClassifier() :

    # Normalize input data (standardization)
    def normalize(xdata):
        global m, s  #The mean (m) and std (s) of the data (xdata)
        m = np.mean(xdata,axis=0)  # mean
        s = np.std(xdata,axis=0)  # standard deviation
        x_norm = (xdata - m) / s
        return(x_norm)

    # Define the linear model
    def combine_inputs(X):
        Y_predicted_linear = tf.matmul(X, W) + b
        return Y_predicted_linear

    # Define the sigmoid inference model over the data X and return the result
    def inference(X):
        Y_prob = tf.nn.softmax(combine_inputs(X)) #Defines the output of the SoftMax (Probabilities)
        Y_predicted = tf.argmax(Y_prob, axis = 1, output_type=tf.int32) #Get the output with the largest probability
        return Y_prob, Y_predicted

    # Compute the loss over the training data using the predictions and true labels Y
    def loss(X, Y):
        Yhat = combine_inputs(X)
        SoftMaxCE = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Yhat, labels=Y)
        loss =  tf.reduce_mean(SoftMaxCE)
        return loss

    # Optimizer
    def train(total_loss):
        learning_rate = 0.1
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        trainable_variables = tf.trainable_variables()
        update_op = optimizer.minimize(total_loss, var_list=trainable_variables)
        return update_op

    def evaluate(Xtest, Ytest):
        Y_prob, Y_predicted = inference(Xtest)
        accuracy= tf.reduce_mean(tf.cast(tf.equal(Y_predicted, Ytest), tf.float32))
        return accuracy

    def inputs():
        # Load data
        filename = 'cleveland.data'
        raw_data = open(filename, 'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        data = np.array(x).astype('float')

        Y_np = data[:,-1]
        data = data[:,0:13]

        X_np = normalize(data)
        return X_np, Y_np

    #Initializations
    num_features = 13
    output_dim = 5
    batch_size = 10

    with tf.variable_scope("other_charge", reuse=tf.AUTO_REUSE) as scope:

        # Variables of the model
        W = tf.get_variable(name='W', shape=(num_features, output_dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b', shape=(output_dim, ), dtype=tf.float32, initializer=tf.constant_initializer(value=0, dtype=tf.float32))

        # Input - output placeholders
        X = tf.placeholder(shape=(batch_size, num_features), dtype=tf.float32)  # Placeholder for a batch of input vectors
        Y = tf.placeholder(shape=(batch_size, ), dtype=tf.int32)  # Placeholder for target values

        # We have a better result with small test set and bigger training set
        X_np,Y_np = inputs()
        X_np, test_data, Y_np, test_labels = train_test_split (X_np, Y_np, test_size = test_size, random_state = 100)

        init_op = tf.global_variables_initializer()

        # Execution: Training and Evaluation of the model
        with tf.Session() as sess:
            sess.run(init_op)
            num_epochs = 50
            num_examples = X_np.shape[0] - batch_size + 1
            total_loss = loss(X,Y)
            train_op = train(total_loss)
            perm_indices = np.arange(num_examples)

            for epoch in range(num_epochs):
                epoch_loss = 0
                np.random.shuffle(perm_indices)

                for i in range(num_examples-batch_size+1):
                    X_batch = X_np[perm_indices[i:i+batch_size], :]
                    Y_batch = Y_np[perm_indices[i:i+batch_size]]
                    feed_dict = {X: X_batch, Y: Y_batch}
                    batch_loss, _ = sess.run([total_loss, train_op] , feed_dict)
                    epoch_loss += batch_loss

                epoch_loss /= num_examples

            # Start the Evaluation based on the trained model
            Xtest = tf.placeholder(shape=(None, num_features), dtype=tf.float32) # Placeholder for one input vector
            Ytest = tf.placeholder(shape=(None, ), dtype=tf.int32)

            Ytest_prob, Ytest_predicted = inference(Xtest) # Define the graphs for the inference (probability) and prediction (binary)
            feed_dict_test = {Xtest: test_data, Ytest: test_labels.astype(int)}
            accuracy_np = evaluate (Xtest,Ytest)

            return "NeuralNetworkSoftMaxClassifier" , sess.run(accuracy_np, feed_dict_test)

def relative_accuracy_score(Test_Classes, Actual_Classes):
    global relative_distance
    error = 0
    for x in range (0,len(Actual_Classes)):
        if (Test_Classes[x] != Actual_Classes[x]) and (Test_Classes[x] != (Actual_Classes[x]-1)) and (Test_Classes[x] != (Actual_Classes[x] + 1)):
            error = error + 1

    relative_accuracy = (1 - (error/len(Actual_Classes)))
    return relative_accuracy

################# MAIN ###################

relative_distance = 1

repetitions = int(input('Enter the number that the experiment will be repeated (Small value is faster but less accurate) : '))
while (repetitions <= 0) :
    repetitions = int(input('Enter the number that the experiment will be repeated (Small value is faster but less accurate) : '))

f= open("Results_3.txt","w+")
filename = 'cleveland.data'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x).astype('float')
data_labels = data[:,-1]
data = data[:,0:13]
# We have a better result with small test set and bigger training set

print ("Please wait. This may take a few minutes.....")

classifiers = [
    xgb.XGBClassifier(objective ='multi:softmax', colsample_bytree = 0.3, learning_rate = 1,max_depth = 20, alpha = 10, n_estimators = 10,num_class =5),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    LinearSVC(),
    #linear_model.SGDClassifier(max_iter=10000, tol=1e-3),
    SVC(kernel="linear", C=0.025),
    NuSVC(probability=True,nu=0.1),
    RandomForestClassifier(max_depth=3, n_estimators=10, max_features=13),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

for test_size in np.arange(0.1,0.6,0.1):
    best_classifier_name = 0
    best_classifier_acc = 0

    classifiers_total_accuracies = {clf:0 for clf in classifiers}
    classifiers_total_relative_accuracies = {clf:0 for clf in classifiers}
    total_neural_network_acc = 0

    X_train, X_test, y_train, y_test = train_test_split (data, data_labels, test_size = test_size, random_state = 100)

    for k in range(0,repetitions):
        for clf in classifiers:
            clf.fit(X_train, y_train)
            name = clf.__class__.__name__

            #print("="*30)
            #print(name)

            #print('****Results****')

            # Cheating

            #train_predictions = clf.predict(X_train)
            #acc = accuracy_score(y_train,train_predictions)

            train_predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, train_predictions)
            rel_acc = relative_accuracy_score(y_test, train_predictions)

            classifiers_total_accuracies[clf] = classifiers_total_accuracies[clf] + acc
            classifiers_total_relative_accuracies[clf] = classifiers_total_relative_accuracies[clf] + rel_acc

            if (acc > best_classifier_acc):
                best_classifier_acc = acc
                best_classifier_name = name
                best_rel_acc = rel_acc

            #print("Accuracy: {:.4%}".format(acc))
            #print("Relative Accuracy: {:.4%}".format(rel_acc))

        #### Neural Network #########
        name, accuracy = NeuralNetworkClassifier()

        total_neural_network_acc = total_neural_network_acc + accuracy

        #print("="*30)
        #print(name)
        #print('****Results****')
        #print("Accuracy: {:.4%}".format(accuracy))

        if (accuracy > best_classifier_acc):
            best_classifier_acc = accuracy
            best_classifier_name = name

        #print("="*30)

    f.write("================================================================================ \n")
    f.write("(Test Size) = (" + str(test_size*100) + "%)\n")
    f.write("\nTotal Accuracy per Classifier : \n")
    for clf, acc in classifiers_total_accuracies.items():
        f.write(str(clf.__class__.__name__)  +  " : " +  str((acc/repetitions)*100) + "%\n")
    f.write("SoftMaxNeuralNetworkClassifier : " + str((total_neural_network_acc/repetitions)*100) + "%\n")

    f.write("\nTotal Relative Accuracy per Classifier : \n")
    for clf, rel_acc in classifiers_total_relative_accuracies.items():
        f.write(str(clf.__class__.__name__)  +  " : " +  str((rel_acc/repetitions)*100) + "%\n")

    f.write("\n================================================================================\n")

#print ("\nThe best classifier is",best_classifier_name,"with accuracy of {:.4%}".format(best_classifier_acc),"and best relavitve accuracy",best_rel_acc*100,"%")
f.close()
print("The program is finished !")
