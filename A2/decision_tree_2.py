#-------------------------------------------------------------------------
# AUTHOR: Van Huynh
# FILENAME: decision_tree_2.py
# SPECIFICATION: Program that read csv file and output a decision tree
# FOR: CS 4210- Assignment #2
# TIME SPENT: 120 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []
    num_attributes = 4
    accuracy = 1

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    for i in dbTraining:
        row = []
        for count, x in enumerate(i):
            if(count != num_attributes):
                if(x == "Young"):
                    row.append(1)
                elif (x == "Presbyopic"):
                    row.append(2)
                elif (x == "Prepresbyopic"):
                    row.append(3)
                elif (x == "Myope"):
                    row.append(1)
                elif (x == "Hypermetrope"):
                    row.append(2)
                elif (x == "No"):
                    row.append(1)
                elif (x == "Yes"):
                    row.append(2)
                elif (x == "Reduced"):
                    row.append(1)
                elif (x == "Normal"):
                    row.append(2)
        #print(row)
        X.append(row)

    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    Y =[0] * len(dbTraining)
    for count, i in enumerate(dbTraining):
        if(i[len(i)-1] == "Yes"):
            Y[count] = 1
        else:
            Y[count] = 2

    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)
                    #print(row)

        true_positive = 0
        true_negative = 0
        total_count = 0
        for z, data in enumerate(dbTest):
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            test_data = []
            row = []
            for num, att in enumerate(data):
                if(num != num_attributes):
                    if(att == "Young"):
                        row.append(1)
                    elif (att == "Presbyopic"):
                        row.append(2)
                    elif (att == "Prepresbyopic"):
                        row.append(3)
                    elif (att == "Myope"):
                        row.append(1)
                    elif (att == "Hypermetrope"):
                        row.append(2)
                    elif (att == "No"):
                        row.append(1)
                    elif (att == "Yes"):
                        row.append(2)
                    elif (att == "Reduced"):
                        row.append(1)
                    elif (att == "Normal"):
                        row.append(2)
            test_data.append(row)
            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            predicted = clf.predict(test_data)[0]
            true_label = 1
            if(data[4] == "No"):
                true_label = 2

            if(true_label == 1 and predicted == 1):
                true_positive += 1
            if(true_label == 2 and predicted == 2):
                true_negative += 1

            total_count += 1

        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
        temp_acc = (true_positive + true_negative) / total_count
        if(temp_acc < accuracy):
            accuracy = temp_acc
    #print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("Final accuracy when training on " + str(ds) + ": " +  str(accuracy))