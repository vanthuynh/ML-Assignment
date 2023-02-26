#-------------------------------------------------------------------------
# AUTHOR: Van Huynh
# FILENAME: knn.py
# SPECIFICATION: Program that read csv file and output a decision tree
# FOR: CS 4210- Assignment #2
# TIME SPENT: 120 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
X = []
Y = []

correctP = 0
incorrectP = 0

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)


#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):
    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    X = []
    Y = []
    test_sample = []
    true_label = 2
    for count, inst in enumerate(db):
        row = []
        if(count != i ):
            for x, att in enumerate(inst):
                if(x != len(inst)-1):
                    row.append(float(att))
                else:
                    if(att == '-'):
                        Y.append(float(1))
                    else:
                        Y.append(float(2))
            X.append(row)
        else:
            row = []
            for x, att in enumerate(inst):
                if(x != len(inst)-1):
                    row.append(float(att))
                else:
                    if(att == '-'):
                        true_label = 1
            test_sample.append(row)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    #testSample =

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict(test_sample)[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if(class_predicted == true_label):
        correctP += 1
    else:
        incorrectP += 1
#print the error rate
#--> add your Python code here
errorRate = incorrectP/correctP
print(errorRate)