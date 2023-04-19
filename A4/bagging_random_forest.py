#-------------------------------------------------------------------------
# AUTHOR: Van Huynh
# FILENAME: baggin_random_forest.py
# SPECIFICATION:
# FOR: CS 4210- Assignment 4
# TIME SPENT: an hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
classVotes = [] #this array will be used to count the votes of each classifier
test_labels = []
accuracy = 1
z = 1

#reading the training data from a csv file and populate dbTraining
#--> add your Python code here
with open('optdigits.tra', 'r') as trainingFile:
   reader = csv.reader(trainingFile)
   for i, row in enumerate(reader):
      dbTraining.append (row)

#reading the test data from a csv file and populate dbTest
#--> add your Python code here

#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
#--> add your Python code here
with open('optdigits.tes', 'r') as testingFile:
   reader = csv.reader(testingFile)
   for i, row in enumerate(reader):
      dbTest.append (row)
      classVotes.append([0,0,0,0,0,0,0,0,0,0]) #inititalizing the class votes for each test sample

print("Started my base and ensemble classifier ...")

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

   bootstrapSample = resample(dbTraining, n_samples=len(dbTraining),  replace=True)

   #populate the values of X_training and Y_training by using the bootstrapSample
   #--> add your Python code here
   for array in bootstrapSample:
      row = []
      yRow = []
      for i, value in enumerate(array):
            if(i == len(array) - 1):
               Y_training.append(value)
            else:
               row.append(value)
      X_training.append(row)
   #fitting the decision tree to the data
   clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
   clf = clf.fit(X_training, Y_training)

   numCorrect = 0
   for i, testSample in enumerate(dbTest):
      X_test = []
      trueLabel = 10
      row = []
      for x, value in enumerate(testSample):
         if(x == len(array) - 1):
               trueLabel = value
               test_labels.append(value)
         else:
               row.append(value)
      X_test.append(row)
      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      #--> add your Python code here
      prediction = clf.predict(X_test)[0]
      classVotes[i][int(prediction)] += 1

      if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
         #--> add your Python code here
         if(prediction == trueLabel):
            numCorrect +=1
            accuracy = numCorrect/ len(dbTest)

   if k == 0: #for only the first base classifier, print its accuracy here
      #--> add your Python code here
      print("Finished my base classifier (fast but relatively low accuracy) ...")
      print("My base classifier accuracy: " + str(accuracy))
      print("")

  #now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
  #--> add your Python code here
numCounter = 0
for num, array in enumerate(classVotes):
   max = 0
   maxIndex = 0
   for index, value in enumerate(array):
      if(value > max):
         max = value
         maxIndex = index
   if(maxIndex == int(test_labels[num])):
      numCounter += 1

print(numCounter)
accuracy = numCounter / len(dbTest)

#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

#Fit Random Forest to the training data
clf.fit(X_training,Y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
#--> add your Python code here
numCounter = 0
trueLabel = 100
for i, testSample in enumerate(dbTest):
   #print(testSample)
   X_test = []
   row = []
   for x, value in enumerate(testSample):
      if(x == len(testSample) - 1):
         trueLabel = value
      else:
         row.append(value)
   X_test.append(row)

   predict_Rf = clf.predict(X_test)[0]
   #print(str(predict_Rf) + " " + str(trueLabel))

#compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
#--> add your Python code here
   if(int(predict_Rf) == int(test_labels[i])):
      numCounter += 1

accuracy = numCounter / len(dbTest)
#printing Random Forest accuracy here
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
