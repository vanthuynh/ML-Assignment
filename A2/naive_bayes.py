#-------------------------------------------------------------------------
# AUTHOR: Van Huynh
# FILENAME: naive_bayes.py
# SPECIFICATION: Program that read csv file and output a decision tree
# FOR: CS 4210- Assignment #2
# TIME SPENT: 120 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

dbTraining = []

#reading the training data
#--> add your Python code here

with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            dbTraining.append(row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []

outlook = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity = {'High': 1, 'Normal': 2}
wind = {'Weak': 1, 'Strong': 2}
classification = {'Yes': 1, 'No': 2}

for row in dbTraining:
    X.append([outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]])
    Y.append(classification[row[5]])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

dbTest = []
#reading the data in a csv file
#--> add your Python code here

with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            dbTest.append(row)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]

for row in dbTest:
    prediction = clf.predict_proba([[outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]]])[0]
    most_probable = 0
    most_probable_class = ""
    if prediction[0] > prediction[1]:
        most_probable = prediction[0]
        most_probable_class += "Yes"
    else:
        most_probable = prediction[1]
        most_probable_class += "No"
    if most_probable >= 0.75:
        print('{:15s}{:15s}{:15s}{:15s}{:15s}{:15s}{:7.8f}'.format(row[0], row[1], row[2], row[3], row[4], most_probable_class, most_probable))