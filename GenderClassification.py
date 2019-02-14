from sklearn import tree;
from sklearn.svm import SVC;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.metrics import accuracy_score
import numpy as np

# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#Classifiers
clfTree = tree.DecisionTreeClassifier()
clfSVC = SVC(gamma='auto')
clfNeighbour = KNeighborsClassifier(n_neighbors=3)

# Training the models
clfTree.fit(X, Y)
clfSVC.fit(X, Y)
clfNeighbour.fit(X, Y)

#Predicting using X data and getting the accuracy score
prediction = clfTree.predict(X)
treeAccScore = accuracy_score(Y, prediction) * 100
print(treeAccScore);

#Predicting using X data and getting the accuracy score
prediction = clfSVC.predict(X)
svcAccScore = accuracy_score(Y, prediction) * 100
print(svcAccScore);

#Predicting using X data and getting the accuracy score
prediction = clfNeighbour.predict(X)
neighbourAccScore = accuracy_score(Y, prediction) * 100
print(neighbourAccScore);

#Comparing the best model
index =np.argmax([treeAccScore, svcAccScore, neighbourAccScore])
classifiers = {0: 'Decision Tree', 1: 'SVC', 2: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))
