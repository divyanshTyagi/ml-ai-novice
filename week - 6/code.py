import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import accuracy_score as s
from sklearn.metrics import average_precision_score


dataset = pd.read_csv('pulsar_stars.csv')
x = dataset.iloc[:,[0,1,2,3,4,5,6,7]].values
y = dataset.iloc[:,[8]].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .2,random_state = 100)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# # #using knn classification 
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 6, p = 2, metric = "minkowski")
# classifier.fit(x_train, y_train.ravel())
# y_pred = classifier.predict(x_test)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test,y_pred)
# print(s(y_test,y_pred))


# average_precision = average_precision_score(y_test, y_pred)


# print('Average precision-recall score: {0:0.2f}'.format(average_precision))


# # *********************************************************************************

# from sklearn.svm import SVC

# classifier = SVC(kernel = 'rbf',random_state = 0)
# classifier.fit(x_train,y_train.ravel())
# y_pred = classifier.predict(x_test)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test,y_pred)

# print(s(y_test,y_pred))
# average_precision = average_precision_score(y_test, y_pred)
# print('Average precision-recall score: {0:0.2f}'.format(average_precision))

# # *********************************************************************************


# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(x_train,y_train.ravel())

# y_pred = classifier.predict(x_test)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test,y_pred)
# average_precision = average_precision_score(y_test, y_pred)
# print(s(y_test,y_pred))
# print('Average precision-recall score: {0:0.2f}'.format(average_precision))

# # *********************************************************************************

# from sklearn.tree import DecisionTreeClassifier

# classifier = DecisionTreeClassifier(criterion = "entropy",random_state = 0)
# classifier.fit(x_train,y_train)
# y_pred = classifier.predict(x_test)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test,y_pred)
# average_precision = average_precision_score(y_test, y_pred)
# print(s(y_test,y_pred))
# print('Average precision-recall score: {0:0.2f}'.format(average_precision))

# # *********************************************************************************

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100,criterion = 'entropy',random_state = 0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
average_precision = average_precision_score(y_test, y_pred)
print("score is " ,s(y_test,y_pred))
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
from sklearn.metrics import f1_score

print(" F1 Score = " , f1_score(y_test, y_pred))


















