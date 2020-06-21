import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


#reading the csv file
dataset = pd.read_csv('Diabetes_Xtest.csv')
X_test = pd.read_csv('Diabetes_Xtest.csv').iloc[:,[0,1,2,4,5,6,7]].values
X_train = pd.read_csv('Diabetes_XTrain.csv').iloc[:,[0,1,2,4,5,6,7]].values
Y = pd.read_csv('Diabetes_Ytrain.csv').iloc[:,:].values
lst = list(dataset.columns)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting the classifier to the training set 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski',p = 2)
knn.fit(X_train,Y)

y_print = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y,knn.predict(X_train))

output_array = np.array(y_print)
np.savetxt("predictions.csv", output_array, delimiter=",")


