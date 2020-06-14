import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

## features 2,3,4,5,6 seem to give the lowest MSE with decision trees


mpl.rcParams['figure.dpi'] = 500

#reading the csv file
dataset = pd.read_csv('datasets_88705_204267_Real estate.csv')

def plotTrain(X,feature_num,Y,Y_pred):
    plt.title('Plot-per-feature-train')
    plt.scatter(X[:,feature_num],Y,color = 'blue',s = 2.5);
    plt.scatter(X[:,feature_num],Y_pred,color = 'orange',s = 2.5);
    plt.xlabel('Feature ' + str(feature_num))
    plt.ylabel('Price')
    plt.savefig('Feature - ' + str(feature_num) + 'Train',dpi = 500)
    plt.show()
    
def plotTest(X,feature_num,Y,Y_pred):
    plt.title('Plot-per-feature-test')
    plt.scatter(X[:,feature_num],Y,color = 'blue',s = 2.5);
    plt.scatter(X[:,feature_num],Y_pred,color = 'orange',s = 2.5);
    plt.xlabel('Feature ' + str(feature_num))
    plt.ylabel('Price')
    plt.savefig('Feature - ' + str(feature_num) + ' Test',dpi = 500)
    plt.show()


def plotRegressorTrain(regressor,Y,Y_pred):
    plt.title('Train Set')
    plt.scatter(np.array(range(310)),Y,color = 'blue',s = 2.5);
    plt.scatter(np.array(range(310)),Y_pred,color = 'orange' , s = 2.5)
    plt.xlabel('Features')
    plt.ylabel('Price')
    plt.savefig('Feature -  DecisionTree - Train',dpi = 500)
    plt.show()
    

def plotRegressorTest(regressor,Y,Y_pred):
    plt.title('Test Set')    
    plt.scatter(np.array(range(104)),Y,color = 'blue',s = 2.5);
    plt.scatter(np.array(range(104)),Y_pred,color = 'orange' , s = 2.5)
    plt.ylabel('Price')
    plt.xlabel('Features')
    plt.savefig('Feature - '  + ' DecisionTree - Test',dpi = 500)
    plt.show()

X = dataset.iloc[:,[2,3,4,5,6]].values
Y = dataset.iloc[:,[7]].values

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
#     remainder='passthrough'                                         # Leave the rest of the columns untouched
# )
# X = ct.fit_transform(X)

# #to avoid the dummy variable trap
# X = X[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]] #---> encoding the date gave a worse mse

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y.reshape(-1,1))

#splitting the dataset into testSet and training set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train,Y_train)


y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

plotRegressorTrain(regressor,Y_train,y_pred_train)
plotRegressorTest(regressor,Y_test,y_pred_test)

print(regressor.score(X_test,Y_test))

for i in range(0,5):
    plotTest(X_test,i,Y_test,y_pred_test)
    plotTrain(X_train,i,Y_train,y_pred_train)
    
from sklearn.metrics import mean_squared_error 
mse = mean_squared_error(Y_test, y_pred_test)
print(mse)
