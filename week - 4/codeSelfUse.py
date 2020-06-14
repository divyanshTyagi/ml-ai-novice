import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



mpl.rcParams['figure.dpi'] = 500

        
#reading the csv file
dataset = pd.read_csv('datasets_88705_204267_Real estate.csv')

def plotRegressorTrain(regressor,Y,Y_pred):
    plt.title('Train Set')
    plt.scatter(np.array(range(310)),Y,color = 'blue',s = 0.25,alpha = 0.8);
    plt.scatter(np.array(range(310)),Y_pred,color = 'orange' , s = .25,alpha = 0.8)
    plt.xlabel('Features')
    plt.ylabel('Price')
    plt.savefig('Feature -  Multivariate - Train',dpi = 500)
    plt.show()
    

def plotRegressorTest(regressor,Y,Y_pred):
    plt.title('Test Set')    
    plt.scatter(np.array(range(104)),Y,color = 'blue',s = 0.25,alpha = 0.8);
    plt.scatter(np.array(range(104)),Y_pred,color = 'orange' , s = .25,alpha = 0.8)
    plt.ylabel('Price')
    plt.xlabel('Features')
    plt.savefig('Feature - '  + ' Multivariate - Test',dpi = 500)
    plt.show()

X = dataset.iloc[:,[1,2,3,4,5,6]].values
Y = dataset.iloc[:,[7]].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)

#to avoid the dummy variable trap
X = X[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]] #---> encoding the date gave a worse mse
#appending table of ones 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y.reshape(-1,1))


X = np.append(arr = np.ones(shape = (414,1)).astype(int),values = X,axis = 1)




#splitting the dataset into testSet and training set
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .25,random_state = 0)

import statsmodels.api as sm
X_opt_train = np.array(X_train[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]],dtype = int)
X_opt_test = np.array(X_test[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]],dtype = int)
regressor_OLS = sm.OLS(endog = Y_train , exog = X_opt_train).fit()
y_pred_test = regressor_OLS.predict(X_opt_test).astype(int)
y_pred_train = regressor_OLS.predict(X_opt_train).astype(int)

plotRegressorTest(regressor_OLS,Y_test,y_pred_test)
plotRegressorTrain(regressor_OLS, Y_train, y_pred_train)


print(regressor_OLS.summary())





