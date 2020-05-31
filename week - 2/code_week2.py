import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def make_graph(num_feature,regressor_OLS,X,X_opt,Y):
    plt.scatter(X[:,[num_feature]],regressor_OLS.predict(X_opt),s=2.5,color = 'blue',label = 'predicted',alpha = .8)
    plt.scatter(X[:,num_feature],Y,color = 'orange',label = 'actual',s=2.5,alpha = 0.8)
    plt.xlabel(str('Feature ' + str (num_feature))) 
    plt.ylabel('Air Quality Parameter')
    plt.title('Air Quality Index Prediction Model')
    plt.legend(loc = 'upper right',fancybox=True, framealpha=1, shadow=True, borderpad=1,prop={'size': 7})
    plt.savefig(str('Features - ' + str(num_feature) + '.png'),dpi = 500)
    plt.show()

dataset = pd.read_csv('Train.csv')
toTest = pd.read_csv('Test.csv')


X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,5].values

#backward elimination 
#stats model doesnt take into account the b0 term, so we will add a column to out dataset having all cells as 1 
import statsmodels.api as sm
X = np.append(arr = np.ones(shape = (1600,1)).astype(int) , values = X,axis = 1)
toTest = np.append(arr = np.ones(shape = (400,1)).astype(int) , values = toTest,axis = 1)

#matrix conatining optimal variable having high statistical sign
X_opt = np.array(X[:, [0,1,2,3,4,5]],dtype = float)
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
y_pred_OLS = regressor_OLS.predict(toTest)

output_array = np.array(y_pred_OLS)
np.savetxt("output.csv", output_array, delimiter=",")


#drawing graphs for features

for i in range(1,6):
    make_graph(i, regressor_OLS, X, X_opt, Y)