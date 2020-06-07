import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl



mpl.rcParams['figure.dpi'] = 500


def allFeaturesTrain(y,y_pred):
    plt.title('Training Set Results(SVR)')
    plt.scatter(np.array(range(400)),y,color = 'blue',s = 2.5,alpha = .5,label = 'Observed')     
    plt.xlabel('Features')
    plt.ylabel('Car Purchase Amount')
    plt.scatter(np.array(range(400)),y_pred,color = 'orange',s = 2.5,alpha = .5,label = 'Predicted')
    plt.legend(loc = 'upper right',fancybox=True, framealpha=1, shadow=True, borderpad=1,prop={'size': 7})
    plt.savefig('TrainingSet.png', dpi = 500)           
    plt.show()
def allFeaturesTest(y,y_pred):
    plt.title('Test Set Results(SVR)')
    plt.scatter(np.array(range(100)),y,color = 'blue',s = 2.5,alpha = .5,label = 'Observed')     
    plt.scatter(np.array(range(100)),y_pred,color = 'orange',s = 2.5,alpha = .5,label = 'Predicted')           
    plt.xlabel('Features')
    plt.ylabel('Car Purchase Amount')
    plt.legend(loc = 'upper right',fancybox=True, framealpha=1, shadow=True, borderpad=1,prop={'size': 7})
    plt.savefig('TestSet.png', dpi = 500)    
    plt.show()


dataset = pd.read_csv('Car_Purchasing_Data.csv',encoding = 'latin-1')


X = dataset.iloc[:,[4,5,7]].values;
Y = dataset.iloc[:,8].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y.reshape(-1,1))

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .2,random_state = 0)




from sklearn.svm import SVR

regressor = SVR(kernel= 'linear').fit(X_train,Y_train)
y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)
allFeaturesTrain(Y_train,y_pred_train)
allFeaturesTest(Y_test, y_pred)
y_print = sc_y.inverse_transform(regressor.predict(X_test))
output_array = np.array(y_print)
np.savetxt("predictions.csv", output_array, delimiter=",")

'''from sklearn.metrics import mean_squared_error 
mse = mean_squared_error(Y_test, y_pred)
print(mse)'''