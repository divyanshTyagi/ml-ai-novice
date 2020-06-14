import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns



#function to draw initial plots of data
def plot(X,feature_num,Y):
    plt.title('Introductory plot')
    plt.scatter(X[:,feature_num],Y,color = 'blue',s = 0.25,alpha = 0.8);
    plt.xlabel('Feature ' + str(feature_num))
    plt.ylabel('Price')
    plt.savefig('Feature - ' + str(feature_num),dpi = 500)
    plt.show()
        
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

mpl.rcParams['figure.dpi'] = 500

#reading the csv file
dataset = pd.read_csv('datasets_88705_204267_Real estate.csv')

dataset['X1 transaction date'].value_counts().plot(kind = 'bar')
plt.title('Transaction date')
plt.xlabel('date',fontsize = 5)
plt.ylabel('Price')
plt.savefig('Transaction Date',dpi = 500)
plt.show()
#observation - The price with the date 2013.471 is maximum

dataset['X5 latitude'].value_counts().plot(kind = 'bar')
plt.title('latitude')
plt.xlabel('latitude',fontsize = 5)
plt.ylabel('Price')
plt.savefig('latitude-bar',dpi = 500)
plt.show()

#plotting longitude and latitude
#using sns (Seaborn)
plt.figure(figsize = (10,10))
sns.jointplot(x = dataset['X5 latitude'].values,y = dataset['X6 longitude'].values,height = 10)
plt.ylabel('longitude')
plt.xlabel('latitude')
plt.savefig('GRID',dpi = 500)
plt.show()



X = dataset.iloc[:,1:7].values
Y = dataset.iloc[:,7].values

#plotting the initial data
for i in range(0,6):
    plot(X, i, Y)
