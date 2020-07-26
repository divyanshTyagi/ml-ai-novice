import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
mpl.rcParams['figure.dpi'] = 500
plt.rcParams['figure.figsize'] = (12, 6)

dataset = pd.read_csv('country.csv')
info = pd.read_csv('countryDictionary.csv')
x = dataset.iloc[:,1:10].values
countries = dataset.iloc[:,[0]].values #list of all countries


#Standardizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#Using Elbow Method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
plt.savefig('Elbow Method',dpi = 500)

# Creating dendograms using wards method 
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Countries')
plt.ylabel('Euclidean distances')
plt.savefig('Dendrogram',dpi = 500)
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
kmeans.fit(x)
y = kmeans.predict(x)
print(y)


temp = info.iloc[1:10,[0]].values
features = []
for val in temp:
    for nval in val:
        features.append(nval)
print(features)

c1 = []
c2 = []
c3 = []
for i in range(0,len(dataset)):
    if(y[i] == 0):
        c1.append(str(countries[i]))
    elif(y[i] == 1 ):
        c2.append(str(countries[i]))
    else:
        c3.append(str(countries[i]))
print(len(c1))
print(len(c2))
print(len(c3))

def plotAll(y,features,x):
    for i in range(0,9):
        for j in range(i+1,9):
            plt.scatter(x[y == 0, i], x[y == 0, j], s = 2.5, c = 'red', label = 'Cluster 1',alpha=.8)
            plt.scatter(x[y == 1, i], x[y == 1, j], s = 2.5, c = 'blue', label = 'Cluster 2',alpha = .8)
            plt.scatter(x[y == 2, i], x[y == 2, j], s = 2.5, c = 'green', label = 'Cluster 3',alpha = .8)
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 2, c = 'yellow', label = 'Centroids')
            xtit = str(features[i])
            ytit = str(features[j])
            plt.title(xtit + ' vs ' + ytit)
            plt.xlabel(xtit)
            plt.ylabel(ytit)
            plt.legend()
            plt.savefig(xtit + ' vs ' + ytit,dpi =  500)
            plt.show()
plotAll(y,features,x)


#saving files
df = pd.DataFrame(c2, columns = ["Priority 1"])
df.to_csv("Priority 1.csv", sep = ',', index = False)

df = pd.DataFrame(c3, columns = ["Priority 2"])
df.to_csv("Priority 2.csv", sep = ',', index = False)

df = pd.DataFrame(c1, columns = ["Priority 3"])
df.to_csv("Priority 3.csv", sep = ',', index = False)



