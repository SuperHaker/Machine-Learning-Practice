
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet = pd.read_csv('Mall_Customers.csv')
X = dataSet.iloc[:, [3,4]].values

# elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Elbow graph")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state = 0)
ykmeans = kmeans.fit_predict(X)

plt.scatter(X[ykmeans==0,0], X[ykmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[ykmeans==1,0], X[ykmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[ykmeans==2,0], X[ykmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[ykmeans==3,0], X[ykmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[ykmeans==4,0], X[ykmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()