import pandas as pd
import numpy as np

#Reading files as panda dataframes
whisky = pd.read_csv("whiskies.txt", sep = ",")
whisky['Region'] = pd.read_csv('regions.txt') #Reading the row of a new dataframe as a column in the old dataframe. Need to check what happens if there are multiple columns in the new df or the order needs to be verified

whisky.head()

#Flavors df
flavors = whisky.iloc[:,2:14] # Gets only the columns with flavor data i.e. columns are flavors and rows are whiskies

corr_flavors = pd.DataFrame.corr(flavors)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.pcolor(corr_flavors)
plt.colorbar()

corr_whisky = pd.DataFrame.corr(flavors.transpose())

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.pcolor(corr_whisky)
plt.colorbar()

#Clustering
from sklearn.cluster.bicluster import SpectralCoclustering
model = SpectralCoclustering(n_clusters = 6, random_state = 0)
model.fit(corr_whisky)
model.rows_ #Output is number of row clusters times number of rows with true indicating it is a part of the cluster and false if it is not a part of the cluster

whisky["Group"] = pd.Series(model.row_labels_, index = whisky.index)	#Adds cluster numbers to the dataset
whisky = whisky.iloc[np.argsort(model.row_labels_)]	#Sorts by cluster number
whisky = whisky.reset_index(drop = True)	#Resets the index to begin at 0

correlations = np.array(pd.DataFrame.corr(whisky.iloc[:,2:14].transpose()))

#Plot differences
plt.figure(figsize=(14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title("Original")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")