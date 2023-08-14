# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 00:13:36 2022

@author: mek001
"""
## for data
import os
import numpy as np
import pandas as pd
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
## for machine learning
from sklearn import cluster
import scipy
import utm as utm


def space_eliminator(string):
    '''this function allow me to remove empty spaces and the end of string.
    input is a string, if there is a space at the end of input I am deleting it.
    if the last character is not empty space, I am stopping there'''
    if string[-1]==" ": #if the last character is a space
        return space_eliminator(string[:-1]) #I am removing last character
    else: #the last character is not a empty space, no need to go further
        return string

os.chdir('\\Users\\IHE-Sale\\Desktop\\master_home')
shp_path = "./barcelona_shape_file/shapefile_distrito_barcelona.shp"
Bar_data = gpd.read_file(shp_path)
#reading coordinate
df3=pd.read_excel("coordinates.xls") #reading coordinates file
df3.set_index(df3["CODI"], drop=True,inplace=True) #assigning index
df3=df3[["X","Y"]] #getting coordinates, rest is unnecessary
coord=utm.to_latlon(df3["X"], df3["Y"], 31, 'T') #turning them to lat and lon
df3["Y_c"]=coord[0] #latitude, _c was added
df3["X_c"]=coord[1] #longitude,_c was added because there is another Y element
#reading data
df=pd.read_excel("B1.xlsx") #reading first field campaign
df.set_index(df["Sample Code"], drop=True,inplace=True) #assigning index
df2=pd.read_excel("B2.xlsx") #reading second field campaign
#there are extra II at the end of well names 
#due to these extra characters it does not match with first field campaign
#so I am deleting these characters
df2_names=[name[:-2] for name in df2["ID"]]
df2.set_index(pd.Index(df2_names),inplace=True) #assigning it as an index
#takign only required data
new_df2=df2.iloc[:,2:] #removing unnecessary columns at the beginning
#df2 and df might have random empty space that cause problem at the end of column names
#I am removing that spaces in both database
col_names=[space_eliminator(i) for i in new_df2.columns]
new_df2.set_axis(col_names, axis="columns", inplace=True) #assigning corrected names for columns
X = new_df2.copy() #creating a copy
X=X.iloc[:,:-5] #I manually calculated last 5 col in excel, removing them
columns_to_drop=["n.a." in X[i].values for i in X.columns] #there are empty columns
X.drop(X.columns[columns_to_drop],axis=1,inplace=True) #dropping empty columns
X.dropna(axis=1,inplace=True) #dropping nan columns
##################cleaning data on first field campaign
new_df=df.iloc[:,1:-5] #first col is name, last 5 col calculated by me
X_1=new_df.copy() #creating a copy
columns_to_drop=["n.a." in X_1[i].values for i in X_1.columns] #there are empty columns
X_1.drop(X_1.columns[columns_to_drop],axis=1,inplace=True) #dropping empty columns
X_1.dropna(axis=1,inplace=True) #dropiing nan columns
#some measurements are lower than detection limit. 
#If there are so much such value, I am droping these columns
threshold=10 #threshold number of measurement lower than detection limit
#if repeating value (which is detection limit) count is higher than threshold,
#I am dropping that column
columns_to_drop=[max(X[i].value_counts())>threshold for i in X.columns]
X.drop(X.columns[columns_to_drop],axis=1,inplace=True) #dropping low info columns
#first field camp
#same algorithm to the first fielc campaign
columns_to_drop=[max(X_1[i].value_counts())>threshold for i in X_1.columns]
X_1.drop(X_1.columns[columns_to_drop],axis=1,inplace=True) #dropping low info columns
#After dropping low info columns, measurements lower than detection limit
#should be taken as half of detection limit. to do that, I remove sign and
#divide them to 2. If the value is higher than max value, I only remove sign
#Lastly, I turned to all values to numeric value
for i in X.columns:
    for j in X.index.values:
        if type(X[i][j]) == str:
            if X[i][j][0]=="<":
                #if it is lower thand etection limit, divide it to 2
                X[i][j]=float(X[i][j][1:])/2
            elif X[i][j][0]==">":
                #if it is higher than max limit. just leave it there
                X[i][j]=float(X[i][j][1:])
X=X.iloc[:, :].astype('float') #making all numbers float
##################ELBOW method for distortions for second field campaign
max_k = 10
## iterations
distortions = [] 
for i in range(1, max_k+1):
    if len(X) >= i:
       model = cluster.KMeans(n_clusters=i, 
                              init='k-means++', 
                              max_iter=300, 
                              n_init=10, 
                              random_state=0)
       model.fit(X)
       distortions.append(model.inertia_)
## best k: the lowest derivative
k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i 
     in np.diff(distortions,2)]))
## plotting distortions vs clusters
fig, ax = plt.subplots()
ax.plot(range(1, len(distortions)+1), distortions)
ax.axvline(k, ls='--', color="red", label="k = "+str(k))
ax.set(title='The Elbow Method', xlabel='Number of clusters', 
       ylabel="Distortion")
ax.legend()
ax.grid(True)
plt.show()
#I will go with cluster count of 5
k = 5 
model = cluster.KMeans(n_clusters=k, init='k-means++')

df3_X = X.copy()
df3_X["cluster"] = model.fit_predict(df3_X)
## find real centroids
closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, 
                     df3_X.drop("cluster", axis=1).values)
df3_X["centroids"] = 0
for i in closest:
    df3_X["centroids"].iloc[i] = 1
#I am adding coordinates back to df3_X, I didn't do that at the beginning
#because I didn't want them to cluster according to coordinates,
#I only wanted cluster according to chemistry parameters
df3_X=pd.concat([df3_X,df3],axis=1,join="inner")
###################PLOTTING SECOND FIELD CAMPAIGN
fig, ax = plt.subplots(figsize=(13,13))
ax.grid(False)
Bar_data.plot(color='none', 
              edgecolor='gainsboro',
              zorder=1,
              ax=ax
              )
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.title('Barcelona Second Field Campaign K-Means Cluster Analysis')
sns.scatterplot(data=df3_X["cluster"],
                x=df3_X["X_c"],
                y=df3_X["Y_c"],
                hue=df3_X["cluster"],
                style=df3_X["cluster"],
                palette="bright",
                s=150,
                ax=ax,
                zorder=2)
fig.tight_layout()
plt.savefig('Second_cluster.png', dpi=300) #high resolution saving
#First campaign elbow method
max_k = 10
## iterations
distortions = [] 
for i in range(1, max_k+1):
    if len(X_1) >= i:
       model = cluster.KMeans(n_clusters=i, 
                              init='k-means++', 
                              max_iter=300, #default
                              n_init=10,  #default
                              random_state=0)
       model.fit(X_1)
       distortions.append(model.inertia_)
## best k: the lowest derivative
k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i 
     in np.diff(distortions,2)]))
## plot
fig, ax = plt.subplots()
ax.plot(range(1, len(distortions)+1), distortions)
ax.axvline(k, ls='--', color="red", label="k = "+str(k))
ax.set(title='The Elbow Method', xlabel='Number of clusters', 
       ylabel="Distortion")
ax.legend()
ax.grid(True)
plt.show()
#I will go for k=5

k = 5
model = cluster.KMeans(n_clusters=k, init='k-means++')

df4_X = X_1.copy()
df4_X["cluster"] = model.fit_predict(df4_X)
## find real centroids
closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, 
                     df4_X.drop("cluster", axis=1).values)
df4_X["centroids"] = 0
for i in closest:
    df4_X["centroids"].iloc[i] = 1

#concatenate
#same reason
df4_X=pd.concat([df4_X,df3],axis=1,join="inner")
#plotting
fig, ax = plt.subplots(figsize=(13,13))
ax.grid(False)
Bar_data.plot(color='none', 
              edgecolor='gainsboro',
              zorder=1,
              ax=ax)
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.title('Barcelona Second Field Campaign K-Means Cluster Analysis')
sns.scatterplot(x=df4_X["X_c"],
                y=df4_X["Y_c"],
                data=df4_X["cluster"],
                hue=df4_X["cluster"],
                style=df4_X["cluster"],
                palette="bright",
                s=150,
                ax=ax,
                zorder=2)
fig.tight_layout()
plt.savefig('First_cluster.png', dpi=300) #high resolution saving