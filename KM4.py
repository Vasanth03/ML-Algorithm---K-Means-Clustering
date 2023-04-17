#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:46:23 2022

@author: vasanthdhanagopal
"""
############################# K-Means Clustering ##############################
# The K-means clustering algorithm computes centroids and repeats until the optimal centroid 
# is found. It is presumptively known how many clusters there are. It is also known as the 
# flat clustering algorithm. The number of clusters found from data by the method is denoted by
# the letter ‘K’ in K-means.

import pandas as pd
# used for data manipualation
import matplotlib.pyplot as plt
# used for graph plotting

tele = pd.read_excel("copy file path")
# read the excel file

tele.describe() #tells the mean, max and min values
tele.info()     # informs the dataypes

tele.isnull().sum() # checks the null value and adds it

tele3 = tele.drop(['Customer ID'],axis=1)
tele2 = tele3.drop(['Count'],axis=1)
tele1 = tele2.drop(['Quarter'],axis=1)

# we can just use repalce fucntion for binary problems
tele1['Referred a Friend'] = tele1['Referred a Friend'].replace({'Yes':1,'No':0})
tele1['Paperless Billing'] = tele1['Paperless Billing'].replace({'Yes':1,'No':0})
tele1['Phone Service']     = tele1['Phone Service'].replace({'Yes':1,'No':0})
tele1['Multiple Lines']    = tele1['Multiple Lines'].replace({'Yes':1,'No':0})
tele1['Internet Service']  = tele1['Internet Service'].replace({'Yes':1,'No':0})
tele1['Online Security']   = tele1['Online Security'].replace({'Yes':1,'No':0})
tele1['Online Backup']     = tele1['Online Backup'].replace({'Yes':1,'No':0})
tele1['Device Protection Plan'] = tele1['Device Protection Plan'].replace({'Yes':1,'No':0})
tele1['Premium Tech Support'] = tele1['Premium Tech Support'].replace({'Yes':1,'No':0})
tele1['Streaming TV'] = tele1['Streaming TV'].replace({'Yes':1,'No':0})
tele1['Streaming Movies'] = tele1['Streaming Movies'].replace({'Yes':1,'No':0})
tele1['Streaming Music'] = tele1['Streaming Music'].replace({'Yes':1,'No':0})
tele1['Unlimited Data'] = tele1['Unlimited Data'].replace({'Yes':1,'No':0})

# Dummy Variable creation
dummy_offer = pd.get_dummies(tele1['Offer'])
tele1 = tele1.join(dummy_offer) # created dummies
tele1 = tele1.rename(columns={'None':'Offer None'})
tele1= tele1.drop(['Offer'], axis=1)


dummy_type = pd.get_dummies(tele1['Internet Type'])
tele1  = tele1.join(dummy_type)  # created dummies
tele1  = tele1.rename(columns={'None':'Internet Type None'})
tele1  = tele1.drop(['Internet Type'], axis=1) 

dummy_Contract = pd.get_dummies(tele1['Contract'])
tele1 = tele1.join(dummy_Contract)  # created dummies
tele1 = tele1.drop(['Contract'], axis=1) 

dummy_Method = pd.get_dummies(tele1['Payment Method'])
tele1 = tele1.join(dummy_Method)  # created dummies
tele1 = tele1.drop(['Payment Method'], axis=1) 
 

#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
tele1_norm  = scaler.fit_transform(tele1.iloc[:,1:])
tele1_norm  = pd.DataFrame(tele1_norm)


from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# scree plot or elbow curve 
TWSS = []
k = list(range(1, 15))

for i in k:
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(tele1_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'b*--');plt.title("Elbow Plot");plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
plt.show()

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(tele1_norm)

model.labels_ # getting the labels of clusters assigned to each row 
tl = pd.Series(model.labels_)  # converting numpy array into pandas series object 
tele1['clust'] = tl # creating a  new column and assigning it to new column 

tele1.head()
tele1_norm.head()

# Moving lasst column to first space
temp_cols=tele1.columns.tolist()
new_cols=temp_cols[-1:] + temp_cols[:-1]
tele1=tele1[new_cols]

tele1.iloc[:, 2:].groupby(tele1.clust).mean()

# Distribution of Clusters
import seaborn as sns
pl = sns.countplot(x=tele1["clust"], palette= "Set2")
pl.set_title("Distribution Of The Clusters")
plt.show()


# Cluster Profiles
pl = sns.scatterplot(x=tele1["Total Charges"], y=tele1["Total Revenue"],hue=tele1["clust"], palette= "pastel")
pl.set_title("Cluster's Profile Based On Revenue and Charges")
plt.legend()
plt.show()

# creating a csv file 
tele.to_excel("tele.xlsx", encoding = "utf-8")

import os
os.getcwd()


