# ML-Algorithm---(Unsupervised Learning) K-Means-Clustering 
Clustering analysis on the telecom dataset is performed to derive insights and get possible information on factors that may affect the churn decision.

The data is a mixture of both categorical and numerical data.

1. The K-means clustering algorithm computes centroids and repeats until the optimal centroid is found. It is presumptively known how many clusters there are. It is also known as the flat clustering algorithm. 
2. The number of clusters found from data by the method is denoted by the letter ‘K’ in K-means.
3. In this method, data points are assigned to clusters in such a way that the sum of the squared distances between the data points and the centroid is as small as possible. 
4. It is essential to note that reduced diversity within clusters leads to more identical data points within the same cluster. 

** Advantages
> It is simple to grasp and put into practice.
> K-means would be faster than Hierarchical clustering if we had a high number of variables.
> An instance’s cluster can be changed when centroids are re-computated.
> When compared to Hierarchical clustering, K-means produces tighter clusters.

** Disadvantages
> The number of clusters, i.e., the value of k, is difficult to estimate.
> A major effect on output is exerted by initial inputs such as the number of clusters in a network.
> The sequence in which the data is entered has a considerable impact on the final output.
> It’s quite sensitive to rescaling. If we rescale our data using normalisation or standards, the outcome will be drastically different.
> It is not advisable to do clustering tasks if the clusters have a sophisticated geometric shape.


#### Elbow plot is drawn and optimal number for cluster is obtained ( cluster = 4) 
