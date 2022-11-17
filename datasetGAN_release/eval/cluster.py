from sklearn.cluster import MiniBatchKMeans
import os 

path = "/home/jechterh/teams/group-9/datasetGAN_2000_generations_masks"
total_clusters = 1
# Initialize the K-Means model
kmeans = MiniBatchKMeans(n_clusters = total_clusters)
# Fitting the model to training set
kmeans.fit(X_train)