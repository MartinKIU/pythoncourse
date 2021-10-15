#!/usr/bin/env python
# coding: utf-8

# #This script cluster images in a folder and creates a folder for each cluster in the folder were the images are. 
# #Each image is then copied into its anotated cluster
# #write: python "name of script" folder name

# In[ ]:


#libraries to import
from skimage.io import imread, imshow
from skimage.transform import resize 
from skimage.feature import hog
from skimage import exposure 
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import sys
import shutil


# # Make a function for importing the list of images in the folder selected

# In[ ]:


'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


# # Make a function were only images of .jpg or .png are kept

# In[ ]:


pwd = os.getcwd()
folder = sys.argv[1]
dirName = f"{pwd}/{folder}"
listofimages = getListOfFiles(dirName)

#load each image to a specific the variable
for i in listofimages:
    if i[-4:] == ".jpg":
        print("jpg")
    elif i[-4:] == ".png":
        print(".png")
    else:
        listofimages.remove(i)


# # Resize the images so they are ready for HOG feature calculation 

# In[ ]:


n = len(listofimages)
numbers = range(n)
mylist_resized = []
for i in numbers:
    mylist_resized.append(resize(imread(listofimages[i]), (128,64)))
print(mylist_resized)


# # Calculate the Hog features for each images

# In[ ]:


n = len(listofimages)
numbers = range(n)
fdlist = []
for i in numbers:
    fd = f"fd{i}"
    hog_image = f"hog_image{i}"
    fd, hog_image = hog(mylist_resized[i], orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
    fdlist.append(fd)


# # make a data frame of the list with hog-features in columns and images in rows, scale the dataframe, so that the mean value for each hog feature is zero and the standardn deviation is 1

# In[ ]:


df = pd.DataFrame(fdlist)
scaler = StandardScaler()
scaled_df=df.copy()
scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)


# # Make a Principal component analysis with the scaled data frame as input 

# In[ ]:


n = len(listofimages)
pca_images = PCA(n_components=n)
principalComponents_images = pca_images.fit_transform(scaled_df)


# In[ ]:


principal_images_df = pd.DataFrame(data = principalComponents_images)


# # Calculate the variane across each PC and the select the top PC:s with the most variation with the KLneeLocator function (+10 PCs) for input for the k-means clustering method.

# In[ ]:


PC_values = np.arange(pca_images.n_components_) + 1
list_pc_var = list(pca_images.explained_variance_ratio_)


# In[ ]:


kl = KneeLocator(
    range(0, 41), list_pc_var, curve="convex", direction="decreasing"
)

kl.elbow


# In[ ]:


numbers = range(kl.elbow+10)
numbers
principal_images_df_simpler = principal_images_df[numbers]


# # Find how many cluster it is in the data by scanning number of selected nodes from 1 to 11 in the K-means clustering method. 

# In[ ]:


kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(principal_images_df_simpler)
    sse.append(kmeans.inertia_)


# # Find the best number of nodes(cluser) for the data with the KneeLocator function, by looking at the sum of standard deviation for the different number of starting nodes

# In[ ]:


kl_sse = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)

kl_sse.elbow


# # Cluster the cells with the the prefered number of starting nodes according to the knee.locator function 

# In[ ]:


kmeans = KMeans(
    init="random",
    n_clusters=kl_sse.elbow,
    n_init=10,
    max_iter=300,
    random_state=42
)


# # Annotate the images to different clusters

# In[ ]:


kmeans.fit(principal_images_df_simpler)
clusterlist = list(kmeans.labels_[0:41])


# # Create folder for each cluster (in the selected folder) and copy the images to each folder 

# In[ ]:


principal_images_cluster = principal_images_df_simpler
principal_images_cluster["cluster"]  = clusterlist


# In[ ]:


clusterfolders = principal_images_cluster["cluster"].unique()
for i in clusterfolders:
    foldername = f"cluster_{i}"
    path = os.path.join(dirName, foldername)
    os.mkdir(path)


# In[ ]:


n = len(listofimages)
numbers = range(n)
nc = len(clusterfolders)
ncnumbers = range(nc)
for i in numbers:
    clusterslist = principal_images_cluster["cluster"]
    original = listofimages[i]
    newfilename = f"image{i}"
    for k in ncnumbers:
        foldername = f"cluster_{k}"
        path1 = os.path.join(dirName, foldername)
        path2 = os.path.join(path1, newfilename)
        if clusterslist[i] == [k]:
            target = path2
            shutil.copyfile(original, target)


# In[ ]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 2', fontsize = 15)
ax.set_title('Plot of 1st Two Principal Components vs. cluster', fontsize = 20)
clusters = principal_images_cluster["cluster"].unique()
colors11 = ['navy', 'turquoise', 'darkorange','red','purple','blue','pink','brown','black','green','orange']
colors = colors11[0:kl_sse.elbow]
for cluster, color in zip(clusters,colors):
    indicesToKeep = principal_images_cluster['cluster'] == cluster
    ax.scatter(principal_images_cluster.loc[indicesToKeep, 0]
               , principal_images_cluster.loc[indicesToKeep, 1]
               , c = color
               , s = 50)
ax.legend(clusters)
ax.grid()


# In[ ]:


location = dirName
fig.savefig(f"{location}/PCA_plot.png")

