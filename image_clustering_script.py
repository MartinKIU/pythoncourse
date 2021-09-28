#!/usr/bin/env python
# coding: utf-8

# #This script/code will generate a HOG image of all the images in a folder. A matrix/dataframe consisting of the hog-features in columns and images in rows will then be generated. 
#Each image will have 3780 features. A Princinpal component analysis based on the features values for the images will then be carried out and the top 15 PC:s resperesenting the highest variation will be used for clustering the images with a K-means clustering method. 
#The knee locator function on SSE vs cluster values will be used for determining how many cluster there are in the data. The images will then be annotated according to their clusters and presented in a PCA plot. 
#The images will also be resorted according to their cluster annotation into new folders.

# In[1]:


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
get_ipython().run_line_magic('matplotlib', 'inline')


# #import each picture as a separate object

# In[64]:


img1 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.07 #4.jpg')
img2 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.07 #5.jpg')
img3 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.05 #2.jpg')
img4 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.09 #2.jpg')
img5 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.07 #7.jpg')
img6 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.07 #6.jpg')
img7 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.09 #3.jpg')
img8 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.05 #3.jpg')
img9 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.05 #7.jpg')
img10 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.07 #2.jpg')
img11 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.09 #6.jpg')
img12 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.07 #3.jpg')
img13 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.05 #6.jpg')
img14 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.05 #4.jpg')
img15 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.09 #4.jpg')
img16 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.09 #5.jpg')
img17 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.05 #5.jpg')
img18 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.08.jpg')
img19 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.09.jpg')
img20 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.06 #9.jpg')
img21 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.06 #8.jpg')
img22 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.06 #6.jpg')
img23 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.08 #3.jpg')
img24 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.08 #2.jpg')
img25 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.06 #7.jpg')
img26 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.06 #5.jpg')
img27 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.06 #4.jpg')
img28 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.08 #5.jpg')
img29 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.07.jpg')
img30 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.06.jpg')
img31 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.08 #4.jpg')
img32 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.08 #6.jpg')
img33 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.06 #3.jpg')
img34 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.10.jpg')
img35 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.05.jpg')
img36 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.06 #2.jpg')
img37 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.08 #7.jpg')
img38 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.10 #2.jpg')
img39 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.10 #3.jpg')
img40 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.07 #8.jpg')
img41 = imread('/Users/maruhr/OneDrive - Karolinska Institutet/python/project1/bilder/Bild 2021-09-16 kl. 15.07 #9.jpg')


# #resize every image

# In[65]:


resized_img1 = resize(img1, (128,64))
resized_img2 = resize(img2, (128,64))
resized_img3 = resize(img3, (128,64))
resized_img4 = resize(img4, (128,64))
resized_img5 = resize(img5, (128,64))
resized_img6 = resize(img6, (128,64))
resized_img7 = resize(img7, (128,64))
resized_img8 = resize(img8, (128,64))
resized_img9 = resize(img9, (128,64))
resized_img10 = resize(img10, (128,64))
resized_img11 = resize(img11, (128,64))
resized_img12 = resize(img12, (128,64))
resized_img13 = resize(img13, (128,64))
resized_img14 = resize(img14, (128,64))
resized_img15 = resize(img15, (128,64))
resized_img16 = resize(img16, (128,64))
resized_img17 = resize(img17, (128,64))
resized_img18 = resize(img18, (128,64))
resized_img19 = resize(img19, (128,64))
resized_img20 = resize(img20, (128,64))
resized_img21 = resize(img21, (128,64))
resized_img22 = resize(img22, (128,64))
resized_img23 = resize(img23, (128,64))
resized_img24 = resize(img24, (128,64))
resized_img25 = resize(img25, (128,64))
resized_img26 = resize(img26, (128,64))
resized_img27 = resize(img27, (128,64))
resized_img28 = resize(img28, (128,64))
resized_img29 = resize(img29, (128,64))
resized_img30 = resize(img30, (128,64))
resized_img31 = resize(img31, (128,64))
resized_img32 = resize(img32, (128,64))
resized_img33 = resize(img33, (128,64))
resized_img34 = resize(img34, (128,64))
resized_img35 = resize(img35, (128,64))
resized_img36 = resize(img36, (128,64))
resized_img37 = resize(img37, (128,64))
resized_img38 = resize(img38, (128,64))
resized_img39 = resize(img39, (128,64))
resized_img40 = resize(img40, (128,64))
resized_img41 = resize(img41, (128,64))


# #creating the HOG features

# In[69]:


fd1, hog_image1 = hog(resized_img1, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd2, hog_image2 = hog(resized_img2, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd3, hog_image3 = hog(resized_img3, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd4, hog_image4 = hog(resized_img4, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd5, hog_image5 = hog(resized_img5, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd6, hog_image6 = hog(resized_img6, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd7, hog_image7 = hog(resized_img7, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd8, hog_image8 = hog(resized_img8, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd9, hog_image9 = hog(resized_img9, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd10, hog_image10 = hog(resized_img10, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd11, hog_image11 = hog(resized_img11, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd12, hog_image12 = hog(resized_img12, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd13, hog_image13 = hog(resized_img13, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd14, hog_image14 = hog(resized_img14, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd15, hog_image15 = hog(resized_img15, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd16, hog_image16 = hog(resized_img16, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd17, hog_image17 = hog(resized_img17, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd18, hog_image18 = hog(resized_img18, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd19, hog_image19 = hog(resized_img19, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd20, hog_image20 = hog(resized_img20, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd21, hog_image21 = hog(resized_img21, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd22, hog_image22 = hog(resized_img22, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd23, hog_image23 = hog(resized_img23, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd24, hog_image24 = hog(resized_img24, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd25, hog_image25 = hog(resized_img25, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd26, hog_image26 = hog(resized_img26, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd27, hog_image27 = hog(resized_img27, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd28, hog_image28 = hog(resized_img28, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd29, hog_image29 = hog(resized_img29, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd30, hog_image30 = hog(resized_img30, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd31, hog_image31 = hog(resized_img31, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd32, hog_image32 = hog(resized_img32, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd33, hog_image33 = hog(resized_img33, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd34, hog_image34 = hog(resized_img34, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd35, hog_image35 = hog(resized_img35, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd36, hog_image36 = hog(resized_img36, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd37, hog_image37 = hog(resized_img37, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd38, hog_image38 = hog(resized_img38, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd39, hog_image39 = hog(resized_img39, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd40, hog_image40 = hog(resized_img40, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
fd41, hog_image41 = hog(resized_img41, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)


# #look at a hog image

# In[68]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True) 

ax1.imshow(resized_img5, cmap=plt.cm.gray) 
ax1.set_title('Input image') 

# Rescale histogram for better display 
hog_image_rescaled = exposure.rescale_intensity(hog_image5, in_range=(0, 10)) 

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
ax2.set_title('Histogram of Oriented Gradients')

plt.show()


# # Make a data frame with features in columns and images in rows

# In[70]:


data = {"fd1":fd1,"fd2":fd2,"fd3":fd3,"fd4":fd4,"fd5":fd5,"fd6":fd6,"fd7":fd7,"fd8":fd8,"fd9":fd9,"fd10":fd10,"fd11":fd11,"fd12":fd12,"fd13":fd13,"fd14":fd14,"fd15":fd15,"fd16":fd16,"fd17":fd17,"fd18":fd18,"fd19":fd19,"fd20":fd20,"fd21":fd21,"fd22":fd22,"fd23":fd23,"fd24":fd24,"fd25":fd25,"fd26":fd26,"fd27":fd27,"fd28":fd28,"fd29":fd29,"fd30":fd30,"fd31":fd31,"fd32":fd32,"fd33":fd33,"fd34":fd34,"fd35":fd35,"fd36":fd36,"fd37":fd37,"fd38":fd38,"fd39":fd39,"fd40":fd40,"fd41":fd41}
df = pd.DataFrame(data)
df.head


# #transpose dataframe so that features are in columns and images are in rows

# In[71]:


df_transpose = df.T 


# #save the data matrix as a text file

# In[72]:


np.savetxt("fd_all.gz", df_transpose, fmt='%10.5f', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)


# In[73]:


df_transpose.head()


# In[8]:


df_transpose.shape


# # Normalize data, meaning that the feature value(columns) will be substracted with mean value of that feature and then divided by the standard diviation of that feature. After this transformation the mean value should be zero and the standard deviation should be one for all columns. 

# In[77]:


scaler = StandardScaler()

scaled_df=df_transpose.copy()
scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)
scaled_df.head()


# In[79]:


#look at the shape of x
scaled_df.shape


# # Make sure that mean is zero and std equals 1

# In[76]:


np.mean(scaled_df),np.std(scaled_df)


# # Run the PCA, calculating 41 principal componenets 

# #new try with scaled_df

# In[80]:


pca_images = PCA(n_components=41)
principalComponents_images = pca_images.fit_transform(scaled_df)


# In[12]:


pca_images = PCA(n_components=41)
principalComponents_images = pca_images.fit_transform(x)


# # create a data fram with PC values

# In[81]:


principal_images_Df = pd.DataFrame(data = principalComponents_images
             , columns = ['PC1', 'PC2','PC3', 'PC4','PC5', 'PC6','PC7', 'PC8','PC9', 'PC10','PC11', 'PC12','PC13', 'PC14','PC15', 'PC16','PC17', 'PC18','PC19', 'PC20','PC21', 'PC22','PC23', 'PC24','PC25', 'PC26','PC27', 'PC28','PC29','PC30', 'PC31','PC32', 'PC33','PC34', 'PC35','PC36', 'PC37','PC38','PC39', 'PC40','PC41'])


# In[82]:


principal_images_Df.tail()


# # look at the variation in each PC1

# In[83]:


PC_values = np.arange(pca_images.n_components_) + 1
plt.plot(PC_values, pca_images.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()


# In[84]:


type(PC_values)
PC_values
pca_images.explained_variance_ratio_


# # Plot PC1 against PC2

# In[99]:


plt.scatter(principal_images_Df["PC1"],principal_images_Df["PC2"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA")


# # Cluster with K-means clustering (using the 15 PC:s with the highest variation)

# In[85]:


type(principal_images_Df)
cols = ['PC1', 'PC2','PC3', 'PC4','PC5', 'PC6','PC7', 'PC8','PC9', 'PC10','PC11', 'PC12','PC13', 'PC14','PC15']
principal_images_Df_15 = principal_images_Df[cols]
principal_images_Df_15


# In[86]:


kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse1 = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(principal_images_Df_15)
    sse1.append(kmeans.inertia_)


# In[87]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse1)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# # Find number of cluster with elbow function

# In[88]:


kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)

kl.elbow


# # Cluster cells with K=4

# In[89]:


kmeans = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)


# In[90]:


kmeans.fit(principal_images_Df_15)


# # Cluster annotation 

# In[92]:


kmeans.labels_[0:41]


# # Add cluster annotation  to original dataframe principal_images_Df_15

# In[106]:


principal_images_cluster = principal_images_Df_15
principal_images_cluster["cluster"]  = ["0", "0", "2", "3", "2", "0", "3", "3", "2", "0", "3", "0", "2", "1", "3", "3", "3", "3", "2", "0", "2", "1", "3", "3", "3", "3", "2", "2", "0", "3", "1", "0", "2", "1", "1", "1", "0", "1", "1", "2", "1"]


# In[107]:


principal_images_cluster


# # plot cluster in PCA plot

# In[110]:


principal_images_cluster['cluster']


# In[111]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 2', fontsize = 15)
ax.set_title('Plot of 1st Two Principal Components vs. cluster', fontsize = 20)
clusters = ['0','1','2','3']
colors = ['navy', 'turquoise', 'darkorange','red']
for cluster, color in zip(clusters,colors):
    indicesToKeep = principal_images_cluster['cluster'] == cluster
    ax.scatter(principal_images_cluster.loc[indicesToKeep, 'PC1']
               , principal_images_cluster.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(clusters)
ax.grid()


# # Stop here

