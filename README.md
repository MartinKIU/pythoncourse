# pythoncourse
Python project for GU:s Python course
The aim with this python code is that it will do the follwing:

#This script/code will generate a HOG image of all the images in a folder and then create a list of "hog"-features generated frome ach image. A matrix/dataframe consisting of the features in columns and images in rows will then be generated.  Each image will have 3780 features. A Princinpal component analysis based on the features values for the images will then be carried out and the first top 15 PC:s resperesenting the highest variation will be used for clustering the images with a K-means clustering method. The knee locator function on SSE vs cluster plot will be used for determining how many cluster there are in the data. The images will then be annotated according to their clusters and presented in a PCA plot. The images will also be resorted according to their cluster annotation into new folders. 
