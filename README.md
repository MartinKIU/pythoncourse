# pythoncourse
Python project for GU:s Python course
The aim with this python code is that it will do the follwing:

#This script/code will generate a HOG image of all the images in a folder. The images most be either in ".jpg" or ".png" format. A matrix/dataframe consisting of the hog-features in columns and images in rows will then be generated.  Each image will have 3780 features. A Princinpal component analysis based on the features values for the images will then be carried out and the top PC:s resperesenting the highest variation will be used for clustering the images with a K-means clustering method. The knee locator function on SSE vs cluster values will be used for determining how many cluster there are in the data. The images will then be annotated according to their clusters and presented in a PCA plot. The images will also be copied according to their cluster annotation into new folders. This script onlu work on a computer with the OS-operative system.
 #To run the script write: python image_clustering_script_v1.py "foldername_of_the_folder_with_the_images"

Dependencies needed in order to run the script:

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
