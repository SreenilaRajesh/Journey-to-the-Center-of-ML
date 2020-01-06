# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.
"""

#Using K-means to compress an image by reducing the number of colors it contains to 16

from PIL import Image
import random
import math
im = Image.open('bird_small.tiff', 'r')

#contains the list of pixel values of an image
pix_val = list(im.getdata())
ori_val=pix_val

#function for finding euclidean distance between two pixel values
def eucDis(a,b):
    dis=0
    for i in range(3):
        dis=dis+(a[i]-b[i])*(a[i]-b[i])
    return math.sqrt(dis)

#clustering the pixel values based on centroids
def findCluster(pix_val,centroids):
    clusters=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in pix_val:
        dis_from_centroids=[]
        for j in range(len(centroids)):
            d=eucDis(i,centroids[j])
            dis_from_centroids.append(d)
        clusters[dis_from_centroids.index(min(dis_from_centroids))].append(i)
    return clusters

#finds centroids(mean of a cluster)
def findCentroids(clusters):
    centroids=[]
    for i in range(16):
        r=0
        g=0
        b=0
        #length of a cluster
        length=0
        for j in clusters[i]:
            r=r+j[0]
            g=g+j[1]
            b=b+j[2]
            length=length+1
        mytup=(r//length,g//length,b//length)
        centroids.append(mytup)
    return centroids
 
#choosing 16 random centroids         
a=random.sample(range(16384), 16)
centroids=[]
for i in a:
    centroids.append(pix_val[i])

#here the centroids is found only for two iterations
clusters=findCluster(pix_val,centroids)
new_centroids=findCentroids(clusters)

new_clusters=findCluster(pix_val,new_centroids)
   
#convergence function can be set similar to this 

#while not(set(centroids)==set(new_centroids)):
#    centroids=new_centroids
#    new_centroids=findCentroids(new_clusters)
#    clusters=new_clusters
#    new_clusters=findCluster(pix_val,new_centroids)


#here pixel values are substituted with their closest centroids of the small bird image
new=[]
for i in ori_val:
    for j in range(16):
        if(i in new_clusters[j]):
            new.append(new_centroids[j])
            break
newimg = Image.new('RGB',im.size) 
# insert saved data into the image
newimg.putdata(new)
newimg.show()
    
            

