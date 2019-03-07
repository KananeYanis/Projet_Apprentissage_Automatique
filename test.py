# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:01:55 2019

@author: Allaoui
"""

import cv2 
import os

from skimage import data 

from skimage.exposure import histogram 

from matplotlib import pyplot as plt 

import numpy as np 

chemin ="test/"
  
'''
img = cv2.imread('Data/Mer/eeeee.jpeg') 

color = ('b', 'g','r') 
 
histr = cv2.calcHist([img], [0], None, [256], [0, 255]).tolist()
#plt.plot(histr, color = 'b') 
#plt.xlim([0, 255]) 
#plt.show() 
 
histr1 = cv2.calcHist([img], [1], None, [256], [0, 255]).tolist()
#plt.plot(histr1, color = 'g') 
#plt.xlim([0, 255]) 
#plt.show() 
 
histr2 = cv2.calcHist([img], [2], None, [256], [0, 255]).tolist()
#plt.plot(histr2, color = 'r') 
#plt.xlim([0, 255]) 
#plt.show() 


vector_image = histr + histr1 + histr2
print(type(vector_image), type(histr))
X_image = np.ravel(vector_image)
print(type(X_image))
print(np.array(vector_image).shape)
'''


def parcours (repertoir) :
    global histr, histr1, histr2, vector_image, X_image
    matrice =[]
    for root, _, files in os.walk(repertoir):
        for filename in files:
            img = cv2.imread(repertoir+filename)
            histr = cv2.calcHist([img], [0], None, [256], [0, 255]).tolist()
            histr1 = cv2.calcHist([img], [1], None, [256], [0, 255]).tolist()
            histr2 = cv2.calcHist([img], [2], None, [256], [0, 255]).tolist()
            
            vector_image = histr + histr1 + histr2
            
            X_image = np.ravel(vector_image)
            
            matrice .append(X_image)
    
    return matrice


print(parcours(chemin))

  

'''img = cv2.imread('grgr.png', 1) 

cv2.imshow('Image', img) 

cv2.waitKey(0) 

cv2.destroyAllWindows()''' 