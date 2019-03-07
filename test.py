# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:01:55 2019

@author: Allaoui
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import cv2 
import os

from skimage import data 

from skimage.exposure import histogram 

from matplotlib import pyplot as plt 

import numpy as np 

chemin ="Data/"
  
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
    for root, _, files in os.walk(repertoir+"Mer/"):
        for filename in files:
            img = cv2.imread(repertoir+"Mer/"+filename)
            histr = cv2.calcHist([img], [0], None, [256], [0, 255]).tolist()
            histr1 = cv2.calcHist([img], [1], None, [256], [0, 255]).tolist()
            histr2 = cv2.calcHist([img], [2], None, [256], [0, 255]).tolist()
            
            vector_image = histr + histr1 + histr2
            
            X_image = np.ravel(vector_image)
            
            matrice .append(X_image)
    for root, _, files in os.walk(repertoir+"Ailleurs/"):
        for filename in files:
            img = cv2.imread(repertoir+"Ailleurs/"+filename)
            histr = cv2.calcHist([img], [0], None, [256], [0, 255]).tolist()
            histr1 = cv2.calcHist([img], [1], None, [256], [0, 255]).tolist()
            histr2 = cv2.calcHist([img], [2], None, [256], [0, 255]).tolist()
            
            vector_image = histr + histr1 + histr2
            
            X_image = np.ravel(vector_image)
            
            matrice .append(X_image)
    
    return matrice






def FileCount(directoryPath1, directoryPath2): 
    numberOfFiles1 = next(os.walk(directoryPath1))[2] 
    numberOfFiles2 = next(os.walk(directoryPath2))[2] 
    return len(numberOfFiles1), len(numberOfFiles2) 

def CreateY(filesList): 
    y = [] 
    for merNumber in range(0, filesList[0]): 
        y.append(1) 
    for merNumber in range(0, filesList[1]): 
        y.append(-1)      
    return y 

  
### Main ### 


filesList = FileCount("Data/Mer", "Data/Ailleurs") 

y = CreateY(filesList) 

X = parcours(chemin)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


classifieur = GaussianNB()
classifieur.fit(X_train, y_train)
y_predits = classifieur.predict(X_test) 

print("Taux de réussite : ", accuracy_score(y_test,y_predits))


print (len(y), len(X))

  

'''img = cv2.imread('grgr.png', 1) 

cv2.imshow('Image', img) 

cv2.waitKey(0) 

cv2.destroyAllWindows()''' 