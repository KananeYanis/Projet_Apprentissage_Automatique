import cv2
import numpy as np
from scipy import misc
from matplotlib import pyplot as plt

'''img = cv2.imread('home.jpg',0)
plt.hist(img.ravel(),256,[0,256])
plt.show()'''

  

'''img = cv2.imread('')'''
M = misc.imread('aaaaa.jpeg')
color = ('b','g','r')
histr = cv2.calcHist([M], [0], None, [256], [0, 255]).tolist()
plt.plot(histr,'b') 
plt.xlim([0, 255])
plt.show()
 
histr1 = cv2.calcHist([M], [1], None, [256], [0, 255]).tolist() 
plt.plot(histr1,'g') 
plt.xlim([0, 255])
plt.show()
 
histr2 = cv2.calcHist([M], [2], None, [256], [0, 255]).tolist()
plt.plot(histr2,'r') 
plt.xlim([0, 255]) 
plt.show()

vector_image = histr + histr1 + histr2
x_image = np.ravel(vector_image)
print(histr) #affiche vecteur pour une imageprint(x_image.shape) #affiche la forme

print(M.shape)
print("--------------------")
print(M[0])
print("--------------------")
print(M[1])
print("--------------------")
print(M[2])
print("--------------------")
    
'''for pixel in range(len(histr1)):
    ratio = pixel/taille_image
    print(ratio)
    
for pixel in range(len(histr2)):
    ratio = pixel/taille_image
    print(ratio)'''
    
   
  
