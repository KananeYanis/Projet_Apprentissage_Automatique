import cv2
import numpy as np
from matplotlib import pyplot as plt

'''img = cv2.imread('home.jpg',0)
plt.hist(img.ravel(),256,[0,256])
plt.show()'''

  

img = cv2.imread('xcpoqsmlze.jpeg') 
color = ('b','g','r')
histr = cv2.calcHist([img], [0], None, [256], [0, 255]).tolist()
plt.plot(histr,'b') 
plt.xlim([0, 255])
plt.show()
 
histr1 = cv2.calcHist([img], [1], None, [256], [0, 255]).tolist() 
plt.plot(histr1,'g') 
plt.xlim([0, 255])
plt.show()
 
histr2 = cv2.calcHist([img], [2], None, [256], [0, 255]).tolist()
plt.plot(histr2,'r') 
plt.xlim([0, 255]) 
plt.show()

vector_image = histr + histr1 + histr2
x_image = np.ravel(vector_image)
print(x_image) #affiche vecteur pour une image
#print(x_image.shape) #affiche la forme