import cv2
import numpy as np
import numpy.random as rng

'''
Bunch of data augmentation functions for images
'''

def rotateScale(img,maxAngle,maxScaleStd):
	rows,cols,channels = img.shape
	maxX, maxY = int(rows*0.05), int(cols*0.05)
	M = cv2.getRotationMatrix2D((cols/2,rows/2),rng.normal(0,maxAngle),rng.normal(1,maxScaleStd))
	tX, tY = rng.randint(-maxX,maxX,2)
	M[0,2], M[1,2] = tX, tY
	return cv2.warpAffine(img,M,(cols,rows),borderMode=0,flags=cv2.INTER_CUBIC)

def gamma(img,gammaStd=0.1):
	'''
	random contrasting 
	'''
	gamma = rng.normal(1,gammaStd)
	return img**(1.0/gamma)

if __name__ == "__main__":
	from matplotlib.pyplot import imshow,ion,show,pause,draw
	import time, glob
	print("Aug demo")
	imgs = glob.glob("/home/msmith/kaggle/whale/imgs/*/head_*")
	for j in xrange(100):
		try:
			img = cv2.imread(imgs[j])
		except:
			img = cv2.imread("../tree.jpg")
		for i in xrange(10):
			imgC = img.copy()
			imgR = rotateScale(imgC,maxAngle=6,maxScaleStd=0.03)
			imgR = gamma(imgR)
			imshow(imgR.astype(np.uint8))
			show(block=False)
			draw()
			pause(0.05)


