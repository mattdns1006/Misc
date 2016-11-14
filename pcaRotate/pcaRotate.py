import cv2
import numpy as np 
import matplotlib.pyplot as plt
import glob
from pylab import rcParams
from scipy.ndimage import measurements
import matplotlib.cm as cm
from fitEllipse import fitEllipse

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def getRedYellow(r,(thresh)):
    notRedOrYellow = (r[:,:,0] < thresh[0]) & (r[:,:,1] < thresh[1]) | (r[:,:,2] > thresh[2])
    r[notRedOrYellow] = 0
    return r

def getRed(r,(thresh)):
    red, green, blue = thresh
    notRed = (r[:,:,0] < red) | (r[:,:,1] > green) | (r[:,:,2] > blue)
    r[notRed] = 0
    return r

def getImgMoments(im,channel):
    img = im.copy()
    h,w,c = img.shape
    c = img[:,:,channel]
    x,y = np.arange(w), np.arange(h)
    X,Y = np.meshgrid(x,y)

    m00 = (c).sum()
    m10 = (c*X).sum()
    m01 = (c*Y).sum()
    xBar = (m10/m00).astype(np.uint32)
    yBar = (m01/m00).astype(np.uint32)
    centroid = np.array([xBar,yBar])

    m20 = (c*(X**2)).sum()
    m02 = (c*(Y**2)).sum()
    m11 = (c*Y*X).sum()

    mu20 = m20/m00 - xBar**2
    mu02 = m02/m00 - yBar**2
    mu11 = m11/m00 - xBar*yBar
    cov = np.array([[mu20,mu11],[mu11,mu02]])

    return centroid, cov

def getEigenVectors(centroid1,centroid2,cov):
    
    eigenVals, eigenVectors = np.linalg.eig(cov)
    idx = abs(eigenVals).argsort()[::-1] #Order eigen-system
    eigenVals = eigenVals[idx]
    eigenVectors = eigenVectors[:,idx]

    e1 = eigenVectors[:,0]
    e2 = eigenVectors[:,1]
    
    redGreenDirection = centroid1.astype(np.float32)-centroid2.astype(np.float32)
    angleBetween = angle_between(redGreenDirection,e1)
    if angleBetween < 90: #If (centroid of green - centroid of red) (dotProd) first eigenvector then flip direction 
        e1*=-1
    if np.linalg.det(eigenVectors)<0: #Make sure determinant==1 so not mirror imaging
        e2*=-1
        
    evs = np.array([e1,e2])
    return evs

def rotate(orig,mask,evs,centroid,scale= 1.0):
    h,w,c = mask.shape
    angle = np.degrees(np.arctan2(evs[1][0],evs[0][0]))
    centroid = totuple(centroid)
    M = cv2.getRotationMatrix2D(centroid,-angle,1)
    #print("To be rotated by {0:.2f} about {1}".format(angle,centroid))
    origDst = cv2.warpAffine(orig, M,(w,h),borderValue=0)
    maskDst = cv2.warpAffine(mask, M,(w,h),borderValue=0)
    return origDst,maskDst
    
def main(orig,mask,ellipseThresh = 20,redThresh = (0,60,250),outputWH =(150,100)):
    h,w,c = orig.shape
    mask = cv2.resize(mask,(w,h),interpolation = cv2.INTER_LINEAR)
    mask = fitEllipse(mask,ellipseThresh,250)
    centroidR, covR = getImgMoments(mask,0)
    centroidG, covG = getImgMoments(mask,1)
    e1,e2 = evs = getEigenVectors(centroid1=centroidG,centroid2=centroidR,cov=covR)
    orig, mask = rotate(orig,mask,evs,centroidR)
    mask = getRed(mask,redThresh)
    red = mask[:,:,0]
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(red)
    oW, oH = outputWH

    x, y = xy = maxLoc - np.array([oW,oH])

    x, y = xy = np.max([x,0]), np.max([y,0])
    x1,y1 = x1y1 = xy + 2*np.array([oW,oH])
    if x1 > w:
        x1 = w
        x = w - oW*2
    if y1 > h:
        y1 = y
        y = h - oH*2

    croppedHead = orig[y:y1,x:x1]
    return croppedHead, orig, mask

if __name__ == "__main__":
    import matplotlib.cm as cm
    def show(img,gray=0):
        if gray ==1:
            plt.imshow(img,cmap=cm.gray)
        else:
            plt.imshow(img)

        plt.show()
    import ipdb
    from random import shuffle
    imgPaths = glob.glob("/home/msmith/kaggle/whale/imgs/test/m1*")
    shuffle(imgPaths)
    i = 0
    while True:
        f = imgPaths[i]
        orig, mask = [cv2.imread(x)[:,:,::-1] for x in [f.replace("m1","w1"),f]]
        croppedHead, origO, maskO = main(orig,mask)
        c = raw_input("Press a key to continue..")
        i += 1

