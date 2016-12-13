import cv2
import numpy as np 
import matplotlib.pyplot as plt
import glob
from pylab import rcParams
from scipy.ndimage import measurements
import matplotlib.cm as cm
from fitEllipse import fitEllipse

def largestContour(contours):
    largestArea = 0
    largestIdx = 0
    nCnts = len(contours)
    for i in xrange(nCnts):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h > largestArea:
            largestArea = w*h
            largestIdx = i
    return contours[largestIdx], largestIdx

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

def getRedHSV(img,hThr=(0.04,0.96),vThr=(0.1)):
    HSV = cv2.cvtColor(img,cv2.COLOR_RGB2HSV).astype(np.float32)
    H = HSV[:,:,0]
    V = HSV[:,:,2]
    H /= 179.0 # cv2 values for HUE in 0, 179
    V /= 255.0 # cv2 values for Value in 0, 255
    r = ((H<hThr[0]) | (H>hThr[1])) & (V>vThr)
    return r.astype(np.float32)

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
    
def main(orig,mask,ellipseThresh = 20,  hThr= [0.04,0.96], vThr=0.1, cntThresh = 0.001, pad = 20, aspectRatio = 1.4):
    h,w,c = orig.shape
    mask = cv2.resize(mask,(w,h),interpolation = cv2.INTER_LINEAR)
    mask = fitEllipse(mask,ellipseThresh,250)
    centroidR, covR = getImgMoments(mask,0)
    centroidG, covG = getImgMoments(mask,1)
    e1,e2 = evs = getEigenVectors(centroid1=centroidG,centroid2=centroidR,cov=covR)
    orig, mask = rotate(orig,mask,evs,centroidR)

    enoughContours = False

    
    countTry = 0 # number of attempts to get contours
    fail = 0
    while enoughContours == False:
        red = getRedHSV(mask,hThr,vThr)
        if countTry > 5:
            fail = 1
            return None, None, None, None, fail
        redC = red.copy()
        ret, thresh = cv2.threshold(redC,cntThresh,1,cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)
        threshC = thresh.copy()
        contours, hierarchy = cv2.findContours(threshC,1,2)
        if len(contours) == 0 : # if we have no contours change thresholding
            hThr[0] += 0.001
            hThr[1] -= 0.001
            vThr -= 0.001
            countTry += 1
            continue
        else: 
            enoughContours = True
            break


    largestCnt,_ = largestContour(contours)
    x,y,dx,dy = cv2.boundingRect(largestCnt)
    if dx*dy < 5000:
        # rubbish area size, quit
        fail = 2
        return None, None, None, None, fail

    x-= pad
    y-= pad
    dx += 2*pad
    dy += 2*pad

    x1 = x + dx 
    y1 = y + dy 
    #cv2.rectangle(mask,(x,y),(x1,y1),(0,255,0),30)
    #cv2.rectangle(red,(x,y),(x1,y1),255,10)
    #cv2.rectangle(thresh,(x,y),(x1,y1),255,10)
    croppedHead = orig[y:y1,x:x1]
    return croppedHead, mask, red, thresh, fail

if __name__ == "__main__":
    import matplotlib.cm as cm
    import ipdb
    def show(img,gray=0):
        if gray ==1:
            plt.imshow(img,cmap=cm.gray)
        else:
            plt.imshow(img)

        plt.show()
    import ipdb
    from random import shuffle
    imgPaths = ["/home/msmith/kaggle/whale/imgs/whale_58972/m1_1542.jpg","/home/msmith/kaggle/whale/imgs/whale_58747/m1_3428.jpg"]
    #imgPaths = glob.glob("/home/msmith/kaggle/whale/imgs/test/m1*")
    shuffle(imgPaths)
    i = 0
    while True:
        f = imgPaths[i]
        orig, mask = [cv2.imread(x)[:,:,::-1] for x in [f.replace("m1","w1"),f]]
        croppedHead, origO, maskO, red = main(orig,mask,cntThresh=0.01)
        print(croppedHead.shape)
        ipdb.set_trace()
        #c = raw_input("Press a key to continue..")
        i += 1

