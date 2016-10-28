import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob, cv2
from pylab import rcParams
rcParams["figure.figsize"] = 25,10

def show(img,gray=0):
    if gray == 1:
        plt.imshow(img,cmap = cm.gray); plt.show();
    else:
        plt.imshow(img); plt.show();

def brgToYuv(img):
    ''' Convert to YUV '''
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def yuvToBrg(img):
    ''' Convert back to BRG '''
    return cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

def cdfImg(img):
    ''' Returns Cumulative distribution x and F(X)'''
    imFlat = img.flatten()
    x, counts = np.unique(imFlat, return_counts=True)
    cdfx = np.cumsum(counts)
    return x, cdfx

def findNearest(array,value):
    ''' Finds nearest value in array and returns the value and its index in the array'''
    idx = (np.abs(array.astype(np.float32)-value)).argmin()
    return array[idx], idx

def histMatch(img1,img2):
    ''' Takes one channeled images;
    Image 1 is converted to have same CDF as image 2 
    Return: 
    Transformed image 1
    Look up table
    '''

    x1, cdf1 = cdfImg(img1)
    x2, cdf2 = cdfImg(img2)
    mapping = np.empty([2**8]) # 256 value look up table

    uint8 = np.arange(mapping.shape[0])
    for g1 in uint8:
        _,x1Nearest = findNearest(x1,g1)
        f1g1 = cdf1[x1Nearest] # Find CDF1 of current value
        f2g2, f2g2I = findNearest(cdf2,f1g1) # return index of nearest value in CDF2
        g2 = x2[f2g2I]

        mapping[g1] = g2

    dst = mapping[img1].astype(np.uint8) # reassign pixel values

    return dst, mapping

def histMatchAllChannels(img1,img2): 

    ''' Wrapper function given two images we match the hist of the first image to the second for each color channel in YUV'''

    yuv1, yuv2 = [brgToYuv(x) for x in [img1,img2]] # Convert to YUV
    dst = np.zeros(yuv1.shape) # Init our final image
    mappings = np.zeros((3,256))
    for chan in range(3): # for YUV
        c1, c2 = [x[:,:,chan] for x in [yuv1, yuv2]]
        dst[:,:,chan], mappings[chan] = histMatch(c1,c2)

    dst = yuvToBrg(dst.astype(np.uint8)) ## Convert back to RGB
    return dst, mappings

if __name__ == "__main__":
    import ipdb

    pass





