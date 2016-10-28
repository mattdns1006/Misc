import numpy as np
import pandas as pd
import os,glob
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline as spline
from scipy.optimize import curve_fit 
import histMatch


def feeder(data,shuffle=False,pad=10):

    '''
    Takes a list of file paths or 4 d tensor
    Returns generator yielding the cdf mapping between two images YUV channels
    '''

    # Create X
    nPoints = 256
    X = np.arange(nPoints).astype(np.float32)
    X /= 255.0 
    # Pad X
    X = np.pad(X,[(pad,pad)],"linear_ramp",end_values=(-0.25,1.25))

    while True:

        if type(data) == list: 
            n = len(data)
            c1, c2 = np.random.choice(n,2)
            img1, img2 = cv2.imread(data[c1]),cv2.imread(data[c2])
        elif type(data) == np.ndarray:
            n = data.shape[0]
            c1, c2 = np.random.choice(n,2)
            img1, img2 = data[c1], data[c2]
        
        # Add padding on each side of array to help extremes of fn be learned
        dst, mappings = histMatch.histMatchAllChannels(img1,img2)
        mappings /= 255.0 

        # Pad Y
        mappings = np.pad(mappings,((0,0),(pad,pad)),"edge")

        x,y = [arr.astype(np.float32) for arr in [X,mappings]]

	yield x,y

def mse(yPred,y):
        '''
        Mean squared error loss function 
        '''
	return np.square(y-yPred).mean()


def tanh(x,a,b,c,d,e,f,g):
        '''
        Main function to be optimized
        '''
	return a*np.tanh((x-b)/c) + d + e*np.arctanh((x-f)/g) 


def fit(x,y):
        '''
        Fit tanh curve to data x,y
        x - all uint8 values
        y - single mapping
        '''

        x[np.where(x<=0)] = 0.001
        x[np.where(x>=1)] = 1 - 0.001
        p0 = [0.4,0.4,0.13,0.52,0.03,0.00,1.2] # Initial value
        ipdb.set_trace()
        coeffs, cov = curve_fit(tanh,x,y,p0=p0,maxfev=100000)

        model = lambda x: tanh(x,coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4],coeffs[5],coeffs[6])

	yPred = model(x)

	return coeffs

def fitAllChannels(x,YUV):
        '''
        Fit tanh curve to each channel data x,Y;
        x - all uint8 values
        YUV - YUV mappings
        '''

        coeffs = {} 
        coeffs["Y"] = fit(x=x,y=YUV[0])
        coeffs["U"] = fit(x=x,y=YUV[1])
        coeffs["V"] = fit(x=x,y=YUV[2])

        return coeffs

if __name__ == "__main__":

	import matplotlib.pyplot as plt
	import ipdb, glob, sys
        sys.path.append("/home/msmith/kaggle/cifar10/")
        from unpickle import unpickle
        data = unpickle("/home/msmith/kaggle/cifar10/cifar-10-batches-py/data_batch_1")
        X, Y = data["data"], data["labels"]
        X = X.reshape(-1,3,32,32).transpose(0,2,3,1)

        feed = feeder(X)
        for i in xrange(10):
            x, y = feed.next()
            coeffs = fitAllChannels(x,y)








