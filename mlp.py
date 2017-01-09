import tensorflow as tf
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

def show(x,y):
    plt.plot(x,y)
    plt.show()

def genData():
    nObs = 1000
    x = np.linspace(-2,2,nObs)
    f = lambda x: 0.3*x + 1.2*x**3 + 0.4*np.cos(x*0.4)
    yT = f(x) #Truth
    y = yT + rng.normal(0,1,nObs)
    return x,y

show(x,y)
