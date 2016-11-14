import numpy as np
from scipy import stats

def anova(x):

    n, m = x.shape[0], x.shape[1]
    mu = x.mean()
    muG = x.mean(0,keepdims=1)

    print("Total mean and group means are {0} and {1}".format(mu,muG))

    tss = np.power(x - mu,2).sum()
    dof = n*m - 1

    print("Total sum of squares = {0}. DOF = {1}".format(tss,dof))

    ssw  = np.power(x - muG,2).sum() # sum of squares within groups
    dofW = m*(n-1)

    print("Total sum of squares within groups = {0}. DOF = {1}".format(ssw,dofW))

    ssb = ((muG - mu).repeat(n)**2).sum()
    dofB = m -1

    print("Total sum of squares between groups = {0}. DOF = {1}".format(ssb,dofB))

    fStat = (ssb/dofB)/(ssw/dofW)
    print("F stastistic (params) = {0} ({1},{2})".format(fStat,dofB,dofW))

    f = stats.f(dofB,dofW)

    print("Pvalue = {0}".format(1-f.cdf(fStat)))

if __name__ == "__main__":
    import pandas as pd
    x1 = np.array(pd.read_csv("/home/msmith/Google Drive/teaching/ST952/teaching.txt",delimiter=" ",header=None))
    x2 = np.array([[3,5,5],[2,3,6],[1,4,7]])
    print(x1)
    anova(x1)
