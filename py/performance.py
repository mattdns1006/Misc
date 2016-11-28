import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Performance():
    def __init__(self,name="Performance"):
        self.epochLosses = []
        self.epochScores = []
        self.epochLossesStd = []
        self.epochScoresStd = []
        self.losses = []
        self.scores = []
        self.name = name
    def add(self,loss,score):
        self.losses.append(loss)
        self.scores.append(score)
    def endEpoch(self):
        self.epochLosses.append(np.array(self.losses).mean())
        self.epochScores.append(np.array(self.scores).mean())
        self.epochLossesStd.append(np.array(self.losses).std())
        self.epochScoresStd.append(np.array(self.scores).std())
        self.losses = []
        self.scores = []

    def currentPerformance(self):
       	print("{0} losses ==> {1}".format(self.name,np.array(self.losses).mean()))
       	print("{0} scores ==> {1}".format(self.name,np.array(self.scores).mean()))

    def displayEpochPerformance(self):
       	print("{0} losses ==> {1}".format(self.name,self.epochLosses))
       	print("{0} scores ==> {1}".format(self.name,self.epochScores))

        x = np.arange(len(self.epochLosses))
        losses = np.array(self.epochLosses)
        scores = np.array(self.epochScores)
        plt.subplot(121)
        plt.plot(x,losses)
        plt.title("{0} epoch losses.".format(self.name))
        plt.subplot(122)
        plt.plot(x,scores)
        plt.title("{0} epoch scores.".format(self.name))
        plt.show()

    def displayCurrentPerformance(self):
        x = np.arange(len(self.losses))
        losses = np.array(self.losses)
        scores = np.array(self.scores)
        plt.subplot(121)
        plt.plot(x,losses)
        plt.title("{0} current names.".format(self.name))
        plt.subplot(122)
        plt.plot(x,scores)
        plt.title("{0} current scores.".format(self.name))
        plt.show()

    def writeToCsv(self,name):
        assert name != None, "Please provide name"
        df = pd.DataFrame(np.array([self.epochLosses,self.epochLossesStd,self.epochScores,self.epochScoresStd]).T)
        df.columns = ["losses mu","losses std","scores mu","scores std"]
        df.to_csv(name,index=0)


if __name__ == "__main__":
    nEpochs = 10
    import ipdb
    perf = Performance()
