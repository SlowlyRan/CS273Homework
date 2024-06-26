import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
from logisticClassify2 import *

iris = np.genfromtxt("data/iris.txt",delimiter=None)
X, Y = iris[:,0:2], iris[:,-1] # get first two features & target
X,Y = ml.shuffleData(X,Y) # reorder randomly (important later)
X,_ = ml.rescale(X) # works much better on rescaled data
XA, YA = X[Y<2,:], Y[Y<2] # get class 0 vs 1
XB, YB = X[Y>0,:], Y[Y>0] # get class 1 vs 2
learner = logisticClassify2(); # create "blank" learner
learner.classes = np.unique(YA) # define class labels using YA or YB
wts = np.array([0.5,-0.25,1.0]); # TODO: fill in values
learner.theta = wts; # set the learner’s parameters
yh = learner.predict(XA)
e = learner.err(XA,YA)
learner.train_with_l2(XA,YA)
