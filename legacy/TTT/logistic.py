import sys
# sys.path.append('../../../Source/TrueSkillThroughTime.py/')
import pandas as pd
import math; from numpy.random import normal, seed;
# from ttt import *
from trueskillthroughtime import *

seed(99); N=1000
def skill(experience, middle, maximum, slope):
    return maximum/(1+math.exp(slope*(-experience+middle)))

target = [skill(i, 500, 2, 0.0075) for i in range(N)]

mus = []
for _ in range(33):
    opponents = normal(target,0.5)
    composition = [[["a"], [str(i)]] for i in range(N)]
    results = [  [1.,0.] if normal(target[i]) > normal(opponents[i]) else [0.,1.] for i in range(N)]
    times = [i for i in range(N)]
    priors = dict([(str(i), Player(Gaussian(opponents[i], 0.2))) for i in range(N)])
    h = History(composition, results, times, priors, mu=2.0, gamma=0.015)
    h.convergence()
    mu = [tp[1].mu for tp in h.learning_curves()["a"]]
    sigma = [tp[1].sigma for tp in h.learning_curves()["a"]]
    mus.append(mu)

mus.append(target)

print("Data: ../../Data/output/logistic.csv")
df = pd.DataFrame({"true" : target, "mu" : mu, "sigma" : sigma})
df.to_csv("../../Data/output/logisitc.csv", index=False)

df = pd.DataFrame(mus)
df.to_csv("../../Data/output/logisitcs_mu.csv", index=False)

