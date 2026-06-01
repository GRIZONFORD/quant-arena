print(
""" 
#############################

# Executing Python examples #

#############################
""")
import sys
#sys.path.append('../../../Source/TrueSkillThroughTime.py/')
import pandas as pd
import math; from numpy.random import normal, seed;
#from ttt import *
from trueskillthroughtime import *
import timeit
import time
from datetime import datetime
import trueskill as ts


runtimes = []

print("Code 1")
mu = 0.0; sigma = 6.0; beta = 1.0; gamma = 0.03

print("Code 2")
p1 = Player(Gaussian(mu, sigma), beta, gamma); p2 = Player(); p3 = Player(); p4 = Player()

print("Code 3")
team_a = [ p1, p2 ]
team_b = [ p3, p4 ]
teams = [team_a, team_b]
g = Game(teams)

runtimes.append(timeit.timeit(lambda: Game(teams), number=1000)/1000)

print("Code 4")
lhs = g.likelihoods
ev = g.evidence
ev = round(ev, 3)
print(ev)

print("Code 5")
pos = g.posteriors()
print(pos[0][0])
print(lhs[0][0] * p1.prior)

print("Code 6")
ta = [p1]
tb = [p2, p3]
tc = [p4]
teams = [ta, tb, tc]
result = [1, 0, 0]
g = Game(teams, result, p_draw=0.25)

runtimes.append(timeit.timeit(lambda: Game(teams, result, p_draw=0.25).posteriors, number=1000)/1000)

print("Code 7")
c1 = [["a"],["b"]]
c2 = [["b"],["c"]]
c3 = [["c"],["a"]]
composition = [c1, c2, c3]
h = History(composition, gamma=0.0)

runtimes.append(timeit.timeit(lambda: History(composition), number=1000)/1000)

print("Code 8")
lc = h.learning_curves()
print(lc["a"])
print(lc["b"])

print("Code 9")
h.convergence()
lc = h.learning_curves()
print(lc["a"])
print(lc["b"])

h = History(composition, gamma=0.0)
runtimes.append(timeit.timeit(lambda: h.convergence(iterations=1, verbose=False), number=1)/1)

print("Code 10")
seed(99); N=1000
def skill(experience, middle, maximum, slope):
    return maximum/(1+math.exp(slope*(-experience+middle)))

target = [skill(i, 500, 2, 0.0075) for i in range(N)]
opponents = normal(target,0.5)

print("Code 11")
composition = [[["a"], [str(i)]] for i in range(N)]
results = [[1,0] if normal(target[i]) > normal(opponents[i]) else [0,1] for i in range(N)]
times = [i for i in range(N)]
priors = dict([(str(i), Player(Gaussian(opponents[i], 0.2))) for i in range(N)])

h = History(composition, results, times, priors, mu=2.0, gamma=0.015)
h.convergence(iterations=1)
mu = [tp[1].mu for tp in h.learning_curves()["a"]]

print("Skill evolution runtime")
start = time.time()
h = History(composition, results, times, priors, mu=2.0, gamma=0.015)
end = time.time()
runtimes.append(end-start)
start = time.time()
h.convergence(iterations=1)
end = time.time()
runtimes.append(end-start)

print("ATP runtime")
df = pd.read_csv('../../Data/input/history.csv', low_memory=False)
columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id, df.double)
composition = [[[w1,w2],[l1,l2]] if d == 't' else [[w1],[l1]] for w1, w2, l1, l2, d in columns ]
times = [ datetime.strptime(t, "%Y-%m-%d").timestamp()/(60*60*24) for t in df.time_start]
start = time.time()
h = History(composition = composition, times = times, sigma = 1.6, gamma = 0.036)
h.convergence(epsilon=0.01, iterations=10)
end = time.time()
runtimes.append(end-start)

print("TrueSkill 0.4.5 two teams")
ts.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=gamma, draw_probability=0)
r1, r2 = (ts.Rating(),ts.Rating())
r3, r4 = (ts.Rating(),ts.Rating())
team_a = [r1, r2]
team_b = [r3, r4]
runtimes.append(timeit.timeit(lambda: ts.rate([team_a, team_b]), number=1000)/1000)

print("TrueSkill 0.4.5 three teams")
ts.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=gamma, draw_probability=0.25)
ta = [r1]
tb = [r2, r3]
tc = [r4]
teams = [ta, tb, tc]
result = [0, 1, 2]
runtimes.append(timeit.timeit(lambda: ts.rate(teams, result), number=1000)/1000)

with open('../../Data/output/runtimes.csv', 'a') as file:
    file.write('Python, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(*runtimes))