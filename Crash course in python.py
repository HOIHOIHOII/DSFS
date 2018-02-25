#encoding= utf-8
from __future__ import division
from collections import Counter
from collections import defaultdict
from matplotlib import pyplot as plt

print 1.8*74, 2.5*74
print 0.6*74, 1.5*74

xs = range(7)
ys = [2536,2029,3043,2536,2283,2790,2536]

print (2536+2029+3043+2536+2283+2790+2536)/7

fig, axs = plt.subplots(1,1)


axs.plot(xs,ys)
plt.show()
