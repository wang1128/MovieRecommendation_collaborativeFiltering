__author__ = 'penghao'
from scipy import spatial
import numpy as np
data1 = [-2.2,1.8,-0.2,-0.2,0.8]
data2 = [-1.667,0,0.333,0,1.33]


result = 1 - spatial.distance.cosine(data1, data2)
print result


arr = np.array([1, 3, 2, 4, 5])
a= arr.argsort()[-4:][::-1]
print a
print arr