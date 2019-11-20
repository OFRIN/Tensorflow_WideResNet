
import numpy as np

from Define import *

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()

def smooth_one_hot(v, classes, delta = 0.001):
    uniform_distribution = np.full(classes, 1. / classes)
    v = (1 - delta) * v + delta * uniform_distribution
    return v