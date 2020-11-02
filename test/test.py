# coding: utf-8
from common.functions import *
import numpy as np

if __name__ == '__main__':
    x = np.array([1, 0, 0, 0, 0])
    t = np.array([0, 0, 0, 0, 0])
    print(cross_entropy_error(x, t))
