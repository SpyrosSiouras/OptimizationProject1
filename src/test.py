# import numpy as np


# def Pm(t: np.array, a: np.array) -> float:
#     """
#     This function returns a 4th grade polyonimal.
#     Inputs: t -> (1,5) dimensional numpy array, which has the powers of t in descending order ex. for t=2 the array is => [16, 8, 4, 2, 1]
#             a -> (5,1) dimensional numpy array, which has the coeffiecients of t.
#     """
#     return t@a


# def f(a, t):
#     y = np.array([100.07, 100.18, 100.15, 100.26, 100.32,
#                   100.36, 100.31, 100.16, 100.15, 100.00,
#                   99.95, 99.72, 99.85, 99.95, 99.95, 100.06,
#                   100.19, 100.06, 100.15, 99.88, 100.10, 100.34, 100.42, 100.52, 100.85])
#     y = y.reshape(25, 1)
#     return np.mean(np.square(y - Pm(t, a)))


# def gradient_f(t):
#     return np.array([-t[:, 0], -t[:, 1], -t[:, 2], -t[:, 3], -t[:, 4]])


# n = 4
# powers = [n-i for i in range(n+1)]
# print(powers)
# tn = np.array([np.arange(1, 26, 1)])
# t = (np.power(tn.reshape(tn.size, 1), powers))
# print(t[1].reshape(1, 5).shape)
# a = np.array([[0.0001], [0], [1], [1.5], [1.3]])
# # print(Pm(t, a))
# print(gradient_f(t)[0])
# # print(f(a, t))
import numpy as np
from nptyping import NDArray, Shape


tn = np.array([np.arange(1, 4, 1)])

print(tn - 1)
