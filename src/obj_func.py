import numpy as np
from statistics import mean
from scipy.linalg import inv

from dataset import plot_data




data = """\
15/11/2022	100.63
14/11/2022	100.47
11/11/2022	100.43
10/11/2022	100.20
09/11/2022	100.13
08/11/2022	100.07
07/11/2022	100.18
04/11/2022	100.15
03/11/2022	100.26
02/11/2022	100.32
01/11/2022	100.36
31/10/2022	100.31
27/10/2022	100.16
26/10/2022	100.15
25/10/2022	100.00
24/10/2022	99.95
21/10/2022	99.72
20/10/2022	99.85
19/10/2022	99.95
18/10/2022	99.95
17/10/2022	100.06
14/10/2022	100.19
13/10/2022	100.06
12/10/2022	100.15
11/10/2022	99.88
10/10/2022	100.10
07/10/2022	100.34
06/10/2022	100.42
05/10/2022	100.52
04/10/2022	100.85"""


given_data = [float(i.split()[1]) for i in reversed(data.splitlines())]
mu = mean(given_data)
given_data = [d-mu for d in given_data]
given_data = np.array(given_data).T




def polynomial(t, a):
    return sum([ coef * t**i  for i,coef in enumerate(a) ])


# the matrix M
def M(data, d):
    # d = len(a)
    N = len(data)
    return 2/N * np.array( [[
        sum([t**(i+j) for t in range(1,N+1)])
        for j in range(d)] for i in range(d)], dtype=np.float64 )

# the vector v
def v(data, d):
    # d = len(a)
    N = len(data)
    return -2/N * np.array( [
        sum([  y_t * t**i for t,y_t in enumerate(data,1)  ])
        for i in range(d)
    ],dtype=np.float64).T

# y**2, y = 1/N**.5*data
def y2(data):
    return 1/len(data) * data.T @ data


d = 5
N = 25
M0 = M(given_data[:N], d)
half_M0 = 1/2 * M0
v0 = v(given_data[:N], d)
v0_T = v0.T
y_squared = y2(given_data)


def obj_func(a):
    a=np.array(a)
    return a.T @ half_M0 @ a + v0_T @ a + y_squared

def gradient_obj_func(a):
    a=np.array(a)
    return M0 @ a + v0

def hessian_obj_func(a):
    a=np.array(a)
    return M0

minimizer = -inv(M0) @ v0#solve(M0, -v0.T)
#         == array([ 7.02693017e-01, -1.61143480e-01,  2.67043227e-03,  5.48709328e-04, -1.80154678e-05])

minimum_polynomial = lambda t: polynomial(t, minimizer)


def plot_against(x_opt, started_at):
    print(
f"""
                  started at = {started_at}

              computed x_opt = {tuple(x_opt)}
                   minimizer = {tuple(minimizer)}
           x_opt - minimizer = {tuple(minimizer-x_opt)}

    computed obj_func(x_opt) = {obj_func(x_opt)}
     minimum obj_func(x_opt) = {obj_func(minimizer)}
                  difference = {obj_func(x_opt) - obj_func(minimizer)}
""")

    from main import Pm, create_polynomial_matrix

    reverse_x_opt = list(x_opt)
    reverse_x_opt.reverse()

    t = create_polynomial_matrix(days=25, nth_power=4)
    plot_data(given_data, N, Pm(t,reverse_x_opt))
    
    
def random_point(dimensions):
    from random import randint, random, choice

    return tuple([
                    choice([-1,1]) * random() * randint(-5,5)
                    for _ in range(dimensions)
    ])




if __name__ == "__main__":
    plot_data(given_data, N, [minimum_polynomial(t) for t in given_data[:N]])