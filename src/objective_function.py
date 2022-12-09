import numpy as np
from statistics import mean
import random
from random import randint, random
from math import isclose



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


data = [float(i.split()[1]) for i in reversed(data.splitlines())]
mu = mean(data)
data = [d-mu for d in data]
data = np.array(data).T




# a polynomial with coefficients a[0], ..., a[n]
def polynomial(t, a):
    return sum([a[i] * t**i  for i in range(len(a))])


# the objective function
def f(a, data):
    return 1/len(data) * sum([
                            (polynomial(t,a)-y_t)**2
                            for t,y_t in enumerate(data,1)
                        ])


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

# f is rewritten in this form.
# hopefully, it's calculation can be more efficient; instead of summation
# we use matrix multiplication; M, v, y2 are independent of a, therefore
# we only need to compute them once for each dataset we are given.
def f_(a, data):
    d=len(a)
    return 1/2* a.T @ M(data,d) @ a + v(data,d).T @ a + y2(data)


# test wether f(a,data) == f_(a,data)
# we know that:
# f(a, fitted_data) == 0, where the data are fitted such as y_t := p(t,a)
# f(0, data) == y**2
# f(a, [0,...,0]) == 1/2 * a.T @ M @ a
def test_implementation(a=None):
    d = randint(1,5)
    if a is None:
        a = np.array([(-1)**randint(1,2)*5*random() for i in range(d)],dtype=np.float64)
    else:
        a = np.array(a)
    a_zero = np.array(d*[0.0],dtype=np.float64)
    random_data = np.array([random() for t in range(1,26)],dtype=np.float64)
    fitted_data = np.array([polynomial(t,a) for t in range(1,26)],dtype=np.float64)
    zeroes = np.array(25*[0.0],dtype=np.float64)

    print(f"\n\n{a=}    <= random")
    closeness = isclose(f(a,random_data),f_(a,random_data))
    print(f""" \n{closeness=}
          {f(a,random_data)=}
      == {f_(a,random_data)=}""")
    closeness = isclose(f(a,fitted_data),f_(a,fitted_data))
    print(f""" \n{closeness=}  => everything looks okay, if they're close enough
          {f(a,fitted_data)=}  <= this is always 0.0
      == {f_(a,fitted_data)=}""")
    closeness = isclose(f(a,zeroes),f_(a,zeroes))
    print(f""" \n{closeness=}  => M is correct
          {f(a,zeroes)=}
      == {f_(a,zeroes)=}""")
    closeness = isclose(f(a,data),f_(a,data))
    print(f""" \n{closeness=}  => applied on the data from our problem
          {f(a,data)=}
      == {f_(a,data)=}""")
    closeness = isclose(f(a_zero,data),f_(a_zero,data))
    print(f""" \n{closeness=}  => y2 is correct
          {f(a_zero,data)=}
      == {f_(a_zero,data)=}""")


if __name__ == "__main__":
    while 1:
        test_implementation()
        input()