import numpy as np
import config

def create_polynomial_matrix(days: int, nth_power: int) -> np.array:
    """
    
    Creates a daysX(nth_power + 1) matrix, where rows represents the powers of the polyonimal for each day.
    ex.
    [[     1      1      1      1      1]
     [     1      2      4      8     16]
     [     1      3      9     27     81]
     [     1      4     16     64    256]
     ...
     [     1     24    576  13824 331776]
     [     1     25    625  15625 390625]]
    """
    powers = [i for i in range(nth_power+1)]
    tn = np.array([np.arange(1, days + 1, 1)])
    t = (np.power(tn.reshape(tn.size, 1), powers))

    return t


def Pm(t: np.array, a: np.array) -> np.array:
    """
    Calculates the polynomial
    """
    return t@a

def objective_function(y, polynomial_function):
    config.function_calls += 1
    return np.mean(np.square(y - polynomial_function))

def gradient(y, polynomial_function, t):

    N = len(y)

    return -2/N * np.array([[np.sum((y - polynomial_function).T@t[:, 0])],
                            [np.sum((y - polynomial_function).T@t[:, 1])],
                            [np.sum((y - polynomial_function).T@t[:, 2])],
                            [np.sum((y - polynomial_function).T@t[:, 3])],
                            [np.sum((y - polynomial_function).T@t[:, 4])]])



def hessian(data, d):

    N = len(data)
    return 2/N * np.array( [[
        sum([t**(i+j) for t in range(1,N+1)])
        for j in range(d)] for i in range(d)], dtype=np.float64 )

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def calc_new_H(y, t, gf, lr, pk, x_new, H):
    
    s = lr * pk
    gf_new = gradient(y, Pm(t, x_new),t)
    y_h = gf_new - gf
    y_h = np.array([y_h])
    s = np.array([s])
    y_h = np.reshape(y_h, (5, 1))
    s = np.reshape(s, (5, 1))
    r = 1/(y_h.T@s)
    li = (np.eye(5)-(r*((s@(y_h.T)))))
    ri = (np.eye(5)-(r*((y_h@(s.T)))))
    hess_inter = li@H@ri
    
    return hess_inter + (r*((s@(s.T))))