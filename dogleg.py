import numpy as np

from problem_functions import *
import config

from math import isclose as is_close






def is_almost_zero(num: float, tolerance: float = 10**-9) -> bool:
    return is_close(num, 0.0, rel_tol=0.0, abs_tol=tolerance)

def bisection_method(
                        function,
                        left: float,
                        right: float,
                        max_iterations: int = 20
                    ) -> float:
    """
    Finds a root of a function in (left, right)

    It assumes that function(left) * function(right) < 0 and left < right.
    """
    if function(left) > 0:
        pos = left
        neg = right
    else:
        pos = right
        neg = left

    for _ in range(max_iterations):
        middle = (pos+neg)/2
        fm = function(middle)
        if is_almost_zero(fm) or is_close(pos, neg):
            return middle

        if fm > 0:
            pos = middle
        else:
            neg = middle

    return middle


def dogleg_point(region_radius: float, gradfk: np.ndarray, Hk: np.ndarray, Bk: np.ndarray) -> np.ndarray:
    """
    Computes the Dogleg point
    """

    pB = - Hk @ gradfk

    if np.linalg.norm(pB) <= region_radius:
        return pB

    gBgT = (gradfk.T @ Bk @ gradfk)
    gradfk_squared = (gradfk.T @ gradfk)
    pU = - gradfk_squared/gBgT * gradfk
    if not np.linalg.norm(pU) >= region_radius:
        return - region_radius/np.linalg.norm(gradfk) * gradfk

    def func(t):
        vec = pU + (t-1) * (pB-pU)
        return np.linalg.norm( vec.T @ vec ) - region_radius**2

    tau = bisection_method(func, 1, 2)

    return pU + (tau-1) * (pB-pU)


def calc_new_HB(y, t, gf, pk, x_new, H, B):

    s = pk
    gf_new = gradient(y, Pm(t, x_new),t)
    y_h = gf_new - gf
    if is_close(np.linalg.norm(y_h), 0):
        return H,B
    y_h = np.array([y_h])
    s = np.array([s])
    y_h = np.reshape(y_h, (5, 1))
    s = np.reshape(s, (5, 1))
    r = 1/(y_h.T@s)
    li = (np.eye(5)-(r*((s@(y_h.T)))))
    ri = (np.eye(5)-(r*((y_h@(s.T)))))

    new_H = li@H@ri + + (r*((s@(s.T))))

    BssTB = B @ s @ s.T @ B
    sTBs = s.T @ B @ s
    yyT = y_h.T @ y_h
    yTs = y_h.T @ s

    new_B = B - BssTB/sTBs + yyT/yTs

    return new_H, new_B


def new_radius(direction, improvement_rate, radius, max_radius) -> float:
    if improvement_rate < 0.25:
        return radius * 0.25
    elif improvement_rate > 0.75 and is_close(np.linalg.norm(direction), radius):
        return min(2*radius, max_radius)
    else:
        return radius


def doglegBFGS(y, t, x_inputs, max_iterations, accuracy, max_radius=1):

    B = hessian(y, 5)
    H = np.linalg.inv(B)

    region_radius = 1/2*max_radius

    while True:
        x = x_inputs[-1]
        Pm_x = Pm(t, x)
        gf = gradient(y, Pm_x,t)
        fx = objective_function(y, Pm_x)

        if config.function_calls > max_iterations:
            print(f"Couldn't reach desired accuracy ({np.linalg.norm(gradient(y, Pm(t, x_inputs[-1]), t))} >  {accuracy})!")
            break
        elif (np.linalg.norm(gf) < accuracy):
            break

        pk = dogleg_point(region_radius, gf, H, B)

        x_cand = x + pk
        Pm_x_cand = Pm(t, x_cand)
        gf_cand = gradient(y, Pm_x_cand,t)
        fx_cand = objective_function(y, Pm_x_cand)

        mk = fx +  gf.T @ pk  +  1/2 * (pk.T @ B @ pk)

        improvement_rate = ( fx - fx_cand ) / (fx - mk)

        region_radius = new_radius(pk,improvement_rate, region_radius, max_radius)

        if improvement_rate > 0.2:
            x_inputs.append(x_cand)
            x = x_cand
            Pm_x = Pm_x_cand
            gf = gf_cand
            fx = fx_cand
        else:
            continue

        H, B = calc_new_HB(y, t, gf, pk, x, H, B)

    return x_inputs
