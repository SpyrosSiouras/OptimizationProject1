from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from mathchinery import Function, Vector, DiffFunction, Diff2Function
from typing import Self, List, Optional, Callable
from collections.abc import Iterator
from warnings import warn
from functools import lru_cache, cached_property

import numpy as np
from numpy import ndarray, eye
from scipy.linalg import solve, LinAlgError
from mathchinery import Point, Vector, Function
from mathchinery import FunctionNotDifferentiableError, FunctionNotTwiceDifferentiableError
from random import random        
from math import isclose


#### todo ADD
#### todo DOCs
#### todo EVERYWHERE!!!

from algorithms import GenericAlgorithm ### todo delete me


minimal_req = """
1. BFGS με Wolfe conditions line search
2. Dogleg BFGS
3. Newton με Wolfe conditions line search
4. Steepest descent με Wolfe conditions line search
"""





# mixins
# https://rednafi.github.io/digressions/python/2020/07/03/python-mixins.html




class GradientBasedMethod:
    """A mixin class implementing functionality for methods that use gradients"""

    _DEFAULT_GRADIENT_TOLERANCE = 10**-4

    @property
    def gradient_tolerance(self) -> float:
        try:
            return self._tolerance
        except AttributeError:
            self._gradient_tolerance = self._DEFAULT_GRADIENT_TOLERANCE
            return self._gradient_tolerance

    @gradient_tolerance.setter
    def gradient_tolerance(self, tolerance: float) -> None:
        self._gradient_tolerance = tolerance

    @property
    def gradient(self) -> Function:
        """Returns the gradient of the objective function"""
        try:
            return self.objective_function.gradient
        except AttributeError:
            pass
        raise FunctionNotDifferentiableError(self.objective_function)

    @property
    def gradf(self) -> Function:
        """
        Returns the gradient of the objective function

        A shorter alias for gradient
        """
        return self.gradient

    @property
    def gradfk(self) -> Vector:
        """Returns the gradient at the current point xk"""
        return self.gradf(self.xk)

    def is_descent_direction(self, vector: Vector) -> float:
        """Is the vector a descent direction?"""
        return vector * self.gradfk < 0

    def is_pk_descent_direction(self) -> float:
        return self.is_descent_direction(self.pk)

    def is_gradient_almost_zero(self) -> bool:
        """Is the gradient close enough to zero?"""
        return isclose( abs(self.gradfk), 0, abs_tol=self.gradient_tolerance )

    @property
    def total_evaluations(self) -> int:
        return self.objective_function.total_evaluations

    @property
    def total_unweighted_evaluations(self) -> int:
        return self.objective_function.total_unweighted_evaluations







    #https://github.com/gjkennedy/ae6310
    #https://github.com/gjkennedy/ae6310/blob/master/Line%20Search%20Algorithms.ipynb
    #https://indrag49.github.io/Numerical-Optimization/line-search-descent-methods.html
class LineSearch(GradientBasedMethod):
    """A mixin class implementing functionality for methods that use line searches"""

    _DEFAULT_c1 = 10**-4
    _DEFAULT_c2 = 0.9
    _DEFAULT_MAX_ITERATIONS_IN_LINE_SEARCH = 10

    def phi(self: GenericAlgorithm, a: float) -> float:
        """Returns the value objective_function(xk + a*pk)"""
        return self.f( self.xk + a * self.pk )

    @property
    def phi0(self: GenericAlgorithm) -> float:
        """Returns the value objective_function(xk)"""
        return self.fk

    def dphi(self: GenericAlgorithm, a: float) -> float:
        """Returns the value objective_function.gradient(xk + a*pk)"""
        return self.pk * self.gradf( self.xk + a * self.pk )

    @property
    def dphi0(self: GenericAlgorithm) -> float:
        """Returns the value objective_function.gradient(xk)"""
        return self.pk * self.gradfk

    def is_armijo_met(self: GenericAlgorithm, a: float, c1: float = _DEFAULT_c1) -> bool:
        """Does the Armijo condition hold for the objective_function on xk + a*pk?"""
        return self.phi(a)  <=  self.phi0 + c1 * a * self.dphi0

    def is_strong_curvature_met(self: GenericAlgorithm, a: float, c2: float) -> bool:
        """Does the strong curvature condition hold for the objective_function on xk + a*pk?"""
        return abs(self.dphi(a))  <=  - c2 * self.dphi0

    def report_line_search_failure(self: GenericAlgorithm, method_name: str, reason: Optional[str] = None ) -> None:
        self.report_warning(
            f"the {method_name} failed after {self._DEFAULT_MAX_ITERATIONS_IN_LINE_SEARCH} iterations {'' if reason is None else reason }"
        )

    def backtrack_line_search_with_armijo(
                                            self,
                                            a_initial: float = 1,
                                            c1: float = _DEFAULT_c1,
                                            max_iterations: int = _DEFAULT_MAX_ITERATIONS_IN_LINE_SEARCH
                                          ) -> float:
        """
        Returns a step length where the Armijo condition is satisfied, if such a point is found in 'max_iterations' iterations.
        Otherwise, it presents a warning and it returns the last candidate length examined.
        """
        if not self.is_pk_descent_direction():
            raise ValueError(f"the direction pk={self.pk} at the current point xk={self.xk} is not a descent direction")

        a_cnd = a_initial
        for i in range(max_iterations):
            if self.is_armijo_met(a_cnd, c1):
                return a_cnd
            a_cnd *= 0.5
        self.report_line_search_failure( method_name = "backtracking line search with the Armijo condition" )
        return a_cnd

    def _zoom(
                self,
                a_lowest_armijo: float,
                a_other: float,
                c1: float = _DEFAULT_c1,
                c2: float = _DEFAULT_c2,
                max_iterations: int = _DEFAULT_MAX_ITERATIONS_IN_LINE_SEARCH
             ) -> float:
        """
        Assuming that a point satisfying the Wolfe conditions exists between in one of the intervals
        (a_lowest_armijo,a_other) or (a_other,a_lowest_armijo), the method zooms in the interval
        until it is found or the iterations are used up.
        In the latter case, the method returns the point with the minimum functional value that
        satisfies the Armijo condition.

        Part of the line search with strong Wolfe conditions algorithm as documented in Numerical
        Optimization by J.Nocedal and S.Wright.
        """

        for i in range(max_iterations):
            a = (a_lowest_armijo + a_other)/2

            if not self.is_armijo_met(a, c1) or self.phi(a) >= self.phi(a_lowest_armijo):
                a_other = a
            else:
                if self.is_strong_curvature_met(a, c2):
                    return a
                if self.dphi(a) * (a_other - a_lowest_armijo) >= 0:
                    a_other = a_lowest_armijo
                a_lowest_armijo = a

        return a



    def line_search_with_wolfe_conditions(
                                            self: GenericAlgorithm,
                                            a_max: float,
                                            c1: float = _DEFAULT_c1,
                                            c2: float = _DEFAULT_c2,
                                            max_iterations: int = _DEFAULT_MAX_ITERATIONS_IN_LINE_SEARCH
                                          ) -> float:
        """
        Performs a line search with strong Wolfe conditions on the current direction

        The method is implemented as documented in Numerical Optimization by J.Nocedal and S.Wright.

        Typical values for 0<c1<c2<1:
            c1 = 10**-4,
            c2 = 0.9 in Quasi Newton methods,
            c2 = 0.1 in Conjugate Gradient methods
        """
        if not self.is_pk_descent_direction():
            raise ValueError(f"the direction pk={self.pk} at the current point xk={self.xk} is not a descent direction")

        a_prev = 0
        a = (a_prev + a_max)/2

        for i in range(max_iterations):
            if not self.is_armijo_met(a, c1) or self.phi(a)>=self.phi(a_prev):
                return self._zoom(a_prev, a, c1, c2, max_iterations)
            if self.is_strong_curvature_met(a, c2):
                return a
            if self.dphi(a) >= 0:
                return self._zoom(a, a_prev, c1, c2, max_iterations)

            a_prev = a
            a = (a_prev + a_max)/2

        return a



class SteepestDescentDirection(GradientBasedMethod):
    """A mixin class that implements the steepest descent direction"""

    def steepest_descent_direction(self) -> Vector:
        """Returns the direction of the steepest descent"""
        return - self.gradfk







class QuadraticModel(GradientBasedMethod, ABC):
    """A mixin class implementing functionality for methods that use quadratic models"""

    @property
    @abstractmethod
    def hessian_approximation(self) -> Function:
        """Returns a matrix valued function that approximates the hessian of the objective function"""
        raise NotImplementedError

    @property
    def Bk(self) -> ndarray:
        """The hessian_approximation at the current point xk"""
        return self.hessian_approximation(self.xk)







class NewtonDirection(QuadraticModel):
    """A mixin class that implements the Newton direction"""

    @property
    def hessian(self) -> Function:
        """Returns a function that computes the hessian of the objective function"""
        try:
            return self.objective_function.hessian
        except AttributeError:
            raise FunctionNotTwiceDifferentiableError from None

    @property
    def hessian_approximation(self) -> Function:
        return self.hessian

    @property
    def hessfk(self) -> ndarray:
        """Returns the hessian at the current point xk"""
        return self.hessian(self.xk)

    def newton_direction(self) -> Vector:
        """
        Returns the Newton direction at the current point xk

        There is no check whether the hessian is positive.
        """
        newton = solve(self.hessfk, -self.gradfk.T, assume_a="sym")
        return Vector(newton.T)

    def newton_direction_with_hessian_modification(self) -> Vector:
        """
        Returns the Newton direction biased toward a direction that makes the hessian positive
        """
        hessfk_isnt_positive = True
        H = self.hessfk
        m = min(H.diagonal())
        I = eye(self.domain_dimensions)
        b = 10**-3

        if m > 0:
            t = 0
        else:
            t = -m + b

        while hessfk_isnt_positive:
            try:
                newton = solve(self.hessfk + t*I, -self.gradfk.T, assume_a="pos") # this call fails when the hessian is not positive
                return Vector(newton.T)
            except LinAlgError:
                t = max(10*t, b)








class TrustRegionMethod(QuadraticModel):
    """A mixin class implementing functionality for trust region methods"""
    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, number: float) -> None:
        self._radius = number

    def update_radius(self) -> None:
        raise NotImplementedError

    @property
    def hessian_approximation(self) -> Function:
        return self._hessian_matrix_approximation

    @hessian_approximation.setter
    def hessian_approximation(self, matrix: ndarray) -> Function:
        self._hessian_matrix_approximation = matrix


    def cauchy_point(self) -> Vector:
        gBgT = self.gradfk @ self.Bk @ self.gradfk.T[0]
        absgradfk = abs(self.gradfk)
        if gBgT <= 0:
            tau = 1
        else:
            tau = min(1, absgradfk**3/(self.radius*gBgT) )
        return - (tau * self.radius) / absgradfk * self.gradfk

    def dogleg_point(self) -> Vector:
        raise NotImplementedError



class QuasiNewton(QuadraticModel):
    """A mixin class implementing functionality for quasi Newton methods"""

class BFGSUpdate(QuasiNewton):
    """A mixin class that implements the BFGS update"""
    def __init__(self):raise NotImplementedError
    



    


    
class Random:
    """A mixin class that implements random parameters"""
    
    def random_step_length(self) -> float:
        return 2*random() - 1
    
    def random_step_direction(self) -> Vector:
        return Vector([np.random.rand(self.domain_dimensions)])
    

        

    
        


    
    
    



    
class TrustRegionMethod(ABC):
    #slide 184
    pass

#class CauchyDirection(TrustRegionMethod):
#    pass

#class DoglegMethod(TrustRegionMethod):
    
class ConjugateGradient:...
    #slide 196+197
    #slide 200 -> three new methods 
    #slide 176 algorithm 4.1.2
    #slide 193
    


    

class DFP(QuasiNewton):...
class BFGS(QuasiNewton):...
class SR1(QuasiNewton):...
class L_BFGS(QuasiNewton):...




        




    
class TrustRegion:
    # ρ_k = predicted reduction = Aredk/Predk
    # Aredk = f(xk) − f(xk + sk) actual reduction
    # Predk = m(0) − m(s_k) predicted reduction
    #slide 318 algorithm 6.1.1
    ...
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    ...
    # from test_functions import *
    #todo unit tests
