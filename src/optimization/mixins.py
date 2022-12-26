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
from mathchinery import Point, Vector, Function
from mathchinery import FunctionNotDifferentiableError, FunctionNotTwiceDifferentiableError
from random import random        


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
    _DEFAULT_GRADIENT_TOLERANCE = 10**-4

    @property
    def gradient(self: GenericAlgorithm) -> Function:
        """Returns the gradient of the objective function"""
        try:
            return self.objective_function.gradient
        except AttributeError:
            pass
        raise FunctionNotDifferentiableError(self.objective_function)

    @property
    def gradf(self: GenericAlgorithm) -> Function:
        return self.gradient

    @property
    def gradfk(self: GenericAlgorithm) -> Vector:
        return self.gradf(self.xk)

    @property
    def pkgradfk(self: GenericAlgorithm) -> float:
        pkgradfk = self.pk * self.gradfk
        if pkgradfk >= 0:
            self.report_warning(
                f"the provided vector pk={self.pk} is not a descent direction since pk * gradf(xk) == {pkgradfk} >= 0",
            )
        return pkgradfk

    def is_gradient_almost_zero(self: GenericAlgorithm, tolerance: float = _DEFAULT_GRADIENT_TOLERANCE) -> bool:
        return abs(self.gradfk) < tolerance







    #https://github.com/gjkennedy/ae6310
    #https://github.com/gjkennedy/ae6310/blob/master/Line%20Search%20Algorithms.ipynb
    #https://indrag49.github.io/Numerical-Optimization/line-search-descent-methods.html
class LineSearch(GradientBasedMethod):
    _DEFAULT_c1 = 10**-4
    _DEFAULT_c2 = 0.9
    _DEFAULT_MAX_ITERATIONS_IN_LINE_SEARCH = 10

    def phi(self: GenericAlgorithm, a: float) -> float:
        return self.f( self.xk + a * self.pk )

    @property
    def phi0(self: GenericAlgorithm) -> float:
        return self.fk

    def dphi(self: GenericAlgorithm, a: float) -> float:
        return self.pk * self.gradf( self.xk + a * self.pk )

    @property
    def dphi0(self: GenericAlgorithm) -> float:
        return self.pkgradfk

    def is_armijo_met(self: GenericAlgorithm, a: float, c1: float = _DEFAULT_c1):
        return self.phi(a)  <=  self.phi0 + c1 * a * self.dphi0

    def is_strong_curvature_met(self: GenericAlgorithm, a: float, c2: float):
        return abs(self.dphi(a))  <=  - c2 * self.pkgradfk

    def report_line_search_failure(self: GenericAlgorithm, method_name: str, reason: Optional[str] = None ) -> None:
        self.report_warning(
            f"the {method_name} failed after {self._DEFAULT_MAX_ITERATIONS_IN_LINE_SEARCH} iterations {'' if reason is None else reason }"
        )

    def backtrack_line_search_with_armijo(
                                            self: GenericAlgorithm,
                                            a_initial: float = 1,
                                            c1: float = _DEFAULT_c1,
                                            max_iterations: int = _DEFAULT_MAX_ITERATIONS_IN_LINE_SEARCH
                                          ) -> float:
        a_cnd = a_initial
        for i in range(max_iterations):
            if self.is_armijo_met(a_cnd, c1):
                return a_cnd
            a_cnd *= 0.5
        self.report_line_search_failure( method_name = "backtracking line search with the Armijo condition" )
        return a_cnd

    def _interpolate_with_bisection(self, a: float, b: float) -> float:
        return (a+b)/2
    
    def _interpolate_quadratically(self, a: float, b: float) -> float:
        return 
        
    def _interpolate_cubically(self, a: float, b: float) -> float:
        return

    def _zoom(
                self: GenericAlgorithm,
                a_lowest_armijo: float,
                a_other: float,
                interpolate: Callable[[float,float], float],
                c1: float = _DEFAULT_c1,
                c2: float = _DEFAULT_c2,
                max_iterations: int = _DEFAULT_MAX_ITERATIONS_IN_LINE_SEARCH
             ) -> float:
        method_name = "zoom phase of the line search with Wolfe conditions"
        start = (a_lowest_armijo, a_other)

        for i in range(max_iterations):
            a = interpolate( min(a_lowest_armijo, a_other), max(a_lowest_armijo, a_other) )

            if not self.is_armijo_met(a, c1) or self.phi(a) >= self.phi(a_lowest_armijo):
                a_other = a
            else:
                if self.is_strong_curvature_met(a, c2):
                    return a
                if self.dphi(a) * (a_other - a_lowest_armijo) >= 0:
                    a_other = a_lowest_armijo
                a_lowest_armijo = a

        self.report_line_search_failure(method_name, reason=f"for {start=}. Returning a_lowest_armijo")
        return a_lowest_armijo



    def line_search_with_wolfe_conditions(
                                            self: GenericAlgorithm,
                                            interpolate: Callable[[float,float], float],
                                            a_max: float,
                                            c1: float = _DEFAULT_c1,
                                            c2: float = _DEFAULT_c2,
                                            max_iterations: int = _DEFAULT_MAX_ITERATIONS_IN_LINE_SEARCH
                                          ) -> float:
        # 0<c1<c2<1
        # c1 = 10**-4
        # c2 = 0.9 quasiNewton
        # c2 = 0.1 conjugate gradient
        a = 1
        a_prev = 0
        method_name = "line search with Wolfe conditions"
        a_max = max(1, a_max)

        for i in range(max_iterations):
            if a > a_max:
                self.report_line_search_failure( method_name, "as a exceeded a_max. Execute backtracking line search with Armijo condition" )
                return self.backtrack_line_search_with_armijo(a, c1, max_iterations)

            if not self.is_armijo_met(a, c1) or self.phi(a)>=self.phi(a_prev):
                return self._zoom(a_prev, a, interpolate, c1, c2, max_iterations)
            if self.is_strong_curvature_met(a, c2):
                return a
            if self.dphi(a) >= 0:
                return self._zoom(a, a_prev, interpolate, c1, c2, max_iterations)

            a_prev = a
            a = self._interpolate( min(a, a_max), max(a, a_max) )


        self.report_line_search_failure( method_name, "as a exceeded a_max" )
        return self.backtrack_line_search_with_armijo(a, c1, max)
    



class SteepestDescentDirection(GradientBasedMethod):
    def steepest_descent_direction(self: GenericAlgorithm) -> Vector:
        return - self.gradient(self.xk)









        
        
    
    
    
        
    







class SecondOrderApproximation(GradientBasedMethod, ABC):
    
    @property
    @abstractmethod
    def hessian_approximation(self):
        return self.function.hessian


        
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class Dogleg:...


class NewtonDirection(GradientBasedMethod):
    def direction(self) -> Vector:
        x = self.current
        return self.inverse_hessian() @ self.negative_gradient


    


class RandomMixin(ABC):
    @abstractmethod
    def random(self) -> float: ...
    
class RandomStepLength:
    def step_length(self) -> Vector:
        return 2*random() - 1
class RandomDirection:
    def direction(self) -> Vector:
        return Vector([np.random.rand(self.dimension)])
    

        

    
        


    
    
    



    
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
    






class QuasiNewton(GradientBasedMethod):
    def __init__(self, function, gradient, positive_matrix, starting_point, compute_step_func) -> None:
        super().__init__(function, gradient, starting_point, compute_step_func)
        self.positive = positive_matrix # positive definite symmetric matrix
    
    def update_matrix(self): #### ????
        # slide 134
        pass
    

class DFP(QuasiNewton):...
class BFGS(QuasiNewton):...
class SR1(QuasiNewton):...
class L_BFGS(QuasiNewton):...




        



class BFGS():
    def step(self):
        return 1
    def direction(self):
        return 1
# a=BFGS(lambda x: x, (0,0))
# print(a)
# print(next(a))
# print(BFGS.__annotations__, BFGS.__dict__, a.__dict__)



    
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
