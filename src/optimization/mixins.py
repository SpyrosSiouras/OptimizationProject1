from __future__ import annotations
from abc import ABC, abstractmethod

from mathchinery import Function, Vector, DiffFunction, Diff2Function
from typing import Self, List, Optional, Callable, Tuple
from collections.abc import Iterator

import numpy as np
from numpy import ndarray, eye
from mathchinery import is_almost_zero, is_close, is_positive, multiplication_with_inverse, Point, Vector, Function
from mathchinery import FunctionNotDifferentiableError, FunctionNotTwiceDifferentiableError, LinAlgError
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
    """A mixin class implementing functionality for methods that use gradients"""

    _DEFAULT_GRADIENT_TOLERANCE = 10**-4

    @property
    def gradient_tolerance(self) -> float:
        try:
            return self._gradient_tolerance
        except AttributeError:
            self._gradient_tolerance = self._DEFAULT_GRADIENT_TOLERANCE
            return self._gradient_tolerance

    @gradient_tolerance.setter
    def gradient_tolerance(self, new_tolerance: float) -> None:
        self._gradient_tolerance = new_tolerance

    @property
    def gradient(self) -> Function:
        """Returns the gradient of the objective function"""
        try:
            return self.objective_function.gradient
        except AttributeError:
            pass
        raise FunctionNotDifferentiableError(self.objective_function)

    gradf = gradient

    @property
    def gradfk(self) -> Vector:
        """Returns the gradient at the current point xk"""
        return self.gradf(self.xk)

    def is_descent_direction(self, vector: Vector) -> float:
        """Is the vector a descent direction?"""
        return vector * self.gradfk < 0

    def is_gradient_almost_zero(self) -> bool:
        """Is the gradient close enough to zero?"""
        return is_almost_zero( abs(self.gradfk), tolerance=self.gradient_tolerance )

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


    def phi(self, a: float) -> float:
        """Returns the value objective_function(xk + a*pk)"""
        return self.f( self.xk + a * self.pk )

    @property
    def phi0(self) -> float:
        """Returns the value objective_function(xk)"""
        return self.fk

    def Dphi(self, a: float) -> float:
        """Returns the value objective_function.gradient(xk + a*pk)"""
        return self.pk * self.gradf( self.xk + a * self.pk )

    @property
    def Dphi0(self) -> float:
        """Returns the value objective_function.gradient(xk)"""
        return self.pk * self.gradfk

    def is_armijo_met(self, a: float, c1: float = _DEFAULT_c1) -> bool:
        """Does the Armijo condition hold for the objective_function on xk + a*pk?"""
        return self.phi(a)  <=  self.phi0 + c1 * a * self.Dphi0

    def is_strong_curvature_met(self, a: float, c2: float) -> bool:
        """Does the strong curvature condition hold for the objective_function on xk + a*pk?"""
        return abs(self.Dphi(a))  <=  - c2 * self.Dphi0

    def report_line_search_failure(self, method_name: str, reason: Optional[str] = None ) -> None:
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
        if not self.is_descent_direction(self.pk):
            raise ValueError(f"the direction pk={self.pk} at the current point xk={self.xk} is not a descent direction") from None

        a_cnd = a_initial
        for i in range(max_iterations):
            if self.is_armijo_met(a_cnd, c1):
                return a_cnd
            a_cnd *= 0.5
        self.report_line_search_failure( method_name = "backtracking line search with the Armijo condition" )
        return a_cnd

    def linear_interpolation(self, a1: float, a2: float) -> float:
        """Returns the average value of a1, a2"""
        return (a1+a2)/2

    def quadratic_interpolation(self, a1: float, a2: float) -> float:
        """
        Returns the minimizer/maximizer of the quadratic polynomial p(t) := x*(t-a1)**2 + y*(t-a1) + z, where
            x := ( fa2 - fa1 - Dfa1*(a2-a1) )/(a2-a1)**2,    y:= Dfa1,    z:= fa1
        These values are picked so that p(a1)==fa1, p(a2)==fa2 and Dp(a1)==Dfa1
        """
        fa1 = self.phi(a1)
        fa2 = self.phi(a2)
        Dfa1 = self.Dphi(a1)

        x = ( fa2 - fa1 - Dfa1*(a2-a1) )/(a2-a1)**2
        y = Dfa1
        return -y/(2*x)

    def interpolation(self, a1: float, a2: float) -> float:
        """
        Returns the maximum step between the quadratic and the linear interpolation
        """
        a_q = self.quadratic_interpolation(a1,a2)
        a_l = self.linear_interpolation(a1,a2)
        return max( a_q, a_l )

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

        for _ in range(max_iterations):
            a = self.interpolation(a_lowest_armijo, a_other)
            if is_close(a_lowest_armijo, a_other):
                return a

            if not self.is_armijo_met(a, c1) or self.phi(a) >= self.phi(a_lowest_armijo):
                a_other = a
            else:
                if self.is_strong_curvature_met(a, c2):
                    return a
                if self.Dphi(a) * (a_other - a_lowest_armijo) >= 0:
                    a_other = a_lowest_armijo
                a_lowest_armijo = a

        return a



    def line_search_with_wolfe_conditions(
                                            self,
                                            a_max: float,
                                            a_initial: float = None,
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

            a_initial = 1 when the method is Newton or Quasi-Newton
        """
        if not self.is_descent_direction(self.pk):
            raise ValueError(f"the direction pk={self.pk} at the current point xk={self.xk} is not a descent direction") from None

        a_prev = 0
        a = a_initial

        if a is None:
            a = self.linear_interpolation(a_prev, a_max)

        for _ in range(max_iterations):
            if not self.is_armijo_met(a, c1) or self.phi(a)>=self.phi(a_prev):
                return self._zoom(a_prev, a, c1, c2, max_iterations)
            if self.is_strong_curvature_met(a, c2):
                return a
            if self.Dphi(a) >= 0:
                return self._zoom(a, a_prev, c1, c2, max_iterations)

            a_prev = a
            a = self.linear_interpolation(a_prev, a_max)

        return a



class SteepestDescentDirection(GradientBasedMethod):
    """A mixin class that implements the steepest descent direction"""

    def steepest_descent_direction(self, p: Vector) -> Vector:
        """Returns the direction of the steepest descent"""
        return -self.gradf(p)







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
    
    def quadratic_model(self, p: Vector) -> float:
        """Evaluates the quadratic model on xk"""
        return  self.fk  +  self.gradfk * p  +  1/2 * (p @ self.Bk @ p.T) [0]
    
    mk = quadratic_model

   







class NewtonDirection(QuadraticModel):
    """A mixin class that implements the Newton direction"""

    @property
    def hessian(self) -> Function:
        """Returns a function that computes the hessian of the objective function"""
        try:
            return self.objective_function.hessian
        except AttributeError:
            raise FunctionNotTwiceDifferentiableError from None

    hessian_approximation = hessian

    @property
    def hessfk(self) -> ndarray:
        """Returns the hessian at the current point xk"""
        return self.hessian(self.xk)

    def newton_direction_unsafe(self, raise_when_not_positive=False) -> Vector:
        """
        Returns the Newton direction at the current point xk

        If the hessian is not positive or it is singular and the flag is set,
        it raises a LinAlgError.
        """
        hessfk_is_assumed_positive = raise_when_not_positive
        return -self.gradfk.multiplied_with_inverse(
            self.hessfk,
            assume_a = "pos" if hessfk_is_assumed_positive else "sym"
        ) # this call fails when the hessian is not positive or it is singular

    def multiplication_with_inverse_modified_hessian(self, matrix: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Computes matrix @ inverse(hessfk)

        As a side effect, it also computes and returns the modified hessian
        """
        hessfk_is_assumed_positive = True
        m = min(self.hessfk.diagonal())
        I = eye(self.domain_dimensions)
        b = 10**-3

        if m > 0:
            t = 0
        else:
            t = -m + b

        while hessfk_is_assumed_positive:
            try:
                result = multiplication_with_inverse(
                    matrix,
                    self.hessfk + t*I,
                    assume_a = "pos"
                ) # this call fails when the hessian is not positive or it is singular
                return result, self.hessfk + t*I
            except LinAlgError:
                t = max(10*t, b)

    def newton_direction_with_hessian_modification(self) -> Vector:
        """
        Returns the Newton direction biased toward a direction that makes the hessian positive
        """
        arr = self.multiplication_with_inverse_modified_hessian(-self.gradfk)[0]
        return Vector(arr)

    @property
    def modified_hessian(self) -> ndarray:
        """Returns the modified hessian and its inverse"""
        return self.multiplication_with_inverse_modified_hessian(-self.gradfk)[1]








class TrustRegionMethod(QuadraticModel):
    """A mixin class implementing functionality for trust region methods"""

    @property
    def hessian_approximation(self) -> Function:
        return self._Bk

    @hessian_approximation.setter
    def hessian_approximation(self, new_matrix: ndarray) -> Function:
        self._Bk = new_matrix

    def step_length(self):
        return 1

    @property
    def radius(self) -> float:
        try:
            return self._radius
        except AttributeError:
            self._radius = self.max_radius
            return self._radius

    @radius.setter
    def radius(self, number: float) -> None:
        self._radius = number

    _DEFAULT_MAX_RADIUS = 1

    @property
    def max_radius(self) -> float:
        try:
            return self._max_radius
        except AttributeError:
            self._max_radius = self._DEFAULT_MAX_RADIUS
            return self._max_radius

    @max_radius.setter
    def max_radius(self, new_max_radius: float) -> None:
        self._max_radius = new_max_radius

    def is_inside_region(self, direction: Vector) -> bool:
        return abs(direction) <= self.radius

    def improvement_rate(self, direction: Vector) -> float:
        """Returns the """
        return ( self.fk - self.f(self.xk + direction) ) / ( self.fk - self.mk(direction) )
        #todo make this numerically stable

    _DEFAULT_MIN_IMPROVEMENT_RATE = 0.2

    @property
    def min_improvement_rate(self) -> float:
        try:
            return self._min_improvement_rate
        except AttributeError:
            self._min_improvement_rate = self._DEFAULT_MIN_IMPROVEMENT_RATE
            return self._min_improvement_rate

    @min_improvement_rate.setter
    def min_improvement_rate(self, new_min_improvement_rate: float) -> None:
        self._min_improvement_rate = new_min_improvement_rate

    def update_radius(self, direction: Vector) -> None:
        if self.improvement_rate(direction) < 0.25:
            self.radius *= 0.25
        elif self.improvement_rate(direction) > 0.75 and is_close(abs(direction), self.radius):
            self.radius = min(2*self.radius, self.max_radius)

    def trustregion_step(self, direction: Vector) -> Vector:
        if self.improvement_rate(direction) > self.min_improvement_rate:
            return direction
        else:
            return self.domain_zero

class CauchyDirection(TrustRegionMethod):
    def cauchy_point(self) -> Vector:
        """Computes the Cauchy point"""
        gBgT = (self.gradfk @ self.Bk @ self.gradfk.T)[0]
        absgradfk = abs(self.gradfk)
        if gBgT <= 0:
            tau = 1
        else:
            tau = min(1, absgradfk**3/(self.radius*gBgT) )
        return -(tau*self.radius)/absgradfk * self.gradfk

class DoglegMethod(TrustRegionMethod):
    def dogleg_point(self) -> Optional[Vector]:
        """
        Computes the Dogleg point

        If Bk is not positive, it returns None
        """

        try:
            pB = -self.gradfk.multiplied_with_inverse(
                self.Bk,
                assume_a="pos"
            )
        except LinAlgError:
            return None

        if self.is_inside_region(pB):
            return pB

        gBgT = (self.gradfk @ self.Bk @ self.gradfk.T)[0]
        pU = - self.gradfk.squared()/gBgT * self.gradfk

        if not self.is_inside_region(pU):
            return -self.radius/abs(self.gradfk) * self.gradfk

        R2 = self.radius**2
        pUsq = pU.squared()
        pB_pU_sq = (pB-pU).squared()
        pU_o_pB_pU_o_2 = pU * (pB-pU) * 2

        func: Callable[[float], float] = lambda t: pUsq + (t-1)**2*pB_pU_sq + (t-1)*pU_o_pB_pU_o_2 - R2

        tau = self.bisection_method(func, 1, 2)

        return pU + (tau-1)*(pB-pU)

    def bisection_method(
                            self,
                            function: Callable[[float], float],
                            left: float,
                            right: float,
                            tolerance: float = 10**-6,
                            max_iterations: int = 20
                        ) -> float:
        """
        Finds a root of a function in (left, right)

        It assumes that function(left) * function(right) < 0 and left < right.
        """
        if is_positive( function(left) ):
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

            if is_positive(fm):
                pos = middle
            else:
                neg = middle

        return middle


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
