from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
# from math_machinery import Function, Vector, DFunction, D2Function
from typing import Self, List, Optional
from collections.abc import Iterator
from copy import deepcopy

import numpy as np
from scipy.linalg import LinAlgError 
from mathchinery import is_almost_zero, Point, Vector, Function, DiffFunction, Diff2Function, is_close,zeros
from random import random        
from functools import cached_property
from warnings import warn, simplefilter

simplefilter("always", RuntimeWarning)




#### todo ADD
#### todo DOCs
#### todo EVERYWHERE!!!



minimal_req = """
1. BFGS με Wolfe conditions line search
2. Dogleg BFGS
3. Newton με Wolfe conditions line search
4. Steepest descent με Wolfe conditions line search
"""


class GenericAlgorithm(Iterator):
    """An abstract class for modeling an optimization algorithm"""

    def __init__(self, objective_function: Function, show_warnings: bool = False) -> None:
        if not isinstance(objective_function, Function):
            raise TypeError("the objective function provided must be an instance of Function")
        self.objective_function: Function = deepcopy(objective_function)
        self.budget: int = 0
        self.current: Point = self.domain_zero
        self.path: List[Point] = []
        self.show_warnings = show_warnings

    @property
    def domain_dimensions(self) -> int:
        return self.objective_function.domain_dimensions

    @property
    def domain_zero(self) -> Vector:
        return Vector.zero(self.domain_dimensions)

    @abstractmethod
    def step_length(self) -> float:
        """Computes the length of the current step"""
    @abstractmethod
    def step_direction(self) -> Vector:
        """Computes the direction of the current step"""

    @property
    def xk(self) -> Vector:
        return self.current

    @xk.setter
    def xk(self, point: Vector) -> None:
        self.current = point

    @property
    def xk_prev(self) -> Vector:
        try:
            return self.path[-2]
        except IndexError:
            raise ValueError("there is no previous point") from None

    @property
    def pk(self) -> Vector:
        return self.step_direction()

    @property
    def ak(self) -> float:
        return self.step_length()

    @property
    def f(self) -> Function:
        return self.objective_function

    @property
    def fk(self) -> float:
        return self.f(self.xk)

    @property
    def fk_prev(self) -> float:
        return self.f(self.xk_prev)

    def perform_one_step(self) -> Point:
        """Computes the step length, the step direction and moves to the next point"""
        self.xk += self.ak * self.pk
        self.path.append( self.xk )
        return self.xk

    __next__ = perform_one_step

    def is_budget_exhausted(self) -> bool:
        return self.budget <= self.objective_function.evaluations

    @property
    def evaluations(self) -> int:
        return self.objective_function.evaluations

    @property
    def unweighted_evaluations(self) -> int:
        return self.objective_function.unweighted_evaluations

    @property
    def total_evaluations(self) -> int:
        return self.evaluations

    @property
    def total_unweighted_evaluations(self) -> int:
        return self.unweighted_evaluations

    @abstractmethod
    def are_stop_criteria_met(self) -> bool:
        """Check whether the stop criteria have been fulfilled"""
        return self.is_budget_exhausted()

    def resume_run(self, with_additional_budget: int = 0) -> Point:
        self.budget += with_additional_budget
        while not self.are_stop_criteria_met():
            self.perform_one_step()
        return self.current

    def run(self, *from_point: float | Point, with_budget: int = 10**9) -> Point:
        if len(from_point) == 1 and isinstance(from_point[0], Vector):
            point = from_point[0]
        else:
            point = Point(from_point)

        if len(point) != self.domain_dimensions:
            raise ValueError(f"the initial point {point} does not belong in the search domain as its dimension is not {self.domain_dimensions}")

        self.current = point
        self.path = [self.current]
        self.objective_function._reset()
        return self.resume_run(with_budget)

    @property
    def length_of_last_run(self):
        return len(self.path)

    def report_current_state(self) -> None:
        print(f"""\
The current point is xk={self.xk} with {self.objective_function}(xk)={self.fk} and a budget of {self.budget-self.total_evaluations} function evaluations left to use. \
The current step length is ak={self.ak} and the current step direction is pk={self.pk}. """)

    def report_warning(self, msg: str) -> None:
        if self.show_warnings:
            warn( f"on xk={self.xk} {msg}", RuntimeWarning )





from mixins import LineSearch, SteepestDescentDirection, NewtonDirection, BFGSUpdate, TrustRegionMethod, DoglegMethod
class SteepestDescent(SteepestDescentDirection, LineSearch, GenericAlgorithm):
    def __init__(self, objective_function: DiffFunction, gradient_tolerance: Optional[float] = None) -> None:
        super().__init__(objective_function)
        if gradient_tolerance:
            self.gradient_tolerance = gradient_tolerance

    def step_direction(self) -> Vector:
        return self.steepest_descent_direction(self.xk)

    def step_length(self) -> float:
        return self.line_search_with_wolfe_conditions(a_max=1)

    def are_stop_criteria_met(self) -> bool:
        return self.is_budget_exhausted() or self.is_gradient_almost_zero()


class Newton(NewtonDirection, LineSearch, GenericAlgorithm):
    """
    The Newton algorithm for finding minima

    No check is performed whether the hessian is positive or even reversible
    """

    def __init__(self, objective_function: Diff2Function, gradient_tolerance: Optional[float] = None) -> None:
        super().__init__(objective_function)
        if gradient_tolerance:
            self.gradient_tolerance = gradient_tolerance

    def step_direction(self) -> Vector:
        return self.newton_direction_unsafe()

    def step_length(self) -> float:
        return self.line_search_with_wolfe_conditions(a_initial=1,a_max=2)

    def are_stop_criteria_met(self) -> bool:
        return self.is_budget_exhausted() or self.is_gradient_almost_zero()

class NewtonWithHessianModification(Newton):
    """
    The Newton algorithm for finding minima

    The hessian is used whenever it is positive, otherwise the positive modified hessian is used

    The rate of convergence is lower compared to the unsafe variant.
    """

    def step_direction(self) -> Vector:
        try:
            return self.newton_direction_unsafe(raise_when_not_positive=True)
        except LinAlgError:
            return self.newton_direction_with_hessian_modification()


class NewtonDogleg(NewtonDirection, DoglegMethod, GenericAlgorithm):
    @property
    def Bk(self) -> Function:
        return self.modified_hessian

    def step_direction(self) -> Vector:
        catleg = self.dogleg_point()
        self.update_radius(catleg)
        return self.trustregion_step(catleg)
    
    def are_stop_criteria_met(self) -> bool:
        return self.is_budget_exhausted() or self.is_gradient_almost_zero()



class BFGSWithLineSearch(BFGSUpdate, LineSearch, GenericAlgorithm):
    ...


class BFGSWithLineSearch(BFGSUpdate, TrustRegionMethod, GenericAlgorithm):
    ...

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     


     
     


class TestAlgorithm(LineSearch, GenericAlgorithm):
    def step_length(self) -> float:
        return self.line_search_with_wolfe_conditions(
            a_max=10,
            interpolate=self._interpolate_with_bisection
        )
    def direction(self) -> Vector:
        return Vector( [1] + zeros(self.domain_dimensions-1) )
    def are_stop_criteria_met(self) -> bool:
        return super().are_stop_criteria_met()

    
        
class AlwaysLeft(GenericAlgorithm):
    def step_length(self) -> float:
        return -2
    def direction(self) -> Vector:
        return Vector( [1] + zeros(self.domain_dimensions-1) )
    def are_stop_criteria_met(self) -> bool:
        self.objective_function(self.current)
        return super().are_stop_criteria_met()
    def __repr__(self) -> str:
        return f"test {self.current}"



# f= Function(lambda s,t: s**2+t**2,argsnum=2)
# g = Function(lambda v: v[0], argsnum=2)
# print(g(Vector(0,0)))
# t=AlwaysLeft(f)
# t2 =AlwaysLeft(g)
# O = Vector(0,0)
# a = t.run(from_point=Vector(0,0), with_budget=10)
# b = t2.run(from_point=Vector(0,0), with_budget=10)
# print(a)
# print(b)

    
    
    
    
if __name__ == "__main__":
    ...
    # from test_functions import *
    #todo unit tests
