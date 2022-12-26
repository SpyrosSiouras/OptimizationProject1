from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
# from math_machinery import Function, Vector, DFunction, D2Function
from typing import Self, List
from collections.abc import Iterator
from copy import deepcopy

import numpy as np
from mathchinery import Point, Vector, Function, DiffFunction, Diff2Function
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

    def __init__(self, objective_function: Function) -> None:
        self.objective_function: Function = deepcopy(objective_function)
        self.budget: int = 0
        self.current: Point = Point( self.domain_dimensions * [0] )
        self.path: List[Point] = []

    @property
    def domain_dimensions(self) -> int:
        return self.objective_function.domain_dimensions

    @abstractmethod
    def step_length(self) -> float:
        """Computes the length of the current step"""
    @abstractmethod
    def direction(self) -> Vector:
        """Computes the direction of the current step"""

    @property
    def xk(self) -> Vector:
        return self.current

    @xk.setter
    def xk(self, value) -> None:
        self.current = value

    @property
    def xk_prev(self) -> Vector:
        try:
            return self.path[-2]
        except IndexError:
            raise ValueError("there is no previous point") from None

    @property
    def pk(self) -> Vector:
        return self.direction()

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

    def __next__(self) -> Point:
        """Gets the next point from the current point with a step"""
        self.xk += self.ak * self.pk
        self.path.append( self.xk )
        return self.xk

    def is_budget_exhausted(self) -> bool:
        return self.budget <= self.objective_function.evaluations

    @property
    def evaluations(self) -> int:
        return self.objective_function.evaluations

    @abstractmethod
    def are_stop_criteria_met(self) -> bool:
        """Check whether the stop criteria have been fulfilled"""
        return self.is_budget_exhausted()

    def run(self, from_point: Point, with_budget: int) -> Point:
        # slide 64, optimization theory and methods nonlinear programming
        #algorithm 1.5.1
        # step 0: (Initial step) Given initial point x0 ∈ Rn and the tolerance epsilon> 0.
        #Step 1. (Termination criterion) If ‖∇f (x_k)‖ ≤ epsilon, stop.
        #Step 2. (Finding the direction) According to some iterative scheme,
        #find dk which is a descent direction.
        #Step 3. (Line search) Determine the stepsize αk such that the objec-
        #tive function value decreases, i.e.,
        # f (x_k + α_k*d_k) < f (x_k).
        #Step 4. (Loop) Set x_k+1 = x_k + α_k*d_k, k := k + 1, and loop to Step 1.

        self.current = from_point
        self.path.append( self.current )
        self.budget = with_budget

        while not self.are_stop_criteria_met():
            next(self)
        return self.current

    def current_state_to_str(self) -> str:
        return f"The current point is xk={self.xk} with {self.objective_function>__name__}(xk)={self.fk} and a budget of {self.budget} function evaluations left to use."
    
    def report_warning(self, msg: str) -> None:
        warn( f"on xk={self.xk} {msg}", RuntimeWarning )
    
    
    
    
from mixins import LineSearch, SteepestDescentDirection
class SteepestDescent(SteepestDescentDirection, LineSearch, GenericAlgorithm):
    def __init__(self, objective_function: Diff2Function) -> None:
        super.__init__(objective_function)
        
    # def step_length(self) -> float:
    #     return self.line_
     
    
     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     

     


     
     

# class SteepestDescent(SteepestDescentMixin, LineSearchMixin, BaseAlgorithm): ...

# class SteepestDescentRandomStepLength(SteepestDescent, RandomStepLengthMixin, BaseAlgorithm): ...

class Newton():...

class TestAlgorithm(LineSearch, GenericAlgorithm):
    def step_length(self) -> float:
        return self.line_search_with_wolfe_conditions(
            a_max=10,
            interpolate=self._interpolate_with_bisection
        )
    def direction(self) -> Vector:
        return Vector( [1] + (self.domain_dimensions-1)*[0] )
    def are_stop_criteria_met(self) -> bool:
        return super().are_stop_criteria_met()

    
        
class AlwaysLeft(GenericAlgorithm):
    def step_length(self) -> float:
        return -2
    def direction(self) -> Vector:
        return Vector( [1] + (self.domain_dimensions-1)*[0] )
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
