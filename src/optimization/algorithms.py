from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from math_machinery import Function, Vector, DFunction, D2Function
from typing import Self, List
from collections.abc import Iterator

import numpy as np
from math_machinery import Point, Vector, Function
from random import random        


#### todo ADD
#### todo DOCs
#### todo EVERYWHERE!!!



minimal_req = """
1. BFGS με Wolfe conditions line search
2. Dogleg BFGS
3. Newton με Wolfe conditions line search
4. Steepest descent με Wolfe conditions line search
"""



class AbstractBaseAlgorithm(Iterator):
    """An abstract class for modeling a generic optimization algorithm"""
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

    def __init__(self, objective_function: Function) -> None:
        self.function: Function = objective_function
        self.current: Point = None
        self.budget: int = None
        
    # @property
    # def objective_function(self) -> Function:
    #     return self.
        
    # @property
    # @abstractmethod
    # def current(self) -> Point: ...
    
    # @property
    # @abstractmethod
    # def budget(self) -> int: ...
    
    @abstractmethod
    def are_stop_criteria_met(self) -> bool:
        """Check wether the stopping criteria have been fulfilled"""
        
    
    @abstractmethod
    def __repr__(self) -> str: ...
    
    # def __iter__(self):
    #     return self
    
    def __next__(self):
        """Gets the """
        self.current += self.step()
        return self.current
            
    def run(self, from_point: Point, with_budget: int):
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
        self.budget = with_budget

        path = []
        while not self.is_budget_exhausted():
            try:
                path.append( tuple(next(self)) )
                if self.are_stop_criteria_met():
                    break
            except KeyboardInterrupt:
                try:
                    if input(self.current_state_to_str() + " (Press c to continue...)")!="c":
                        continue
                except SyntaxError:
                    pass
                finally:
                    break
        return path
        
    def current_state_to_str(self):
        return f"current point={self.current}, with a budget of {self.budget} function evaluations left."
    
    def is_budget_exhausted(self):
        return self.budget <= self.function.evaluations
    
    # def current_objective_function_value(self):
    #     try:
    #         if self.current == self._current_func_val[0]:
    #             print("evals",self.objective_function.evaluations, self._current_func_val)
    #             return self._current_func_val[1]
    #     except AttributeError:
    #         pass
    #     x = self.current
    #     self._current_func_val = x, self.objective_function(x)
    #     print("evals",self.objective_function.evaluations, self._current_func_val)
    #     return self._current_func_val[1]
    
    
    @abstractmethod
    def step_length(self) -> float: ...
    @abstractmethod
    def direction(self) -> Vector:
        """Computes a descend direction"""
    def step(self) -> Vector:
        """Computes and returns a step to be use in the next iteration of the algorithm"""
        return self.step_length() * self.direction()
     
    

# mixins
# https://rednafi.github.io/digressions/python/2020/07/03/python-mixins.html

     
     
class GradientBasedMethodMixin:
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
    
    gradient = function.gradient
        
    def is_gradient_within_tolerance(self) -> bool:
        return abs(self.gradient(self.current)) < self.tolerance
    
    @property
    def tolerance(self): # <=10**-4
        return self._tolerance
    # slide 64, optimization theory and methods nonlinear programming
        #algorithm 1.5.1
        # step 0: (Initial step) Given initial point x0 ∈ Rn and the tolerance epsilon> 0.
        #Step 1. (Termination criterion) If ‖∇f (x_k)‖ ≤ epsilon, stop.
        #Step 2. (Finding the direction) According to some iterative scheme,
        #find dk which is a descent direction.
        #Step 3. (Line search) Determine the stepsize αk such that the objective function value decreases, i.e.,
        # f (x_k + α_k*d_k) < f (x_k).
        #Step 4. (Loop) Set x_k+1 = x_k + α_k*d_k, k := k + 1, and loop to Step 1.    
    
    @tolerance.setter
    def tolerance(self, value=10**-4):
        self._tolerance = value
    
    # 0<c1<c2<1
    # c1 = 10**-4
    # c2 = 0.9 quasiNewton
    # c2 = 0.1 conjugate gradient
    # slide 122
    # test other values for constants        
    def are_wolfe_conditions_met(self, cnd_step, c1, c2):
        f = self.objective_function
        gradf = self.gradient
        gradfx0T = gradf(x0).T
        x0 = self.current
        a = cnd_step
        p = self.direction()
        x1 = x0 + a * p
        armijo_condition = f(x1) < f(x0) + c1 * a * (gradfx0T * p)
        if not armijo_condition:
            return False
        curvature_condition = gradf(x1).T * p > c2 * (gradfx0T * p)
        
        return armijo_condition and curvature_condition
    
    ### TODO!!!!!!
    # 0<c1<c2<1
    # c1 = 10**-4
    # c2 = 0.9 quasiNewton
    # c2 = 0.1 conjugate gradient
    # slide 122
    # test other values for constants
    def are_strong_wolfe_conditions_met(self, cand_step, c1, c2):
        partial_result = self.wolfe_condition(cand_step, c1, c2)
        f = self.function
        gradf = self.gradient
        gradfx0T = gradf(x0).T
        x0 = self.current
        a = cand_step
        p = self.direction()
        x1 = x0 + a * p
        armijo_condition = f(x1) < f(x0) + c1 * a * (gradfx0T * p)
        if not armijo_condition:
            return False
        curvature_condition = abs(gradf(x1).T * p) < c2 * abs( (gradfx0T * p) )




class BFGS:...

class LineSearchMixin:...

class Doglet:...

class Newton:...

class SteepestDescent:...
    

        
class Test(AbstractBaseAlgorithm):
    def step_length(self) -> float:
        return -2
    def direction(self) -> Vector:
        return Vector(1,0)
    def are_stop_criteria_met(self) -> bool:
        self.objective_function(self.current)
        return super().are_stop_criteria_met()
    def __repr__(self) -> str:
        return f"test {self.current}"



f= Function(lambda s,t: s**2+t**2,argsnum=2)
g = Function(lambda v: v[0], argsnum=2)
print(g(Vector(0,0)))
t=Test(f)
t2 =Test(g)
O = Vector(0,0)
a = t.run(from_point=Vector(0,0), with_budget=10)
b = t2.run(from_point=Vector(0,0), with_budget=10)
print(a)
print(b)

class RandomMixin(ABC):
    @abstractmethod
    def random(self) -> float: ...
    
    
class RandomStepLengthMixin(RandomMixin):
    def step_length(self) -> float:
        return self.random()

class RandomDirectionMixin(RandomMixin):
    def direction(self) -> Vector:
        return Vector([random() for i in self.current])

        


exit()

    
    
    


class GradientBasedAlgorithm(BaseAlgorithm):
    def __init__(self, function, gradient, starting_point, compute_step_func) -> None:
        super().__init__(function, starting_point, compute_step_func)
        self.gradient = gradient
        self.neg_gradient = - gradient #objective_function_gradient
    
    # 0<c1<c2<1
    # c1 = 10**-4
    # c2 = 0.9 quasiNewton
    # c2 = 0.1 conjugate gradient
    # slide 122
    # test other values for constants        
    def wolfe_condition(self, cand_step, c1, c2):
        f = self.function
        gradf = self.gradient
        gradfx0T = gradf(x0).T
        x0 = self.current
        a = cand_step
        p = self.direction()
        x1 = x0 + a * p
        armijo_condition = f(x1) < f(x0) + c1 * a * (gradfx0T * p)
        if not armijo_condition:
            return False
        curvature_condition = gradf(x1).T * p > c2 * (gradfx0T * p)
        
        return armijo_condition and curvature_condition
    
    ### TODO!!!!!!
    # 0<c1<c2<1
    # c1 = 10**-4
    # c2 = 0.9 quasiNewton
    # c2 = 0.1 conjugate gradient
    # slide 122
    # test other values for constants
    def strong_wolfe_condition(self, cand_step, c1, c2):
        partial_result = self.wolfe_condition(cand_step, c1, c2)
        f = self.function
        gradf = self.gradient
        gradfx0T = gradf(x0).T
        x0 = self.current
        a = cand_step
        p = self.direction()
        x1 = x0 + a * p
        armijo_condition = f(x1) < f(x0) + c1 * a * (gradfx0T * p)
        if not armijo_condition:
            return False
        curvature_condition = abs(gradf(x1).T * p) < c2 * abs( (gradfx0T * p) )
    

class LineSearchMethod(ABC):
    #slide 109
    __slots__ = ()
    def __init__(self, f, p_k, x_k):
        pass
    def get_next_step(self):
        return a
        
        #slide 115
    def bisect(): ...       
    def backtracking():...
        #slide 118
    def bracketing(): ...
    def wolfe_condition():...
    def armijo_condition():...
    def curviture_condition():...
    def genikosxhma():... #?????
        #slide 135
    def interpolation(): ... #slide 145
    #bisection
    #quadratic
    #cubic
    #slide 147
    #slide 149
    
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
    
class SteepestDescent(GradientBasedAlgorithm):
    def direction(self):
        x = self.current
        return self.neg_gradient(x)
    

class Newton(GradientBasedAlgorithm):
    def __init__(self, function, gradient, hessian, starting_point, compute_step_func) -> None:
        super().__init__(function, gradient, hessian, starting_point, compute_step_func)
        self.hessian = hessian
    #slide  151 arxiko bhma 1 ????
    
    def direction(self):
        x = self.current
        # todo if self.hessian is positive
        return np.inverse (self.hessian(x)) * self.neg_gradient(x)
        # todo todo todo





class QuasiNewton(GradientBasedAlgorithm):
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



class LineSearch(GradientBasedAlgorithm):
    def __init__(self, function, gradient, positive_matrix, starting_point, compute_step_func) -> None:
        super().__init__(function, gradient, starting_point, compute_step_func)
        self.positive = positive_matrix # positive definite symmetric matrix
        



class BFGS(BaseAlgorithm):
    def step(self):
        return 1
    def direction(self):
        return 1
# a=BFGS(lambda x: x, (0,0))
# print(a)
# print(next(a))
# print(BFGS.__annotations__, BFGS.__dict__, a.__dict__)




class Doglet:
    ...


class WolfeConditions:
    ...
    

    
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
