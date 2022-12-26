import numpy as np
import math

class Function:
    def __init__(self, body):
        self.__body = body

    def __call__(self, x):
        return self.__body(x)

    def __add__(self, other):
        return Sum(self, other)
    
class DifferentiableFunction(Function):
    def __init__(self, body, derivative_body):
        super().__init__(body)
        self.__derivative_body = Function(derivative_body)
    def derivative(self):
        return self.__derivative_body

class Constant(DifferentiableFunction):
    def __init__(self, c):
        super().__init__(lambda x: c, lambda x: 0)

class Polynomial(DifferentiableFunction):
    def __init__(self, coefficients):
        ... #todo
        super().__init__(body, derivative_body)

sin = DifferentiableFunction( math.sin, math.cos )
exp = DifferentiableFunction( math.exp, math.exp )

# D(sin.o(exp) + exp)
    

if __name__=="__main__":
    f = Function( exp )
    g = Function( sin )
    print( f(10), g(10), sin.derivative(math.pi) )
    

class Sum(Function):
    def __init__(self, add1, add2):
        self.add1 = add1
        self.add2 = add2
        super()

class Composition(Function):
    ...


# class DifferentiableFunction(Function):
#     ...    

class PositiveDefinite:
    ...

class Methods:
    ...

class LineSearch:
    ...

class TrustRegion:
    ...

class WolfeConditions:
    ...

class SteepestDescent:
    ...

class NewtonMethod:
    ...

class QuasiNewton:
    ...

class ArmijoCondition:
    ...

class Backtracking:
    ...

class CurvitureCondition:
    ...

class StrongWolfeConditions:
    ...

class CauchyDirection:
    ...

class Dogleg:
    ...

class ConjugateDirection:
    ...

class TestFunction:
    ...
