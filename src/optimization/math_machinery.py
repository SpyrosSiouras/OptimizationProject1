from __future__ import annotations
from math import acos
import numpy as np
from numpy import array, double, ndarray, float64
from typing import Any, Callable, Optional
from numpy.typing import ArrayLike
from numbers import Number
from collections.abc import Iterable




__all__ = [
    "Function",
    "DifferentiableFunction",
    "DFunction",
    "TwiceDifferentiableFunction",
    "D2Function",
    "Vector"
]




class Vector:
    """A class modeling a point in R**n"""

    __slots__ = ("_coords", "T", "shape")

    def __init__(self, *coordinates: ArrayLike):
        if len(coordinates) == 1:
            self._coords = array(*coordinates, dtype=float64, ndmin=2).T
        else:
            self._coords = array(coordinates, dtype=float64, ndmin=2).T

        self.T = self._coords.T
        self.shape = self._coords.shape

    def __len__(self) -> int:
        return self._coords.shape[1]

    def __getitem__(self: Vector, index: int) -> double:
        try:
            return self._coords[0][index]
        except IndexError:
            raise IndexError("%s has %d coordinates" % (self,len(self)) ) from None

    @staticmethod
    def _validate_vector_input(vector_operation: Callable[[Vector, Vector], Any]):
        def operator(self: Vector, other: Vector) -> Any:
            try:
                return vector_operation(self, other)
            except AttributeError:
                raise TypeError("unsupported vector operation with '%s'" % type(other)) from None
            except ValueError:
                raise ValueError(
                        f"operands have different shapes: {self.shape}, {other.shape}"
                    ) from None
        return operator

    def __array__(self):
        return self._coords

    def __rmul__(self, num: Number) -> Vector:
        """
        Returns the scalar product num * Vector
        """
        if isinstance(num, Number):
            return Vector(num * self._coords)
        else:
            raise TypeError("unsupported operation with '%s'" % type(num)) from None

    @_validate_vector_input
    def __add__(self, other: Vector) -> Vector:
        return Vector(self._coords + other._coords)

    __radd__ = __add__

    @_validate_vector_input
    def __sub__(self, other: Vector) -> Vector:
        return Vector(self._coords - other._coords)
    __rsub__ = __sub__

    def __neg__(self: Vector) -> Vector:
        return Vector(-self._coords)

    @_validate_vector_input
    def __mul__(self: Vector, other: Vector) -> float64:
        """Returns the inner product of the vectors"""
        return (self._coords @ other._coords.T)[0][0]

    def squared(self: Vector) -> float64:
        return self * self

    def __abs__(self: Vector) -> float64:
        return self.squared()**.5

    def cos_angle(self: Vector, other: Vector) -> double:
        return self * other / abs(self) / abs(other)

    def angle(self: Vector, other: Vector) -> float:
        return acos( self.cos_angle(other) )

    def __repr__(self) -> str:
        return self.__class__.__name__ + self._coords.__repr__()[5:]





def mean_squared_difference(vector1: Vector, vector2: Vector) -> double:
    difference = vector1 - vector2
    return 1/len(vector1) * difference.squared()


class Point(Vector):
    """
    A class modeling a point in R**n
    
    Since a point P point P is essentially equivalent to the vector OP, 
    where O=(0,...,0), we merely aliasing the class Vector.
    """
    
    __slots__ = ("_coords", "T", "shape")



class DataSet:
    def __init__(self, vector: Vector):
        self.vector = vector
    def rescale(self): ...
    def mean(self) -> float:
        return 0
    @staticmethod
    def load_from(file) -> DataSet: 
        with open(file, "r") as f:
            return DataSet( Vector( [] ) )
    def normalize(self):
        #https://stats.stackexchange.com/questions/70553/what-does-normalization-mean-and-how-to-verify-that-a-sample-or-a-distribution
        pass
    
    
    




















class InvalidCallError(BaseException):
    def __init__(self, function: Function, *args, message: str="") -> None:
        m = function._argsnum
        err_msg = f"{function} takes {m} argument{'s' if m != 1 else ''}. It was given the arguments: {args}."
        self.message = err_msg + message
        super().__init__(self.message)





class Function:
    __slots__ = ("_argsnum", "_func", "_evaluations", "_identifier", "_formula")
    
    def __init__(
                    self,
                    pyfunction: Callable,
                    argsnum: int = None,
                    *,
                    identifier: Optional[str] = None,
                    formula_str: Optional[str] = None
                ) -> None:
        """_summary_

        Args:
            pyfunction (Callable): _description_
            grad_pyfunction (Callable, optional): _description_. Defaults to None.
            argsnum (int, optional): _description_. Defaults to None.
            identifier (Optional[str], optional): _description_. Defaults to None.
            formula_str (Optional[str], optional): _description_. Defaults to None.
            grad_formula_str (Optional[str], optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
        """
        self._func = pyfunction
        self._evaluations = 0
        
        self._set_attributes_or_copy_from_function(pyfunction, argsnum, identifier, formula_str)

        if self._argsnum is None:
            raise TypeError(f"{self.__class__.__name__}.__init__() missing 1 required positional argument: 'argsnum'")



    def _set_attributes_or_copy_from_function(
                                self,
                                function: Callable,
                                argsnum: int,
                                identifier: Optional[str],
                                formula_str: Optional[str]
                               ):
        """Sets the attributes to the non-None values provided and copies the rest from the given function, if they're available.'"""
        if argsnum is None and isinstance(function, Function):
            self._argsnum = function._argsnum
        else:
            self._argsnum = argsnum
        if identifier is None and isinstance(function, Function):
            self._identifier = function._identifier
        else:
            self._identifier = identifier
        if formula_str is None and isinstance(function, Function):
            self._formula = function._formula
        else:
            self._formula = formula_str

        if function.__doc__:
            self.__doc__ = function.__doc__

    def _apply_on_separate_arguments(self, *args: Any) -> Any:
        """Evaluates the function for separate arguments"""
        result = self._func(*args)
        self.evaluations += 1
        return result
    
    def __call__(self, *args: Any) -> Any:
        """
        Evaluates the function for the arguments provided and records that a evaluation was performed

        Arguments can be provided separately or in a container to be unpacked.

        The evaluation is recorded only if the call is successful.
        """

        try:
            # test if the arguments are provided separately
            return self._apply_on_separate_arguments(*args)
        except TypeError:
            pass
        try:
            # test if the arguments are provided in a container
            try:
                return self(*args[0])
            except IndexError:
                pass
        except TypeError:
            pass

        raise InvalidCallError(self, *args)

    @property
    def evaluations(self) -> int:
        return self._evaluations

    
    @evaluations.setter
    def evaluations(self, value) -> None:
        self._evaluations = value

    
    def __repr__(self) -> str:
        i = self._identifier
        f = self._formula
        left = "{ " if i or f else ""
        right = " }" if i or f else ""
        equals = " = " if i and f else ""
        info = f"""{left}{i or ""}{equals}{f or ""}{right}"""

        return f"{self.__class__.__name__}{info}" if info!="" else super().__repr__()





class DifferentiableFunction(Function):
    """ #todo!! A differential"""

    __slots__ = ("_argsnum", "_func", "_grad", "_evaluations", "_identifier", "_formula")

    def __init__(
                    self,
                    pyfunction: Callable,
                    grad_pyfunction: Callable = None,
                    argsnum: int = None,
                    *,
                    identifier: Optional[str] = None,
                    formula_str: Optional[str] = None,
                    grad_formula_str: Optional[str] = None
                ) -> None:
        """#todo!! """
        super().__init__(pyfunction, argsnum, identifier=identifier, formula_str=formula_str)
        gradient_identifier = f"D{self._identifier}" if self._identifier else None
        
        if isinstance(pyfunction, DifferentiableFunction):
            self._grad = pyfunction.gradient
            self._grad._identifier = gradient_identifier
        else:
            self._grad = Function(grad_pyfunction, self._argsnum, identifier=gradient_identifier, formula_str=grad_formula_str)
        
        
        if self.gradient is None:
            raise TypeError(f"{self.__class__.__name__}.__init__() missing 1 required positional argument: 'grad_pyfunction'")
        
            

    @property
    def gradient(self) -> Function:
        return self._grad
    
    derivative = gradient

    @property
    def evaluations(self) -> int:
        """
        Returns the number of times the function and its derivative have been evaluated
        
        The cost of the derivative is a multiple of the number of argument the function has.
        """
        return super().evaluations + self._argsnum * self.gradient.evaluations
    
    @evaluations.setter
    def evaluations(self, value) -> None:
        self._evaluations = value
        
    def __repr__(self) -> str:
        return f"{super().__repr__()}, \n{self.gradient.__repr__()}"

DFunction = DifferentiableFunction





class TwiceDifferentiableFunction(DifferentiableFunction):
    __slots__ = ("_argsnum", "_func", "_grad", "_hess", "_evaluations", "_identifier", "_formula")

    def __init__(
                    self,
                    pyfunction: Callable,
                    grad_pyfunction: Callable,
                    hess_pyfunction: Callable,
                    argsnum: int = None, 
                    *,
                    identifier: Optional[str] = None,
                    formula_str: Optional[str] = None,
                    grad_formula_str: Optional[str] = None,
                    hess_formula_str: Optional[str] = None
                ) -> None:  
        super().__init__(
            pyfunction,
            grad_pyfunction,
            argsnum,
            identifier=identifier,
            formula_str=formula_str,
            grad_formula_str=grad_formula_str
        )
        hessian_identifier = f"D**2{self._identifier}" if self._identifier else None
        self._hess = Function(
                                hess_pyfunction,
                                self._argsnum,
                                identifier=hessian_identifier,
                                formula_str=hess_formula_str
                            )
        self._grad = DifferentiableFunction(self.gradient, self.hessian)
        self.gradient.gradient._identifier = hessian_identifier
        
        
        

    @property
    def hessian(self) -> Function:
        return self._hess
    
    def derivative(self, degree=1) -> Function:
        if degree == 0:
            return self
        elif degree == 1:
            return self.gradient
        elif degree == 2:
            return self.hessian
        else:
            raise NotImplementedError()
    
    d = derivative
        
    @property
    def evaluations(self) -> int:
        """
        Returns the number of times the function, its gradient and its hessian have been evaluated
        
        The cost of the derivative is a multiple of the number of argument the function has.
        The cost of the hessian is a multiple of the square of the number of argument that the
        function has.
        """
        return super().evaluations + self._argsnum**2 * self.hessian.evaluations
    
    @evaluations.setter
    def evaluations(self, value) -> None:
        self._evaluations = value
    
D2Function = TwiceDifferentiableFunction























class Polynomial(TwiceDifferentiableFunction):
    def __init__(self, coefficients: ArrayLike) -> None:
        self.coefficients = array(coefficients)
        
    @classmethod
    def from_coefficients(cls, *list):
        return Polynomial(list)
    
    def __add__(self: Polynomial, other: Polynomial) -> Polynomial:
        try:
            sum = self.coefficients + other.coefficients
            return Polynomial(sum[sum!=0])
        except:
            raise
    
    def __mult__(self: Polynomial, other: Polynomial):
        ...
    
    def __get_item__(self: Polynomial, i: int) -> float:
        return self.coefficients[i]
    
    def degree(self: Polynomial) -> int:
        return len(self.coefficients)
    
    def derivative(self: Polynomial) -> Polynomial:
        return Polynomial([0])
    
    
    def __call__(self: Polynomial, *args: Any, **kwds: Any) -> Any:
        pass
  
    
    
    
    
    



if __name__ == "__main__":
    a = Vector(1.,2.,3,4,5,6,7,8,9,10)
    b = Vector(3,4,5,6,7,8,9,10,11,12)
    print(a, b, a+b, a-b, a*b, a@b.T, 15*a, a.angle(b) , sep="\n")
    
    c: Vector =a+b
    mean_squared_difference(c,b)
    
    print()
    print()
    
    f = Function(lambda t: t**2, 1, formula_str="t**2")
    print("f=t**2", "t**2(t=100) =",f(100))
    g = DifferentiableFunction(lambda t: t**2, lambda t: 2*t, 1)
    Dg = g.gradient
    print("t**2(t=100) =",g(100),"  Dt**2(t=100) =",Dg(100))
    
    
    f = lambda x,y: x**3 + y**3
    Df = lambda x,y: Vector(3*x**2,3*y**2)
    D2f = lambda x,y: array([[6*x,0],[0,6*y]]) 
    
    F = Function(f,2, identifier="F(x,y)", formula_str="x**3 + y**3")
    F1 = DFunction(F,Df, grad_formula_str="(3*x**2, 3*y**2)")
    F2 = D2Function(F,F1.gradient,D2f, hess_formula_str="[[6*x,0],[0,6*y]]")
    
    w = a.coords.T @ b.coords
    
