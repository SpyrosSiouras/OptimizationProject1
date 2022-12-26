"""The mathematical machinery required"""

from __future__ import annotations

from math import acos
import numpy as np
from numpy import array, ndarray, float64
from numpy.linalg import inv
from typing import Any, Callable, Optional, Tuple, List
from numpy.typing import ArrayLike
from numbers import Number
from collections.abc import Iterable, Collection
from functools import lru_cache, cache, reduce, wraps, cached_property
from collections import deque
from inspect import signature
from copy import deepcopy
from abc import ABC, abstractmethod

try:
    from typing import Unpack
except ImportError:
    pass





__all__ = [
    "Vector",
    "Function",
    "DiffFunction",
    "Diff2Function",
]




class Vector:
    """
    A class modeling a vector in R**n

    The vectors are represented as rows, contrary to the common mathematical
    convention of treating vectors as columns.
    """

    __slots__ = ("_coords",)

    def __init__(self, *coordinates: ArrayLike):
        """
        The coordinates can be given in a container or separately one by one
        """
        if len(coordinates) == 1:
            self._coords = array(*coordinates, dtype=float64, ndmin=2)
        else:
            self._coords = array(coordinates, dtype=float64, ndmin=2)

    @property
    def T(self):
        return self._coords.T

    @property
    def shape(self):
        return self._coords.shape

    @property
    def dimension(self):
        return len(self)

    def __len__(self) -> int:
        return self.shape[1]

    def __getitem__(self: Vector, index: int) -> float:
        try:
            return self._coords[0][index]
        except IndexError:
            raise IndexError( "tried to access %s[%d], when it has %d coordinates" % (self, index, len(self)) ) from None

    def __array__(self):
        return self._coords

    def _validate_vector_input(vector_operation: Callable[[Vector, Vector], Any]):
        @wraps(vector_operation)
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

    def __rmul__(self, num: Number) -> Vector:
        """Returns the scalar product num * Vector"""
        if isinstance(num, Number):
            return Vector(num * self._coords)
        else:
            raise TypeError("unsupported operation with '%s'" % type(num)) from None

    @_validate_vector_input
    def __add__(self, other: Vector) -> Vector:
        """Returns the sum of two vectors"""
        return Vector(self._coords + other._coords)

    __radd__ = __add__

    @_validate_vector_input
    def __sub__(self, other: Vector) -> Vector:
        """Returns the difference of two vectors"""
        return Vector(self._coords - other._coords)

    __rsub__ = __sub__

    def __neg__(self: Vector) -> Vector:
        """Returns the negative vector"""
        return Vector(-self._coords)

    @_validate_vector_input
    def __mul__(self: Vector, other: Vector) -> float:
        """Returns the inner product of the vectors"""
        return (self._coords @ other._coords.T)[0][0]

    def squared(self: Vector) -> float:
        """Returns the inner product of the vector with itself"""
        return self * self

    def __abs__(self: Vector) -> float:
        """Returns the euclidean norm of the vector"""
        return self.squared()**.5

    def cos_angle(self: Vector, other: Vector) -> float:
        """Returns the cosine of the angle between two vectors"""
        return self * other / abs(self) / abs(other)

    def angle(self: Vector, other: Vector) -> float:
        """Returns the angle between two vectors"""
        return acos( self.cos_angle(other) )

    def __repr__(self) -> str:
        return self.__class__.__name__ + self._coords.__repr__()[5:]




class Point(Vector):
    """
    A class modeling a point in R**n

    Since a point P point P is essentially equivalent to the vector OP,
    where O=(0,...,0), we merely aliasing the class Vector.
    """

    __slots__ = Vector.__slots__






class Function:
    """
    A class modeling a function f: X ⊆ R**n -> S

    When an evaluation of the function is requested on a vector, if the function has not recently
    been evaluated for that vector the result is computed, the cost of the evaluation is recorded
    and the result is store for future reference. Otherwise, the stored result is returned and the
    number of evaluations remains unaffected.
    """

    __slots__ = ("_wrapped", "_dimensions", "_evaluations", "_name", "_cost", "_doc")

    _CACHE_SIZE = 32

    def __init__(self, pyfunc: Callable, cost_per_call: int = 1, name: Optional[str] = None) -> None:
        self._init_fields_using(pyfunc)
        if name:
            self._name = name
        self._cost = cost_per_call

    def _init_fields_using(self, pyfunc: Callable):
        self._name = pyfunc.__name__
        if isinstance(pyfunc, Function):
            self._wrapped = pyfunc._wrapped
            self._dimensions = pyfunc.domain_dimensions
        else:
            self._wrapped = pyfunc
            self._dimensions = len(signature(pyfunc).parameters)
        if pyfunc.__doc__:
            self._doc = pyfunc.__doc__
        else:
            self._doc = None
        self._evaluations = 0

    @property
    def __name__(self) -> str:
        return self._name
    
    @property
    def __doc__(self) -> Optional[str]:
        return self._doc

    @property
    def domain_dimensions(self) -> int:
        """
        Returns the dimension of the domain of the function

        In other words, it's the number of arguments expected
        """
        return self._dimensions

    @domain_dimensions.setter
    def domain_dimensions(self, value) -> None:
       self._dimensions = value

    @property
    def evaluations(self) -> int:
        """Returns the number of unique function evaluations performed thus far"""
        return self._evaluations

    @lru_cache(maxsize=_CACHE_SIZE)
    def _compute(self, args: Tuple[float]) -> Any:
        """
        Computes the function on a tuple, memoizes the result and records unique successful calls.
        It uses the Least Recently Used caching policy provided by the standard library.
        """
        result_when_call_succeeds = self._wrapped(*args)
        self._evaluations += self._cost
        return result_when_call_succeeds

    def _cache_info(self):
        return self._compute.cache_info()

    def __call__(self, vector: Vector) -> Any:
        """
        Applies the arguments collectively on the function, as in f(v) where v is a container; eg v = (v1, ..., vn)

        If a single number is provided as an argument, it's wrapped in a container automatically.
        """
        try:
            return self._compute( tuple(vector) )
        except TypeError as te:
            if "is not iterable" in te.args[0]:
                return self._compute( (vector,) )
            else:
                raise

    def call_separately_on(self, *args: float) -> Any:
        """Applies the arguments separately on the function, as in f(a1, a2, ..., an)"""
        return self(args)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.__name__} at {super().__repr__().split(' at ')[-1]}"


class DiffFunction(Function):
    """
    A class modeling a continuously differentiable function f: X ⊆ R**n -> S

    When an evaluation of the function or its derivative is requested on a vector, if the function has
    not recently been evaluated for that vector the result is computed, the cost of the evaluation is
    recorded and the result is store for future reference. Otherwise, the stored result is returned
    and the number of evaluations remains unaffected.

    Each new gradient call costs n times as much a single function call, since n partial derivatives
    are computed.
    """

    __slots__ = Function.__slots__ + ("gradient",)

    def __init__(self, function: Callable, gradient: Callable[..., Vector], name: Optional[str] = None) -> None:
        super().__init__(function, name=name)
        self.gradient: Callable[[Vector], Vector] = Function(
                                                                gradient,
                                                                cost_per_call=self.domain_dimensions,
                                                                name=(f"gradient of {name}" if name else None)
                                                            )

    @property
    def evaluations(self) -> int:
        """
        Returns the total number of unique function evaluations performed thus far
        That number also includes the evaluations of the gradient.
        """
        return self._evaluations + self.gradient.evaluations


class Diff2Function(DiffFunction):
    """
    A class modeling a twice continuously differentiable function f: X ⊆ R**n -> S

    When an evaluation of the function or its derivative is requested on a vector, if the function has
    not recently been evaluated for that vector the result is computed, the cost of the evaluation is
    recorded and the result is store for future reference. Otherwise, the stored result is returned
    and the number of evaluations remains unaffected.

    Each new gradient call costs n times as much a single function call, since n partial derivatives
    are computed.

    Each new hessian call costs n*(n+1)/2 times as much a single function call, since n*(n+1)/2 partial
    derivatives are computed.
    """

    __slots__ = DiffFunction.__slots__ + ("hessian", "inverse_hessian")

    def __init__(self, function: Callable, gradient: Callable, hessian: Callable, name: Optional[str] = None) -> None:
        super().__init__(function, gradient, name=name)
        n = self.domain_dimensions
        self.hessian: Callable[[Vector], ndarray] = Function(
                                                                hessian,
                                                                cost_per_call=n*(n+1)//2,
                                                                name=(f"hessian of {name}" if name else None)
                                                            )


        def inverse_hessian_as_a_Function(*vector: Vector) -> ndarray:
            return inv(self.hessian(vector))

        self.inverse_hessian: Callable[[Vector], ndarray] = Function(
                                                                        inverse_hessian_as_a_Function,
                                                                        name=(f"inverse hessian of {name}" if name else None)
                                                                    )
        self.inverse_hessian.domain_dimensions = self.domain_dimensions


    @property
    def evaluations(self) -> int:
        """
        Returns the total number of unique function evaluations performed thus far

        This number also includes the evaluations of the gradient and the hessian.
        """
        return self._evaluations + self.gradient.evaluations + self.hessian.evaluations







class FunctionNotDifferentiableError(BaseException):
    def __init__(self, function: Function, extend_msg: str="") -> None:
        msg = f"the function {function} is not defined as differentiable. Perhaps its gradient is not defined. {extend_msg}"
        super().__init__(msg)

class FunctionNotTwiceDifferentiableError(BaseException):
    def __init__(self, function: Function, extend_msg: str="") -> None:
        msg = f"the function {function} is not defined as differentiable. Perhaps its hessian is not defined. {extend_msg}"
        super().__init__(msg)







# def _doc_mappings(a_name):#todo!!
#     return f"""
#     A class modeling {a_name}

#     When it's called, it either retrieves the value from a cache or it computes the result anew,
#     if it wasn't recently computed.
#     Each one of those fresh computations is recorded as an evaluation of the function.
#     """



# class Function(Mapping):
#     __doc__ = _doc_mappings("a real-valued function of n real variables, f: X ⊆ R**n -> R")

#     __slots__ = Mapping.__slots__

#     def __init__(self, function: Callable) -> None:
#         super().__init__(function, cost_per_call=1)

#     def __call__(self, vector: Vector) -> float:
#         """
#         Computes the function on a point provided by the vector

#         There is no check to validate whether the dimension of the vector provided is correct.
#         Recent results are memoized.
#         """

#         return self._compute( tuple(vector) )


# class Gradient(Mapping):
#     __doc__ = _doc_mappings("the gradient of a function of n real variables, f: X ⊆ R**n -> R")

#     __slots__ = Mapping.__slots__

#     def __init__(self, vector_field: Callable, dimensions: int) -> None:
#         super().__init__(function, cost_per_call=1)

#     def __call__(self, vector: Vector) -> float:
#         """
#         Computes the gradient on a point provided by the vector

#         There is no check to validate whether the dimension of the vector provided is correct.
#         Recent results are memoized.
#         """

#         return self._compute( tuple(vector) )





# ### ??? do we need these???
# class NotAContainerError(BaseException):
#     def __init__(self, type_name: str, argsize, extend_msg: str="") -> None:
#         msg = f"a {type_name} must have at least two components, {argsize} functions were provided " + extend_msg
#         super().__init__(msg)

# class IncompatibleDomainsError(BaseException):
#     def __init__(self, extend_msg: str="") -> None:
#         msg = "there are at least two components with different number of domain dimensions " + extend_msg
#         super().__init__(msg)

# ### ???


# class VectorValuedFunction(Mapping):
#     __doc__ = _doc_mappings("a vector-valued function of n real variables, f: X ⊆ R**n -> R**m")

#     __slots__ = Mapping.__slots__ + ("_components",)

#     def __init__(self, *functions: Function) -> None:
#         self._components = tuple([Function(f) for f in functions])

#         def mapping_as_vector(vector: Vector):
#             return Vector([f(vector) for f in self._components])

#         self._wrapped = mapping_as_vector
#         self._dimensions = self[0].domain_dimensions
#         self._name = "(" + ", ".join([f.__name__ for f in self._components]) + ")"

#         for f in self._components:
#             if self.domain_dimensions != f.domain_dimensions:
#                 raise IncompatibleDomainsError

#     @property
#     def evaluations(self) -> int:
#         return sum([p.evaluations for p in self])

#     @evaluations.setter
#     def evaluations(self, value: int) -> None:
#         pass

#     def __len__(self) -> int:
#         return len(self._components)

#     def __getitem__(self, index: int) -> Function:
#         try:
#             return self._components[index]
#         except IndexError:
#             raise IndexError( "tried to access %s[%d] when it has %d components" % (self, index, len(self)) ) from None

#     def __call__(self, vector: Vector) -> Vector:
#         return Vector([p(vector) for p in self._components])



# class Gradient(VectorValuedFunction):
#     __doc__ = _doc_mappings("the gradient of a function of n real variables, f: X ⊆ R**n -> R")

#     __slots__ = VectorValuedFunction.__slots__

#     def __init__(
#                     self,
#                     dfunction: Function,
#                     *partials: Function
#                 ) -> None:
#         if len(partials) != dfunction.domain_dimensions:
#             raise ValueError(
#                 f"{len(partials)} partial derivatives provided for {dfunction.__name__}, although {dfunction.domain_dimensions} derivatives should have been provided"
#             )
#         super().__init__(*partials)


# class DFunction(Function):
#     __doc__ = _doc_mappings("a differentiable function of n real variables, f: X ⊆ R**n -> R")

#     __slots__ = Function.__slots__ + ("gradient",)

#     def __init__(self, function: Function, *partials: Function) -> None:
#         super().__init__(function)
#         self.gradient = Gradient(function, *partials)

#     @property
#     def evaluations(self) -> int:
#         return self._evaluations + self.gradient.evaluations

#     @evaluations.setter
#     def evaluations(self, value: int) -> None:
#         self._evaluations = value


# class Hessian(VectorValuedFunction):
#     __doc__ = _doc_mappings("the hessian matrix of a function of n real variables, f: X ⊆ R**n -> R")
    
#     __slots__ = VectorValuedFunction.__slots__

#     def __init__(
#                     self,
#                     dfunction: Function,
#                     *partials: Function
#                 ) -> None:
#         n = dfunction.domain_dimensions
#         if len(partials) != n*(n+1)/2:
#             raise ValueError(
#                 f"{len(partials)} partial derivatives provided for {dfunction.__name__}, although {n} derivatives should have been provided"
#             )
#         super().__init__(*partials)
        
# class D2Function:
#     pass


# @Function
# def f(x,y):
#     return x*y

# @Function
# def D1f(x,y):
#     return y

# @Function
# def D2f(x,y):
#     return x

# v = DifferentiableFunction(f, D1f, D2f)


# class Hessian(BaseMapping):#todo!
#     __doc__ = _doc_mappings("a vector-valued function of n real variables, f: X ⊆ R**n → R**m")

#     __slots__ = BaseMapping.__slots__ + ("_components",)

#     def __init__(
#                     self,
#                     d2function: DFunction,
#                     *second_partials: Callable[[float], float] | ScalarFunction
#                 ) -> None:
#         if len(second_partials) < 4: raise NotAContainerError(self.__class__.__name__, len(second_partials))

#         self._set_accumulator(d2function)
#         self.evaluations = 0

#         self._components = tuple([ScalarFunction(f, self) for f in functions])

#         def mapping(*args):
#             return Vector([f(*args) for f in self._components])

#         self._wrapped = mapping
#         self.domain_dimensions = self[0].domain_dimensions

#         for f in self._components:
#             if self.domain_dimensions != f.domain_dimensions:
#                 raise IncompatibleDomainsError

#     @property
#     def __name__(self) -> str:
#         return "(" + ", ".join([f.__name__ for f in self._components]) + ")"

#     def __len__(self) -> int:
#         return len(self._components)

#     def __getitem__(self, index: int) -> ScalarFunction:
#         try:
#             return self._components[index-1]
#         except IndexError:
#             raise IndexError( "tried to access %s[%d] when it has %d components" % (self, index, len(self)) ) from None

#     # def __neg__(self) -> VectorValuedFunction:
#     #     negative = VectorValuedFunction( *[-f for f in self._components], accumulator=self._accumulator )
#     #     negative
#     #     return negative

#     def __call__(self, vector: Vector) -> Vector:
#         return Vector([p(vector) for p in self])

#     def __init__(
#                     self,
#                     d2function: DFunction,
#                     *second_partials: Callable[[float], float] | ScalarFunction
#                 ) -> None:
#         dim = d2function.domain_dimensions
#         if len(second_partials) != dim*(dim+1)/2:
#             raise ValueError(
#                 f"""wrong number {len(second_partials)} of second partial derivative provided for {d2function.__name__}. \
#                     Only n*(n+1)/2 derivatives should be provided, arranged as a flatten triangular matrix.
#                 """
#             )
#         super().__init__(*second_partials, accumulator=d2function)
        

#     def cholesky(self):
#         pass

#     def invert(self):
#         pass

#     def __getitem__(self, indices: Tuple[int, int]) -> ScalarFunction:
#         i,j = indices
#         i,j = i-1,j-1
        
#         raise NotImplementedError
#         return self._components[1]

# class Function(ScalarFunction):
#     __slots__ = ScalarFunction.__slots__

#     def __init__(self, function: Callable[[Tuple[float]], float] | ScalarFunction) -> None:
#         super().__init__(function)

# class DFunction(Function):
#     __slots__ = ScalarFunction.__slots__ + ("_grad",)

#     def __init__(
#                     self,
#                     function: Callable[[Tuple[float]], float] | ScalarFunction,
#                     gradient: Gradient
#                 ) -> None:
#         super().__init__(function)

#         self._grad = gradient
#         self._grad._accumulator = self

#     @property
#     def gradient(self):
#         return self._grad


# class D2Function(DFunction):
#     __slots__ = ScalarFunction.__slots__ + ("_hess",)

#     def __init__(
#                     self,
#                     function: Callable[[Tuple[float]], float] | ScalarFunction,
#                     gradient: Gradient,
#                     hessian: Hessian
#                 ) -> None:
#         super().__init__(function, gradient)
        
#         self._hess = hessian
#         self._hess._accumulator = self
        
#     @property
#     def hessian(self):
#         return self._hess














"""


class Gradient(VectorValuedFunction):
    __doc__ = _doc_mappings("the gradient of a function, f: X ⊆ R**n → R")

    def __init__(self, function: Function, *partial_derivatives: Callable[[float], Vector]) -> None:
        super().__init__(*partial_derivatives)
        dim = partial_derivatives[0].domain_dimensions
        if len(partial_derivatives) != dim:
            raise TypeError("the number of the partial derivatives given is different from the number of the domain dimensions")


class DifferentiableFunction(ScalarFunction):
    def __init__(self, function: Callable[[Tuple[float]], float], gradient: Gradient) -> None:
        super().__init__(function)
        if self.domain_dimensions != gradient.domain_dimensions:
            raise
        self.gradient = gradient
        
    def cache_info(self):
        return (self.cache_info(), self.gradient.cache_info())

    def _flush_cache(self) -> None:
        self.gradient._flush_cache()
        self._flush_cache()



@ScalarFunction
def f(x,y):
    return x**3+y**2

@ScalarFunction
def D1f(x,y):
    return 3*x**2

@ScalarFunction
def D2f(x,y):
    return 2*x

v=VectorValuedFunction(D1f,D2f)

1/0



# class MatrixValuedFunction(BaseMapping):
#     __doc__ = _doc_mappings("a matrix-valued function of n real variables, f: X ⊆ R**n → R**{n x m}")

#     def __init__(self, *functions: Tuple[Callable[[float], Vector]]) -> None:
#         if len(functions) < 2:
#             raise TypeError(f"a matrix-valued function must have at least two components, {len(functions)} provided")
#         self.domain_dimensions = self[0].domain_dimensions
        
#         self.shape = shape

#         if len(functions) != shape[0]*shape[1]:
#             raise TypeError(f"wrong shape given during the initialization of the matrix-valued function, {shape}!={len(functions)}")

#         dim = self.domain_dimension
#         temp_transfm = []
#         for i in range(self.domain_dimensions):
#             temp_transfm.append([])
#             for j in range(self.domain_dimensions):
#                 temp_transfm[i].append(ScalarFunction( functions[dim*i+j] ))
#             temp_transfm[i] = tuple(temp_transfm[i])
        
#         self._components = tuple(temp_transfm)

#         for f in self:
#             if self.domain_dimensions != f.domain_dimensions:
#                 raise "there are at least two components with different number of domain dimensions"

#         super().__init__(functions[0])
#         self.__name__ = str([f.__name__ for f in self])

#     @property
#     def evaluations(self) -> int:
#         return sum([f.evaluations for f in self])

#     def __len__(self) -> int:
#         return len(self._components)

#     def __getitem__(self, index1: int, index2: int) -> ScalarFunction:
#         try:
#             return self._components[index1][index2]
#         except IndexError:
#             raise IndexError( "tried to access %s[%d][%d]" % (self, index1, index2) ) from None

#     def __neg__(self) -> VectorValuedFunction:
#         for f in self: -f
#         return self

#     def cache_info(self):
#         return tuple([f._compute.cache_info() for f in self])

#     def _flush_cache(self) -> None:
#         for f in self:
#             f._flush_cache()

#     def __call__(self, vector: Vector) -> ndarray:
#         return array([p(vector) for p in self])



# class hessian




@ScalarFunction
def f(x,y):
    return x+y

# (d1f,d2f)
# DifferentiableFunction(f, Gradient())

print(f.evaluations,f(Vector(0,0)), f.evaluations)
print(f.evaluations,f(Vector(0,0)), f.evaluations)
print(f.evaluations,f(Vector(0,0)), f.evaluations)
print(f.evaluations,f(Vector(0,0)), f.evaluations)
print(f.evaluations,f(Vector(1,0)), f.evaluations)
print(f.evaluations,f(Vector(0,0)), f.evaluations)
print(f.cache_info())






1/0





### ??? probably not needed ???
class InvalidCallError(BaseException):
    def __init__(self, function: Function, *args, message: str="") -> None:
        m = function.argsnum
        err_msg = f"{function} takes {m} argument{'s' if m != 1 else ''}. It was given the arguments: {args}."
        self.message = err_msg + message
        super().__init__(self.message)



# try:
#     FloatCallable = Callable[[Unpack[float]], Any]
# except NameError:
#     FloatCallable = "Callable[[Unpack(float)], Any]"


class InvalidCallError(BaseException):
    def __init__(self, function: Function, *args, message: str="") -> None:
        m = function.argsnum
        err_msg = f"{function} takes {m} argument{'s' if m != 1 else ''}. It was given the arguments: {args}."
        self.message = err_msg + message
        super().__init__(self.message)



class CachedFunction:
    \"""
    A cache to save function evaluation results

    At any given time there are saved at most `SIZE` results.
    \"""
    SIZE = 8
    def __init__(self, function: Callable, size: int = self.SIZE) -> None:
        self.function = function
        self.memory = deque(maxlen=self.SIZE)
        self._unique_evaluations = 0

    @property
    def unique_evaluations(self) -> int:
        \"""
        A unique evaluation is an evaluation of the function that doesn't use the cache.
        It is performed when that value isn't available in the cache.
        \"""
        return self._unique_evaluations

    @unique_evaluations.setter
    def unique_evaluations(self, value) -> None:
        self._unique_evaluations = value

    def retrieve_value_or_evaluate(self, *args: Any) -> Any:
        \"""
        When the function is evaluated for some arguments, retrives the cached value
        if such a value exists among the last `SIZE` evaluations. Otherwise, it computes
        the result from `self.function`
        \"""
        for i,value in self.memory:
            if i == args:
                return value
        
        value = self.function(*args)
        self.memory.appendleft((args,value))
        self.unique_evaluations += 1
        return value
    
    def __repr__(self) -> str:
        return f"{self.memory}"





class NegativeCallable:
    def __init__(self, mapping: Callable) -> None:
        self.inner = mapping

    def __neg__(self) -> Callable:
        return self.inner

    def __call__(self, *args: Any) -> Any:
        return -self.inner(*args)



CallableTuple = Tuple[Callable]
CallableMatrix = Tuple[Tuple[Callable]]


class Function:
    __slots__ = ("_argsnum", "_func", "_identifier", "_formula", "_cache", "_type")
    
    ARGTYPES = ["packed", "separated"]
    
    def __init__(
                    self,
                    pyfunction: Callable | Function,
                    argsnum: Optional[int] = None,
                    *,
                    identifier: Optional[str] = None,
                    formula_str: Optional[str] = None
                ) -> None:
        \"""_summary_

        Args:
            pyfunction (Callable): _description_
            grad_pyfunction (Callable, optional): _description_. Defaults to None.
            argsnum (int, optional): _description_. Defaults to None.
            identifier (Optional[str], optional): _description_. Defaults to None.
            formula_str (Optional[str], optional): _description_. Defaults to None.
            grad_formula_str (Optional[str], optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
        \"""
        self._cache: CachedFunction = CachedFunction(pyfunction)
        
        self._set_attributes_or_copy_from_function(pyfunction, argsnum, identifier, formula_str)

        if self._argsnum is None:
            raise TypeError(f"{self.__class__.__name__}.__init__() missing 1 required positional argument: 'argsnum'")

    
    def _set_attributes_or_copy_from_function(
                                self,
                                function: Callable | Function,
                                argsnum: Optional[int],
                                identifier: Optional[str],
                                formula_str: Optional[str]
                               ):
        \"""Sets the attributes to the non-None values provided and copies the rest from the given function, if they're available.'\"""
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

    def __neg__(self) -> NegativeCallable:
        return NegativeCallable(self)

    def __call__(self, *args: Any) -> Any:
        \""" #todo!!
        Evaluates the function for the arguments provided and records that evaluation was performed

        Arguments can be provided separately or in a container to be unpacked.

        The evaluation is recorded only if the call is successful.
        \"""

        if self._argstype == "separated":
            return self._cache.retrieve_value_or_evaluate(*args)
        elif self._argstype == "packed" and len(args) and isinstance(args[0], Collection):
            return self._cache.retrieve_value_or_evaluate(args)
        raise InvalidCallError(self, *args)

    @property
    def cache_size(self):
        return self._cache.SIZE

    @property
    def evaluations(self) -> int:
        return self._cache.unique_evaluations

    @property
    def argsnum(self):
        return self._argsnum

    def __repr__(self) -> str:
        i = self._identifier
        f = self._formula
        left = "{ " if i or f else ""
        right = " }" if i or f else ""
        equals = " = " if i and f else ""
        info = f\"""{left}{i or ""}{equals}{f or ""}{right}\"""

        return f"{self.__class__.__name__}{info}" if info!="" else super().__repr__()



class VectorField:
    def __init__(self, *functions: Callable | Function) -> None:
        self.partials = functions

    def __getitem__(self, index: int) -> Function:
        return self.partials[index]

    def __call__(self, *args: Any) -> Any:
        return Vector([f(*args) for f in self])

    def __iter__(self):
        return self.partials

    @property
    def evaluations(self) -> int:
        return sum([f.evaluations for f in self])








class DifferentiableFunction(Function):
     #todo!! A differential

    __slots__ = ("_argsnum", "_func", "_grad", "_identifier", "_formula")

    def __init__(
                    self,
                    pyfunction: Callable,
                    grad_pyfunction: Tuple[Function],
                    argsnum: int = None,
                    *,
                    identifier: Optional[str] = None,
                    formula_str: Optional[str] = None
                ) -> None:
        #todo!! 
        super().__init__(pyfunction, argsnum, argstype=argstype, identifier=identifier, formula_str=formula_str)
        gradient_identifier = f"D{self._identifier}" if self._identifier else None
        
        if isinstance(pyfunction, DifferentiableFunction):
            self._grad = pyfunction.gradient
            self._grad._identifier = gradient_identifier
        else:
            self._grad = Gradient(grad_pyfunction)
        
        if self.gradient is None:
            raise TypeError(f"{self.__class__.__name__}.__init__() missing 1 required positional argument: 'grad_pyfunction'")
        
            
    @property
    def gradient(self) -> Gradient:
        return self._grad
    
    derivative = gradient

    @property
    def evaluations(self) -> int:
        \"""
        Returns the number of times the function and its derivative have been evaluated
        
        The cost of the derivative is a multiple of the number of argument the function has.
        \"""
        # todo!! wrong score
        return super().evaluations + self.gradient.evaluations
    
    # def __repr__(self) -> str:
    #     return f"{super().__repr__()}, {self.gradient.__repr__()}"

DFunction = DifferentiableFunction

D2Function = DFunction


class TwiceDifferentiableFunction(DifferentiableFunction):
    __slots__ = ("_argsnum", "_argstype", "_func", "_grad", "_hess", "_identifier", "_formula")

    def __init__(
                    self,
                    pyfunction: Callable,
                    grad_pyfunction: CallableTuple,
                    hess_pyfunction: CallableMatrix,
                    argsnum: Optional[int] = None, 
                    *,
                    argstype: Optional[str] = None,
                    identifier: Optional[str] = None,
                    formula_str: Optional[str] = None,
                    grad_formula_str: Optional[str] = None,
                    hess_formula_str: Optional[str] = None
                ) -> None:  
        super().__init__(
            pyfunction,
            grad_pyfunction,
            argsnum,
            argstype=argstype,
            identifier=identifier,
            formula_str=formula_str,
            grad_formula_str=grad_formula_str
        )
        hessian_identifier = f"D**2{self._identifier}" if self._identifier else None
        self._hess = Function(
                                hess_pyfunction,
                                self._argsnum,
                                argstype=argstype,
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
        \"""
        Returns the number of times the function, its gradient and its hessian have been evaluated
        
        The cost of the derivative is a multiple of the number of argument the function has.
        The cost of the hessian is a multiple of the square of the number of argument that the
        function has.
        \"""
        # todo!! wrong score
        return super().evaluations + sum([self.hessian[i][j].evaluations for j in range(self.argsnum) for i in range(self.argsnum)])
    
    
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
  
    
    
    
"""
    



if __name__ == "__main__":
    a = Vector(1.,2.,3,4,5,6,7,8,9,10)
    b = Vector(3,4,5,6,7,8,9,10,11,12)
    print(a, b, a+b, a-b, a*b, a@b.T, 15*a, a.angle(b) , sep="\n")
    
    c: Vector =a+b
    
    print()
    print()
    
    @Function
    def D0f(x,y): return x**2+y**2

    @Function
    def D1f(x,y): return 2*Vector(x,y)

    @Function
    def D2f(x,y): return np.array([[2,0],[0,2]])

    f = Diff2Function(D0f, D1f, D2f)
    
    from math import sin, cos
    
    g= Diff2Function(lambda t: sin(t), lambda t: cos(t), lambda t: -cos(t) )
    
    h = deepcopy(g)
    
    exit(0)
    f = Function(lambda t: t**2, 1, formula_str="t**2")
    print("f=t**2", "t**2(t=100) =",f(100))
    g = DiffFunction(lambda t: t**2, lambda t: 2*t, 1)
    Dg = g.gradient
    print("t**2(t=100) =",g(100),"  Dt**2(t=100) =",Dg(100))
    
    
    f = lambda x,y: x**3 + y**3
    Df = lambda x,y: Vector(3*x**2,3*y**2)
    D2f = lambda x,y: array([[6*x,0],[0,6*y]]) 
    
    F = Function(f,2, identifier="F(x,y)", formula_str="x**3 + y**3")
    F1 = DiffFunction(F,Df, grad_formula_str="(3*x**2, 3*y**2)")
    F2 = Diff2Function(F,F1.gradient,D2f, hess_formula_str="[[6*x,0],[0,6*y]]")
    
    w = a._coords.T @ b._coords


