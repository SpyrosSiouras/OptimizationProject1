from math import cos, pi
from math_machinery import D2Function, Point, Vector
from numpy import array, ndarray, eye
from typing import List


__all__ = [
    "rosenbrock",
    "sphere2",
    "sphere3",
    "sphere10",
    "sphere20"
]


class TestFunction(D2Function):
    """
    A test function

    It's behavior is well understood and its minimums are known.
    """
    @property
    def minimizers(self) -> List:
        return self._minimizers
    @minimizers.setter
    def minimizers(self, value):
        self._minimizers = value



def _rosenbrock(x: float, y: float) -> float:
    """
    The Rosenbrock function

    Rosenbrock(x,y) := (1-x)**2 + 100*(y-x**2)**2
    
    It attains a global minimum at (1,1).
    
    https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    return (1-x)**2 + 100*(y-x**2)**2

def _D_rosenbrock(x: float, y: float) -> Vector:
    return Vector( 2*(x-1) + 400*(x**2-y)*x, 200*(y-x**2) )

def _D2_rosenbrock(x: float, y: float) -> ndarray:
    return array([
                   [ 1200*x**2 - 400*y + 2, -400*x ],
                   [        -400*x        ,   200  ]
                ])

rosenbrock = TestFunction(
    _rosenbrock,
    _D_rosenbrock,
    _D2_rosenbrock,
    argsnum=2,
    identifier="Rosenbrock(x,y)",
    formula_str="(1-x)**2 + 100*(y-x**2)**2"
)
rosenbrock.minimizers = Point(1,1)


def _sphere(*x: float) -> float:
    """
    The sphere test function

    Sphere(*x) := sum([ t**2 for t in x])

    It has a global minimum at O:=(0,...,0).
    """
    return Vector(x).squared()

def _D_sphere(*x: float) -> Vector:
    return 2 * Vector(x)

def _D2_sphere(*x: float) -> ndarray:
    return eye(len(x))

sphere = lambda dimensions: TestFunction(
    _sphere,
    _D_sphere,
    _D2_sphere,
    argsnum=dimensions,
    identifier=f"{dimensions}DSphere"
)

sphere2 = sphere(2)
sphere2.minimizers = Vector(0,0)

sphere3 = sphere(3)
sphere3.minimizers = Vector(0,0,0)

sphere10 = sphere(10)
sphere10.minimizers = Vector(10*[0])

sphere20 = sphere(20)
sphere20.minimizers = Vector(20*[0])









# def _himmelblau(x: float, y: float) -> float:
#     """
#     Himmelblau's function
    
#     Himmelblau(x,y) := (x**2 + y - 11)**2 + (x + y**2 - 7)**2

#     https://en.wikipedia.org/wiki/Himmelblau%27s_function
#     """
#     return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# def _D_himmelblau(x: float, y: float) -> Vector:
#     return Vector( 2*x*(x**2 + y - 11) + 2*(x + y**2 - 7), 2*(x**2 + y - 11) + 2*(x + y**2 - 7)*y  )
    
# def _D2_himmelblau(x: float, y: float) -> ndarray:
#     return array([
#                     2*(x**2 + y - 11) + 4*x**2 + 2, 2*y + 4*y
                    
#                  ])
    

# himmelblau = TestFunction(
    
# )



# def rasting(x, n):
#     """Rastrigin function

#     https://en.wikipedia.org/wiki/Rastrigin_function
#     """
#     A = 10
#     return A*n + sum([x[i]**2 - A*cos(2*pi*x[i]) for i in range(n)])


# def himmelblau(x, y):
#     """Himmelblau's function

#     https://en.wikipedia.org/wiki/Himmelblau%27s_function
#     """
#     return (x**2 + y - 1)**2 + (x + y**2 - 7)**2



# def levy(x):
#     d = len(x)
#     w = [1 + (x[i]-1)/4 for i in range(d)]
#     return sin(pi*w[0])**2 \
#          + sum([
#              (w[i]-1)**2*(1+10*sin(pi*w[i]+1)**2)
#             for i in range(1,d-1)]) \
#          + (w[d-1]-1)**2*(1+sin(2*pi*w[d-1])**2)
#     # https://www.sfu.ca/~ssurjano/levy.html
#     #slide 430

# def ff1(x,y):
#     return 4 - x**2 - y**2

# def ff2(x1,x2):
#     return (x1 − 2)**2 + (x2 − 1)**2
#     #s.t.: x1**2-x2<=0 and x1+x2<=2
#     #slide 28
#     #minimum at intersection of constraints


# def ff3(x1,x2):
#     return cos(x1)**2 + sin(x2)**2 + 0.1
#     # slide 423
#     # (mπ/2,nπ)


#page 650
#rosenbrock function
#extended rosenbrock function
#wood function
#power singular function
#cube function
#trigonometric function
#helical valley function   // https://al-roomi.org/benchmarks/unconstrained/3-dimensions/88-fletcher-powell-s-helical-valley-function
