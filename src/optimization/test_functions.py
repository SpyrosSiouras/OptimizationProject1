from math import cos, sin
from mathchinery import Diff2Function, Function, Point, Vector
from numpy import array, ndarray, eye
from typing import List



__all__ = [
    "rosenbrock",
    "griewank_1st_order",
    "griewank_2nd_order"
    "himmelblau",
    "sphere2",
    "sphere3",
    "sphere10",
    "cubic"
]




class TestFunction(Diff2Function):
    """
    A test function

    It's behavior is well understood and its minimums/minimizers are known.
    """

    __slots__ = Diff2Function.__slots__ + ("_minimizers",)

    @property
    def minimizers(self) -> List:
        return self._minimizers

    @minimizers.setter
    def minimizers(self, value):
        self._minimizers = value






@Function
def D0_rosenbrock(x: float, y: float) -> float:
    """
    The Rosenbrock function

    rosenbrock(x,y) := (1-x)**2 + 100*(y-x**2)**2

    It attains a global minimum at (1,1).

    https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    return (1-x)**2 + 100*(y-x**2)**2

@Function
def D1_rosenbrock(x: float, y: float) -> Vector:
    """Rosenbrock's gradient"""
    return Vector(  2*(x-1) + 400*(x**2-y)*x,  200*(y-x**2)  )

@Function
def D2_rosenbrock(x: float, y: float) -> ndarray:
    """Rosenbrock's Hessian"""
    return array([
                    [  1200*x**2 - 400*y + 2,  -400*x  ],
                    [         -400*x,            200   ]
                ])


rosenbrock = TestFunction( D0_rosenbrock, D1_rosenbrock, D2_rosenbrock, name = "rosenbrock" )
rosenbrock.minimizers = [Point(1,1)]






def D0_griewank_1st_order(x: float) -> float:
    """
    The first-order Griewank function

    griewank(x) := 1 + (1/4000) * x**2 - cos(x)

    It attains a global minimum at 0 and a lot of minima all over the place.

    https://en.wikipedia.org/wiki/Griewank_function
    """
    return 1 + (1/4000) * x**2 - cos(x)

@Function
def D1_griewank_1st_order(x: float) -> Vector:
    """First-order Griewank's gradient"""
    return Vector( 1/2000 * x + sin(x) )

@Function
def D2_griewank_1st_order(x: float) -> ndarray:
    """First-order Griewank's Hessian"""
    return array(1/2000 + cos(x), ndmin=2)


griewank_1st_order = TestFunction( D0_griewank_1st_order, D1_griewank_1st_order, D2_griewank_1st_order, name="griewank_1st_order" )
griewank_1st_order.minimizers = [Point(0)]






def D0_griewank_2nd_order(x: float, y: float) -> float:
    """
    The second-order Griewank function

    griewank(x) := 1 + (1/4000) * (x**2 + y**2) - cos(x)*cos(y/2**.5)

    It attains a global minimum at (0,0) and a lot of minima all over the place.

    https://en.wikipedia.org/wiki/Griewank_function
    """
    return 1 + (1/4000)*(x**2 + y**2) - cos(x)*cos(y/2**.5)

@Function
def D1_griewank_2nd_order(x: float, y: float) -> Vector:
    """Second-order Griewank's gradient"""
    return Vector(
        1/2000*x + sin(x)*cos(y/2**.5),
        1/2000*y + 1/2**.5*cos(x)*sin(y/2**.5)
    )

@Function
def D2_griewank_2nd_order(x: float, y: float) -> ndarray:
    """Second-order Griewank's Hessian"""
    return array([
                    [   1/2000 + cos(x)*cos(y/2**.5),    -1/2**.5*sin(x)*sin(y/2**.5)     ],
                    [   -1/2**.5*sin(x)*sin(y/2**.5),   1/2000 + 1/2*cos(x)*cos(y/2**.5)  ]
                ])


griewank_2nd_order = TestFunction( D0_griewank_2nd_order, D1_griewank_2nd_order, D2_griewank_2nd_order, name="griewank_2nd_order" )
griewank_2nd_order.minimizers = [Point(0,0)]






def D0_himmelblau(x: float, y: float) -> float:
    """
    The Himmelblau's function

    himmelblau(x,y) := (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    It attains a global minimum at (0,0) and a lot of minima all over the place.

    https://en.wikipedia.org/wiki/Himmelblau%27s_function
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

@Function
def D1_himmelblau(x: float, y: float) -> Vector:
    """Himmelblau's gradient"""

    return Vector(
        4*x**3 + 4*x*y - 42*x + 2*y**2 - 14,
        4*y**3 + 4*x*y - 26*y + 2*x**2 - 22
    )

@Function
def D2_himmelblau(x: float, y: float) -> ndarray:
    """Himmelblau's Hessian"""
    return array([
                    [   12*x**2 + 4*y - 42,        4*x + 4*y       ],
                    [       4*x + 4*y,        12*y**2 + 4*x - 26   ]
                ])


himmelblau = TestFunction( D0_himmelblau, D1_himmelblau, D2_himmelblau, name="himmelblau" )
himmelblau.minimizers = [Point(3,2), Point(-2.805118,3.131312), Point(-3.779310,-3.283186), Point(3.584428,-1.848126)]






def D0_sphere(*x: float) -> float:
    """
    The sphere test function

    sphere(*x) := sum([ t**2 for t in x])

    It has a single global minimum at O:=(0,...,0).
    """
    return Vector(x).squared()

def D1_sphere(*x: float) -> Vector:
    return 2 * Vector(x)

def D2_sphere(*x: float) -> ndarray:
    return eye(len(x))


sphere2 = TestFunction(
    lambda x,y: D0_sphere(x,y),
    lambda x,y: D1_sphere(x,y),
    lambda x,y: D2_sphere(x,y),
    name="sphere2"
)
sphere2.minimizers = [Point(0,0)]

sphere3 = TestFunction(
    lambda x,y,z: D0_sphere(x,y,z),
    lambda x,y,z: D1_sphere(x,y,z),
    lambda x,y,z: D2_sphere(x,y,z),
    name="sphere3"
)
sphere3.minimizers = [Point(0,0,0)]

sphere10 = TestFunction(
    lambda a,b,c,d,e,f,g,h,i,j: D0_sphere(a,b,c,d,e,f,g,h,i,j),
    lambda a,b,c,d,e,f,g,h,i,j: D1_sphere(a,b,c,d,e,f,g,h,i,j),
    lambda a,b,c,d,e,f,g,h,i,j: D2_sphere(a,b,c,d,e,f,g,h,i,j),
    name="sphere10"
)
sphere10.minimizers = [Point(0,0,0,0,0,0,0,0,0,0)]






@Function
def D0_cubic(t):
    """
    A cubic test function

    cubic(t) := t**3 - 3*t

    It attains a local minimum of -2 at 1.
    """
    return t**3 - 3*t

@Function
def D1_cubic(t):
    return 3 * Vector(t**2 - 1)

@Function
def D2_cubic(t):
    return 6 * Vector(t)

cubic = TestFunction( D0_cubic, D1_cubic, D2_cubic, name="cubic" )
cubic.minimizers = [Point(1)]











# def rasting(x, n):
#     """Rastrigin function

#     https://en.wikipedia.org/wiki/Rastrigin_function
#     """
#     A = 10
#     return A*n + sum([x[i]**2 - A*cos(2*pi*x[i]) for i in range(n)])



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
