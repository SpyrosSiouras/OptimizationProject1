from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, TimedAnimation
import numpy as np
from math import pi, isclose
from scipy.optimize import line_search
import matplotlib.style as mplstyle
mplstyle.use(['ggplot', 'fast'])


from optimization.test_functions import griewank_1st_order

def gr(t):
    return griewank_1st_order(Vector(t))
def Dgr(t):
    return griewank_1st_order.gradient(Vector(t))[0]






def start_visualize(f,gradf,x,p,amax, ymax= None,c1=10**-4,c2=0.9):
    if gradf(x)*p>=0: raise ValueError("p is not a descent direction")
    a = np.linspace(-amax,amax,500)
    
    phi0 = f(x)
    Dphi0 = p*gradf(x)

    phi = np.array([f(x+a_*p) for a_ in a])
    
    l =  phi0+a *c1* Dphi0
    Dphi_line = phi0 + a* Dphi0

    # curv = -p*gradf(x+a*p) + c2*p*gradf(x) # >=


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.spines['left'].set_position('center')
    #ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    
    # plot the function
    plt.grid(True)
    plt.plot(a, phi, 'r')
    plt.plot(a, l, 'g')
    plt.plot(a, Dphi_line, "b")
    plt.plot([0],[f(x)], "yo")
    plt.plot([0,0],[plt.ylim()[0],f(x)],":k")
    if ymax is None:
        plt.ylim( 4/3*min(phi), 4/3*max(phi) )
    else:
        plt.ylim( -ymax, ymax )
    
    lim = lambda a: max([ abs(i) for i in plt.ylim() ])
    pink=plt.fill_between(x=a, y1=-lim(a), y2=phi, color="pink",where=(phi<f(0)), label="φ is smaller")
    green = plt.fill_between(x=a, y1=lim(a), y2=phi, color="green",where=phi<=l, label="armijo is true")
    yellow=plt.fill_between(x=a, y1=lim(a), y2=phi, color="yellow",where=np.logical_not( np.array([
                                                                                            armijo_isnt_met(f, f(x), p*gradf(x), x, p, a_, c1=c1) for a_ in a
                                                                                            ])
                                                                                        ),
                                                                                        label="armijo is true"
                                                                                        )
    purple=plt.fill_between(x=a, y1=-lim(a)*2/3, y2=phi , color="purple",where=np.logical_not(
                                                                                                np.logical_not(np.array([
                                                                                                    strong_curviture_isnt_met(gradf, p*gradf(x), x, p, a_, c2=c2) for a_ in a
                                                                                                ]))),
                                                                                                label="curviture is true"
    )
    orange=plt.fill_between(x=a, y1=phi, y2=-lim(a)/5*2, color="orange",where=np.logical_and(
                                                                                            np.logical_not(np.array([
                                                                                                    strong_curviture_isnt_met(gradf, p*gradf(x), x, p, a_, c2=c2) for a_ in a
                                                                                                ])
                                                                                                ),
                                                                                            np.logical_not(
                                                                                                np.array([
                                                                                            armijo_isnt_met(f, f(x), p*gradf(x), x, p, a_, c1=c1) for a_ in a
                                                                                            ]) 
                                                                                                           )
                                                                                        ),label="armijo and strong curviture is true"
                     )
    plt.legend()
    
    

    # show the plot
    # plt.show(block=False)

    return fig



def visualize(a_points,f,gradf,x,p,amax,ymax=None,c1=10**-4,c2=0.9):
    print(a_points)
    fa_points = [f(x+a[0]*p) for a in a_points]
    point_set = set()
    
    fig = start_visualize(f,gradf,x,p,amax,ymax,c1=c1,c2=c2)
    graph, = plt.plot([], [], 'ok')
    labels = []
    for n,a in enumerate( a_points ):
        if a[0] not in point_set:
            point_set.add(a[0])
            labels.append( plt.annotate("", (a[0],fa_points[n])) )
    def update(i):
        graph.set_data([a[0] for a in a_points[:i+1]], fa_points[:i+1])
        point_set = set()
        for n, label in enumerate( labels[:i+1] ):
            if a_points[n][0] not in point_set:
                point_set.add( a_points[n][0] )
                label.set_text(a_points[n][1]+" "+str(n))
        return graph
    
    return FuncAnimation(
        fig,
        update,
        frames=len(a_points),
        repeat=True,
        repeat_delay=500
    )

    plt.show()


def line_search_backtrack_till_smaller_values(f, x, p, debug=True): ### todo!!!
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    aes = [1]
    a = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(*x)
    while f(*(x+a*p)) > fx:
        a *= 0.5
        aes.append(a)
        if debug: print(a,x+a*p)
    
    return aes



def armijo_isnt_met(f, fx, pgradfx, x, p, a, c1=10**-4):
    return f(x+a*p) >= fx + c1 * a * pgradfx

def curviture_isnt_met(gradf, pgradfx, x, p, a, c2 = 0.9):
    return p*gradf(x+a*p) <= c2*pgradfx

def strong_curviture_isnt_met(gradf, pgradfx, x, p, a, c2 = 0.9):
    return abs(p*gradf(x+a*p)) >= c2*abs(pgradfx)


def line_search_backtrack_armijo(f, gradf, x, p, c1=10**-4,c2=0.9, debug=True):
    '''
    BACKTRACK LINE SEARCH WITH AMRijo CONDITIONS
    '''
    #file:///C:/Users/steve/Desktop/msc%20cse/%CE%943/Numerical%20Optimization_NOCEDAL%20and%20WRIGHT_2nd%20Ed_2006.pdf
    # slide 56
    aes = [1]
    a = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)
    pgradfx = p * gradf(x)
    if pgradfx >= 0: raise ValueError(f"{p} is not a descent direction")
    try:
        while armijo_isnt_met(f, fx, pgradfx, x, p, a,c1):
            if debug: print(x+a*p)
            a *= 0.5
            aes.append(a)
    except KeyboardInterrupt:
        pass
    return aes 



#https://github.com/gjkennedy/ae6310
#https://github.com/gjkennedy/ae6310/blob/master/Line%20Search%20Algorithms.ipynb
#https://indrag49.github.io/Numerical-Optimization/line-search-descent-methods.html
def line_search(f, gradf, x, p, a_max, c1=10**-4, c2=0.9, debug=True):
    phi = lambda a: f(x+a*p)
    pgradfx = p*gradf(x)
    if pgradfx>=0: raise ValueError(f"{p} is not a descent direction")
    Dphi = lambda a: p*gradf(x+a*p)
    fx = f(x)
    a_= [(0,"φ0")]
    a_next = (a_[-1][0]+a_max)/2
    
    def zoom(a_l, a_h):
        A_L = []
        print(f"start zooming {a_l=} {a_h=}")
        try:
            while not isclose(a_l,a_h):
                A_L.append((a_l,"a_l"))
                A_L.append((a_h, "a_h"))
                a = (a_l+a_h)/2
                A_L.append((a,"a"))
                print(f"{a_l=},{a_h=} {a=}",end=" ")
                print(f"{phi(a_l)=} {phi(a_h)=}")
                if phi(a_l)>phi(a_h): 
                    print(f"wtf???? {phi(a_l)=} {phi(a_h)=}")
                else:
                    print()
                if armijo_isnt_met(f, fx, pgradfx, x, p, a, c1) or (phi(a)>=phi(a_l)):
                    print("change high")
                    a_h = a
                else:
                    if abs(Dphi(a))<= -c2*Dphi(0):
                        print("found it!")
                        return A_L, a
                    if Dphi(a)*(a_h-a_l)>=0:
                        print("switcharoo")
                        a_h = a_l
                    a_l = a
            #a = (a_l + a_h)/2
            #return a
            if armijo_isnt_met(f, fx, pgradfx, x, p, a, c1) or abs(Dphi(a))> -c2*Dphi(0):
                print("zoom failed")
                return A_L, a
                raise ValueError("zoom failed")
            else:
                return a
        except KeyboardInterrupt:
            return a

    try:
        while not isclose(a_[-1][0],a_max):
            print(f"{a_=} {a_next=}")
            if armijo_isnt_met(f, fx, pgradfx, x, p, a_next, c1):
                print("zoom1a")
                if phi(a_next)<phi(a_[-1][0]): 
                    print(f"wtf!!! {a_[-1]=} {a_next=} ")
                    print(f"       {phi(a_[-1][0])} {phi(a_next)=}")
                A_L, a_0 = zoom(a_[-1][0],a_next)
                return a_ + [(a_next,"a_next")] + A_L + [(a_0,"last")]
            if (phi(a_next)>=phi(a_[-1][0])) and len(a_)>1:
                print("zoom1b")
                A_L, a_0 = zoom(a_[-1][0],a_next)
                return a_ + [(a_next,"a_next")] + A_L + [(a_0,"last")]
            if abs(Dphi(a_next))<=-c2*Dphi(0):
                print("found wolfe!")
                return a_
            if Dphi(a_next)>=0:
                print(f"zoom3")
                A_L, a_0 = zoom(a_next,a_[-1])
                return a_ + [(a_next,"a_next")] + A_L + [(a_0,"last")]
            a_.append((a_next,"a_"))
            a_next = (a_next+a_max)/2
        print("line search failed")
        return a_
        if armijo_isnt_met(f, fx, pgradfx, x, p, a[-1], c1) or abs(Dphi(a[-1]))> -c2*Dphi(0):
            raise ValueError("line search failed")
        else:
            return a
        
    except KeyboardInterrupt:
        return a_

def viz_line_search(f, gradf, x, p, a_max, ymax= None, c1=10**-4, c2=0.9, debug=True):
    a = line_search(f, gradf, x, p, a_max, c1, c2, debug)
    ani =  visualize(a,f,gradf,x,p,a_max,ymax,c1,c2)
    plt.show(block=False)
    return ani



from optimization.mathchinery import Vector


def f(x,y):
    return x**2+y**2

def gradf(x,y):
    return 2*Vector(x,y)

def s(x):
    return x*x

def Ds(x):
    return 2*x

def g(t):
    return np.sin(pi*t)

def Dg(t):
    return pi*np.cos(pi*t)

def c(t):
    return t**3-3*t

def Dc(t):
    return 3*t**2-3
    
