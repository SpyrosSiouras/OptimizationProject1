import numpy as np
import matplotlib.pyplot as plt


from obj_func import *



def rosenbrock2(z):
    x,y = z
    return (1-x)**2 + 100*(y-x**2)**2

def grad_rosenbrock2(z):
    x,y = z
    return np.array( (2*(x-1) + 400*(x**2-y)*x, 200*(y-x**2) ) )

def hess_rosenbrock2(z):
    x,y = z
    return np.array([
        [1200*x**2 - 400*y + 2, -400*x],
        [-400*x, 200]
    ])

def line_search(f, x, p):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)
    x_new = x + a * p
    while f(x_new) >= fx:
        a *= 0.5
        x_new = x + a * p
    return a


def newton(f, grad, hess, x0, max_it, plot=False):
    '''
    DESCRIPTION
    Newton with Wolfe line search
    INPUTS:
    f:      a function to be optimised 
    grdf:   a function of the gradient of the function to be optimised
    hessf:  a function of the hessian of the function to be optimised
    x0:     start point
    max_it: maximum iterations 
    plot:   if the problem is 2 dimensional, returns 
            a trajectory plot of the optimisation scheme.
    OUTPUTS: 
    x:      the optimal solution of the function f 
    '''
    d = len(x0)  # dimension of problem
    gradfx = grad(x0)  # initial gradient
    Hfx = hess(x0)  # initial hessian
    x = x0[:]
    it = 2
    if plot == True:
        if d == 2:
            x_store = np.zeros((1, 2))  # storing x values
            x_store[0, :] = x
        else:
            print('Too many dimensions to produce trajectory plot!')
            plot = False

    while np.linalg.norm(gradfx) > 1e-5:  # while gradient is positive
        if it > max_it:
            print('Maximum iterations reached!')
            break
        it += 1
        p = -np.linalg.inv(Hfx) @ gradfx  # search direction (Newton Method)
        a = line_search(f, x, p)  # line search
        x += a * p
        gradfx = grad(x)  # initial gradient
        Hfx = hess(x)
        if plot == True:
            x_store = np.append(x_store, [x], axis=0)  # storing x
    if plot == True:
        x1 = np.linspace(min(x_store[:, 0]-0.5), max(x_store[:, 0]+0.5), 30)
        x2 = np.linspace(min(x_store[:, 1]-0.5), max(x_store[:, 1]+0.5), 30)
        X1, X2 = np.meshgrid(x1, x2)
        Z = f([X1, X2])
        plt.figure()
        plt.title('OPTIMAL AT: ' +
                  str(x_store[-1, :])+'\n IN '+str(len(x_store))+' ITERATIONS')
        plt.contourf(X1, X2, Z, 30, cmap='jet')
        plt.colorbar()
        plt.plot(x_store[:, 0], x_store[:, 1], 'x-r')
        plt.plot(x0[0], x0[1], '*-r')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.show()
    return x


def sphere(z):
    x,y = z
    return x**2 + y**2

def grad_sphere(z):
    x,y = z
    return np.array([2*x, 2*y])

def hess_sphere(z):
    x,y = z
    return np.array([
        [1,0],
        [0,1]
    ])




# tests
# x_opt = newton(rosenbrock2, grad_rosenbrock2, hess_rosenbrock2, random_point(2), 100, plot=True)
# x_opt = newton(sphere, grad_sphere, hess_sphere, random_point(2), 100, plot=True)


start_at = random_point(5)
x_opt = newton(obj_func, gradient_obj_func, hessian_obj_func, start_at, 1000)
plot_against(x_opt, start_at)
