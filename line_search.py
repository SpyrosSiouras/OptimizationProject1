from problem_functions import *


def line_search_wolf_conditions(t,fx ,x,y, p, a0):

    a_max = 1
    a = (a0 + a_max) / 2
    c1 = 1e-4
    c2 = 0.9
    count = 0
    x_new = x + a * p
    gfx = gradient(y,Pm(t,x),t).T@p
    Pm_new = Pm(t, x_new)
    fx_new = objective_function(y, Pm_new)
    armijo_condition = ( fx_new > fx + c1*a*gfx)
    second_armijo_condition = ( fx_new >= fx and count > 1) 
    while count < 20:

        if armijo_condition or second_armijo_condition:
            a = zoom(a0, a, y, t, fx, gfx, x, p, c1, c2)
            return a

        gf = gradient(y, Pm_new,t).T @ p

        if np.linalg.norm(gf) <= c2*np.linalg.norm(gfx):
            return a 
        
        if gf >= 0:
            a = zoom(a, a0, y,t, fx, gfx, x, p, c1, c2)

        a0 = a    
        a = (a0 + a_max) / 2
        x_new = x + a * p
        fx_old = fx_new
        Pm_new = Pm(t, x_new)
        fx_new = objective_function(y, Pm_new)
        armijo_condition = ( fx_new > fx + c1*a*gfx)
        second_armijo_condition = ( fx_new >= fx_old ) 
        count += 1 

    return a


def zoom(a_low, a_high, y, t, fx, gfx, x,p, c1, c2):

    for _ in range(40): 

        a = (a_low + a_high) / 2
        Pm_n = Pm(t, x + a * p)
        obj = objective_function(y, Pm_n)

        if obj > fx + c1*a*gfx or obj >= objective_function(y, Pm(t, x + a_low * p)):

            a_high = a

        else: 
            
            gf = gradient(y, Pm_n, t).T @ p
 
            if np.fabs(gf) <= c2* np.fabs(gfx):
                return a 

            if gf* (a_high - a_low) >= 0:
                a_high = a_low

            a_low = a
        
    return a


def line_search_optimization(y,t, x_inputs,max_iterations, accuracy, method ):

    H = None

    if method == "BFGS": 
        # H = np.eye(5)  # initial hessian
        H = hessian(y, 5)

    else:
        H = hessian(y, 5)
    
    # print(f"Hessian of objective function is always positive definite: {is_pos_def(H)}")
    # _ = input("Press enter to continue! ")

    Pm_x = Pm(t, x_inputs[-1])
    gf = gradient(y, Pm_x,t)

    while (np.linalg.norm(gf) > accuracy):

        if config.function_calls > max_iterations:
            # print(f"Couldn't reach desired accuracy ({np.linalg.norm(gradient(y, Pm(t, x_inputs[-1]), t))} >  {accuracy})!")
            break

        x = x_inputs[-1]
        Pm_x = Pm(t, x)
        gf = gradient(y, Pm_x,t)
        fx = objective_function(y, Pm_x)

        if method == "Steepest-Descent": 
            pk = -gf 
            lr = line_search_wolf_conditions(t, fx, x, y, pk, a0 = 0)
        elif method == "Newton":
            pk = -np.linalg.inv(H) @ gf 
            lr = line_search_wolf_conditions(t, fx, x, y, pk, a0 = 1)
        elif method == "BFGS":
            pk = -H @ gf
            lr = line_search_wolf_conditions(t, fx, x, y, pk, a0 = 1)

        else:
            print("Didn't choose correct method! Try again with valid input!")
            exit(-1)

        x_new = x + lr * pk
        if method == "BFGS":
            H = calc_new_H(y, t, gf, lr, pk, x_new, H)


        x_inputs.append(x_new)

    return x_inputs