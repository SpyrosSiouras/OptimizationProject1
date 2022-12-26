from dataset import load_data, transform_data, plot_data
import numpy as np

FILE_PATH = 'Government_Securities_GR.txt'

def create_polynomial_matrix(days: int, nth_power: int) -> np.array:
    
    powers = [nth_power-i for i in range(nth_power+1)]
    tn = np.array([np.arange(1, days + 1, 1)])
    t = (np.power(tn.reshape(tn.size, 1), powers))

    return t

def Pm(t: np.array, a: np.array) -> np.array:

    return t@a

def objective_function(y, polynomial_function):

    return np.mean(np.square(y - polynomial_function))

def gradient(y, polynomial_function, t):

    
    return -2 * np.array([[np.mean((y - polynomial_function)*t[:, 0])],
                          [np.mean((y - polynomial_function)*t[:, 1])],
                          [np.mean((y - polynomial_function)*t[:, 2])],
                          [np.mean((y - polynomial_function)*t[:, 3])],
                          [np.mean((y - polynomial_function)*t[:, 4])]])

def line_search(y, t, f_cur ,p, nabla):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = 1e-4
    c2 = 0.9
    count = 0
    fx = f_cur
    x_new = x + a * p
    fx_new = Pm(t, x_new)
    nabla_new = gradient(y, fx_new, t)
    while objective_function(y, fx_new) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p:
        if count >= 100:
            break
        # if a < 1e-11:
        #     return a 
        a *= 0.5
        x_new = x + a * p
        nabla_new = gradient(y, Pm(t, x_new), t)
        fx_new = Pm(t,x_new)
        # print(f"{fx_new}")
        count = count + 1
        # print(count)
    return a

def line_search_wolf_conditions(y,t,fx, x,p,grad_fx, ):

    alpha_0 = 0
    alpha_max = 1

    c1 = 1e-4
    c2 = 0.9
    count = 0
    max_iterations = 50
    while count < max_iterations:
        # print(f"Iteration: {count}")
        alpha = (alpha_0 + alpha_max) / 2
        x_new = x + alpha * p
        if objective_function(y, Pm(t, x_new)) > fx + (c1*alpha*grad_fx.T@p) or (objective_function(y, Pm(t, x_new)) >= objective_function(y, Pm(t,x + alpha_0 * p)) and count>1):
            # print(f"1: {alpha}-> {x_new}")
            alpha_max = alpha
            return alpha
        else:
            print("else")
            nabla_new = gradient(y, Pm(t, x_new), t)
            # print(Pm(t, x_new))
            if abs(nabla_new.T@p) <= -c2*p.T@p:
                print("Returning alpha")
                z = input()
                return alpha
            if nabla_new.T@p * (alpha_max - alpha_0) >= 0:
                # print(f"2: Changing alpha bounds")
                alpha_max = alpha_0
                alpha_0 = alpha
        
        count = count + 1 

    return alpha 


    
if __name__ == "__main__":

    data = load_data(FILE_PATH)

    data = transform_data(data)

    train_data = data[:25]
    test_data = data[25:]

    # plot_data(train_data, train_data.size, polynomial_function = None)

    t = create_polynomial_matrix(days = train_data.size, nth_power=4)
    # a = np.array([[0.00000011], [0], [0], [0], [0.1]])

    # print(objective_function(train_data,Pm(t,a)))

    # print(gradient(train_data, Pm(t,a), t))

    # print(t[:,0])
    x_inputs = [np.array([[0], [0], [0], [0], [0]])] #list of inputs, for plot 

    lr = 0.0001 #initialize step
    
    for i in range(100):

        x = x_inputs[-1]
        # print(x)
        # z = input()
        Pm_x = Pm(t,x)
        f_x = objective_function(train_data, Pm_x)
        pk = -gradient(train_data, Pm_x,t)
        # lr = line_search_wolf_conditions( train_data,t, f_x,x, pk, -pk)
        lr = line_search( train_data,t, f_x, pk, -pk)
        next_x = x + lr * pk
        # print(f"x={next_x}, lr={lr}, pk={pk}")
        # print(lr)
        x_inputs.append(next_x)

    # print(x_inputs[-1])
    # print(x_inputs[0])
    # print(x_inputs[1])
    # print(x_inputs[2])

    x_opt = x_inputs[-1]
    print(f"Before: {x_opt}")
    # x_opt = x_opt[::-1]
    print(f"After: {x_opt}")
    plot_data(train_data, train_data.size, Pm(t,x_opt))

    print(objective_function(train_data, Pm(t,x_opt)))