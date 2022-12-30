from dataset import Data
from problem_functions import *
import config
from line_search import line_search_wolf_conditions

np.set_printoptions(precision=15)


def calc_new_H(y, t, gf, lr, pk, x_new, H):
    
    s = lr * pk
    gf_new = gradient(y, Pm(t, x_new),t)
    y_h = gf_new - gf
    y_h = np.array([y_h])
    s = np.array([s])
    y_h = np.reshape(y_h, (5, 1))
    s = np.reshape(s, (5, 1))
    r = 1/(y_h.T@s)
    li = (np.eye(5)-(r*((s@(y_h.T)))))
    ri = (np.eye(5)-(r*((y_h@(s.T)))))
    hess_inter = li@H@ri
    
    return hess_inter + (r*((s@(s.T))))




def line_search_optimization(y,t, x_inputs,max_iterations, accuracy, method ):

    H = None

    if method == "Bfgs": 
        H = np.eye(5)  # initial hessian
    else:
        H = hessian(train_data, 5)
    
    print(f"Hessian of objective function is always positive definite: {is_pos_def(H)}")
    # _ = input("Press enter to continue! ")

    Pm_x = Pm(t, x_inputs[-1])
    gf = gradient(y, Pm_x,t)

    while (np.linalg.norm(gf) > accuracy):

        if config.function_calls > max_iterations:
            print(f"Couldn't reach desired accuracy ({np.linalg.norm(gradient(y, Pm(t, x_inputs[-1]), t))} >  {accuracy})!")
            break

        x = x_inputs[-1]
        Pm_x = Pm(t, x)
        gf = gradient(y, Pm_x,t)
        fx = objective_function(y, Pm_x)

        if method == "Steepest": 
            pk = -gf 
            lr = line_search_wolf_conditions(t, fx, x, y, pk, a0 = 0)
        elif method == "Newton":
            pk = -np.linalg.inv(H) @ gf 
            lr = line_search_wolf_conditions(t, fx, x, y, pk, a0 = 0.0005)
        elif method == "Bfgs":
            pk = -H @ gf
            lr = line_search_wolf_conditions(t, fx, x, y, pk, a0 = 0.01)

        else:
            print("Didn't choose correct method! Try again with valid input!")
            exit(-1)

        x_new = x + lr * pk
        if method == "Bfgs":
            H = calc_new_H(y, t, gf, lr, pk, x_new, H)

        # print(f"x_new: {x_new}")
        # print(f"{objective_function(y, Pm(t, x_new))} < {objective_function(y, Pm(t, x))}" )

        x_inputs.append(x_new)

    return x_inputs


def run_simulation(method: str, data_points: list, train_data,t):

    function_values = []

    for i in range(len(data_points)):

        x_inputs = [data_points[i]]
        xs = line_search_optimization(train_data,t, x_inputs, config.MAX_ITERATIONS, config.ACCURACY, method)
        x_opt = xs[-1]
        sol = objective_function(train_data, Pm(t,x_opt))

        print(f"Minimizer = {x_opt} with min value: {sol} in {config.function_calls} function calls! ")
        
        data.plot_data(train_data,method,Pm(t,x_opt))
        function_values.append(sol)
        config.function_calls = 0
    
    function_values.sort()



if __name__ == "__main__":
    
    data = Data(config.FILE_PATH)
    dataset = data.load_data()
    transformed_data = data.transform_data(dataset)
    train_data, test_data = data.split_train_test(transformed_data, 25)
    np.random.seed(0)

    line_search_algorithms = ["Steepest", "Newton", "Bfgs"]

    data_points = [np.random.uniform(-5,5, [5,1]) for _ in range(config.START_POINTS)]



    t = create_polynomial_matrix(len(train_data), nth_power=4)

    run_simulation(line_search_algorithms[1], data_points, train_data, t)
    # x_inputs = [data_points[5]]

    # # x_inputs = [np.array([[ 1.276516314339567e-05],
    # #                     [ 1.413268420360388e-04],
    # #                     [ 1.283290949325388e-03],
    # #                     [-2.917351541258404e-04],
    # #                     [ 1.022544616397106e-05]])]


    # xs = bfgs(train_data,t, x_inputs, MAX_ITERATIONS, ACCURACY)
    # x_opt = xs[-1]
    # sol = objective_function(train_data, Pm(t,x_opt))

    # print(f"Minimizer = {x_opt} with min value: {sol} in {config.function_calls} function calls! ")
    # print(f"start point: {xs[0]}")

    # data.plot_data(train_data,Pm(t,x_opt))

    # t = create_polynomial_matrix(30, 4)

    # data.plot_data(transformed_data, polynomial_function = Pm(t,x_opt))


