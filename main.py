from dataset import Data
from problem_functions import *
import config
from line_search import line_search_wolf_conditions
from solution import Solution

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

    solutions = []

    for i in range(len(data_points)):

        x_inputs = [data_points[i]]
        xs = line_search_optimization(train_data,t, x_inputs, config.MAX_ITERATIONS, config.ACCURACY, method)
        x_opt = xs[-1]
        sol = objective_function(train_data, Pm(t,x_opt))

        print(f"Minimizer = {x_opt} with min value: {sol} in {config.function_calls} function calls! ")
        
        data.save_plot(train_data,method,i,Pm(t,x_opt), sol)
        solutions.append(Solution(sol, x_opt, config.function_calls))
        config.function_calls = 0
    
    solutions.sort(key= lambda x: x.function_value)

    return solutions


def calculate_statistics(sorted_list_of_numbers: list):

    min = sorted_list_of_numbers[0]
    max = sorted_list_of_numbers[-1]
    mean = sum([x for x in sorted_list_of_numbers]) / len(sorted_list_of_numbers)
    std = (1/len(sorted_list_of_numbers) * sum([(x - mean)**2 for x in sorted_list_of_numbers]))**0.5


    return min, max, mean, std


if __name__ == "__main__":
    
    data = Data(config.FILE_PATH)
    dataset = data.load_data()
    transformed_data = data.transform_data(dataset)
    train_data, test_data = data.split_train_test(transformed_data, 25)
    np.random.seed(0)

    line_search_algorithms = ["Steepest", "Newton", "Bfgs"]

    data_points = [np.random.uniform(-5,5, [5,1]) for _ in range(config.START_POINTS)]



    t = create_polynomial_matrix(len(train_data), nth_power=4)
    
    for algorithm in line_search_algorithms:
        solutions = run_simulation(algorithm, data_points, train_data, t)
        # mean = sum([x.function_value for x in solutions]) / len(solutions)
        # min = solutions[0].function_value
        # max = solutions[-1].function_value
        # std = (1/len(solutions) * sum([(x.function_value - mean)**2 for x in solutions]))**0.5
        function_calls = sorted([x.function_calls for x in solutions])
        function_values = [x.function_value for x in solutions]
        minfc, maxfc, meanfc, stdfc = calculate_statistics(function_calls)
        minfv, maxfv, meanfv, stdfv = calculate_statistics(function_values)
        print("FUNCTION CALLS STATS")
        print(f"Mean: {meanfc}")
        print(f"Min: {minfc}")
        print(f"Max: {maxfc}")
        print(f"Std: {stdfc}")
        print("FUNCTION VALUE STATS")
        print(f"Mean: {meanfv}")
        print(f"Min: {minfv}")
        print(f"Max: {maxfv}")
        print(f"Std: {stdfv}")
        _ = input("Press enter to continue! ")


    # # x_inputs = [np.array([[ 1.276516314339567e-05],
    # #                     [ 1.413268420360388e-04],
    # #                     [ 1.283290949325388e-03],
    # #                     [-2.917351541258404e-04],
    # #                     [ 1.022544616397106e-05]])]


    ####################TESTING
    # x_inputs = [data_points[5]]

    # xs = line_search_optimization(train_data,t, x_inputs, config.MAX_ITERATIONS, config.ACCURACY, method="Newton")
    # x_opt = xs[-1]
    # sol = objective_function(train_data, Pm(t,x_opt))

    # print(f"Minimizer = {x_opt} with min value: {sol} in {config.function_calls} function calls! ")
    # print(f"start point: {xs[0]}")

    # # data.plot_data(train_data,Pm(t,x_opt))

    # t = create_polynomial_matrix(5, 4)

    # data.plot_data(test_data, "Newton",polynomial_function = Pm(t,x_opt))


