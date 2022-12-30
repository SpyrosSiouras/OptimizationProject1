from dataset import Data
from problem_functions import *
import config
from line_search import line_search_optimization
from dogleg import doglegBFGS
from solution import Solution

np.set_printoptions(precision=15)





def run_simulation(method: str, data_points: list, train_data,t):

    solutions = []

    for i in range(len(data_points)):

        x_inputs = [data_points[i]]
        if method == "doglegBFGS":
            xs = doglegBFGS(train_data,t, x_inputs, config.MAX_ITERATIONS, config.ACCURACY)
        else:
            xs = line_search_optimization(train_data,t, x_inputs, config.MAX_ITERATIONS, config.ACCURACY, method)
        x_opt = xs[-1]
        sol = objective_function(train_data, Pm(t,x_opt))

        # print(f"Minimizer = {x_opt} with min value: {sol} in {config.function_calls} function calls! ")
        
        start_x = x_inputs[0].tolist()
        start_x = [j for sub in start_x for j in sub]
        start_x = [ '%.2f' % elem for elem in start_x]
        start_x = [float(x) for x in start_x]
        data.save_plot(train_data,method,i,Pm(t,x_opt), sol, start_x)
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

    optimization_algorithms = ["Steepest-Descent", "Newton", "BFGS", "doglegBFGS"]

    data_points = [np.random.uniform(-5,5, [5,1]) for _ in range(config.START_POINTS)]



    t = create_polynomial_matrix(len(train_data), nth_power=4)
    
    for algorithm in optimization_algorithms:
        print("Running...")
        solutions = run_simulation(algorithm, data_points, train_data, t)

        function_calls = sorted([x.function_calls for x in solutions])
        function_values = [x.function_value for x in solutions]
        minfc, maxfc, meanfc, stdfc = calculate_statistics(function_calls)
        meanfc = round(meanfc,2)
        stdfc = round(stdfc,2)
        minfv, maxfv, meanfv, stdfv = calculate_statistics(function_values)
        print("")
        print(f"\t\t{algorithm} statistics")
        print("")
        print(f"\tFunction calls\t Function value")
        print(f"Mean:\t  {meanfc} \t {meanfv}")
        print(f"Min: \t  {minfc}        \t {minfv}")
        print(f"Max: \t  {maxfc}         \t {maxfv}")
        print(f"Std: \t  {stdfc}   \t {stdfv}")
        print("")

        _ = input("\nPress enter to continue! ")



