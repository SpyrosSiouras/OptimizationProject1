from dataset import load_data, transform_data, plot_data
import numpy as np

FILE_PATH = '../resources/Government_Securities_GR.txt'

def create_polynomial_matrix(days: int, nth_power: int) -> np.array:
    
    powers = [nth_power-i for i in range(nth_power+1)]
    tn = np.array([np.arange(1, days + 1, 1)])
    t = (np.power(tn.reshape(tn.size, 1), powers))

    return t

def Pm(t: np.array, a: np.array) -> float:

    return t@a

def objective_function(y, polynomial_function):

    return np.mean(np.square(y - polynomial_function))

def gradient(y, polynomial_function, t):

    return -2 * np.array([[np.mean((y - polynomial_function)*t[:, 0])],
                          [np.mean((y - polynomial_function)*t[:, 1])],
                          [np.mean((y - polynomial_function)*t[:, 2])],
                          [np.mean((y - polynomial_function)*t[:, 3])],
                          [np.mean((y - polynomial_function)*t[:, 4])]])


if __name__ == "__main__":

    data = load_data(FILE_PATH)

    data = transform_data(data)

    train_data = data[:25]
    test_data = data[25:]

    plot_data(train_data, train_data.size, polynomial_function = None)

    t = create_polynomial_matrix(days = train_data.size, nth_power=4)
    a = np.array([[0], [0], [0], [0], [0.1]])

    # print(objective_function(train_data,Pm(t,a)))

    # print(gradient(train_data, Pm(t,a), t))

    x_inputs = [np.array([[1],[1.5],[1],[1], [1]])] #list of inputs, for plot 

    lr = 0.0001 #initialize step
 
    for i in range(10):

        x = x_inputs[-1]
        pk = -gradient(train_data, Pm(t,a),t)
        next_x = x + lr * pk
        # print(pk)
        x_inputs.append(next_x)

    print(x_inputs[-1])

    plot_data(train_data, train_data.size, Pm(t,x_inputs[-1]))

    print(objective_function(train_data, Pm(t,x_inputs[-1])))
