import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path: str) -> np.array:
    """
    The file is epxected to be in a 2 column format, with a space as an delimeter.

    Input: A string that represents the relative path to the file. 
    Output: A numpy array 
    """
    try:
        with open(file_path, 'r') as f:

            data = [column[1] for column in csv.reader(f, delimiter='\t')]
    except:
        print(
            "Something went wrong when reading the file. Please check your path and file, before trying again!")
        exit(0)

    data = [float(number) for number in data]
    data.reverse()  # or  data = data[::-1]
    data = np.array([data])
    return data.reshape(data.size, 1)


def transform_data(dataset: np.array) -> np.array:
    """
    Transforms the data by substracting the mean value of the data.

    Input: One dimensional numpy array 
    Output: One dimensional numpy array 

    """
    return dataset - np.mean(dataset)


def plot_data(data: np.array, N=None, polynomial_function = None) -> None:
    """
    Plot of the data.

    Input: One dimensional Numpy array, N-> number of data points to plot.

    """
    x_axis = np.arange(1, N + 1, 1)
    if N:
        plt.scatter(np.arange(1, N + 1, 1), data[:N])
        if polynomial_function is not None:
            plt.plot(np.arange(1, N + 1, 1), polynomial_function)
        plt.show()
    else:
        plt.scatter(np.arange(1, 30 + 1, 1), data[:30])
        plt.show()

