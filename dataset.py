import numpy as np
import matplotlib.pyplot as plt
import csv


class Data:

    def __init__(self, file_path: str) -> None:
        
        self.file_path = file_path
    
    def load_data(self):

        try:
            with open(self.file_path, 'r') as f:

                data = [column[1] for column in csv.reader(f, delimiter='\t')]
        except:
            print(
                "Something went wrong when reading the file. Please check your path and file, before trying again!")
            exit(0)

        data = [float(number) for number in data]
        data.reverse()  # or  data = data[::-1]
        data = np.array([data])

        return data.reshape(data.size, 1)
    
    def transform_data(self, dataset) -> np.array:
        """
        Transforms the data by substracting the mean value of the data.

        Input: One dimensional numpy array 
        Output: One dimensional numpy array 

        """
        return dataset - np.mean(dataset)
    
    
    def split_train_test(self, dataset ,train_length: int):

        return dataset[:train_length], dataset[train_length:]
    
    def plot_data(self,data: np.array, method : str,polynomial_function = None) -> None:
        
        plt.clf()
  
        N = len(data)
        
        plt.scatter(np.arange(1, N + 1, 1), data[:N], label = 'data')
        if polynomial_function is not None:
            plt.plot(np.arange(1, N + 1, 1), polynomial_function, color = 'red', label = 'fitted curve')
        plt.title(method)
        plt.legend()
        plt.show()
        
    
    def save_plot(self, data, method, n_plot, polynomial_function, objective_function_value):
        
        plt.clf()
    
        N = len(data)
        
        plt.scatter(np.arange(1, N + 1, 1), data[:N], label = 'data')
        if polynomial_function is not None:
            plt.plot(np.arange(1, N + 1, 1), polynomial_function, color = 'red', label = 'fitted curve')
            plt.title(f"{method}")
        if n_plot:
            plt.title(f"{method} {n_plot}: f(x) = {objective_function_value}")
            plt.legend()
            plt.savefig(f"Plots/{method}/{method + str(n_plot)}")

            plt.figure().clear()