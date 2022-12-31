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
        plt.close()
        
    
    def save_plot(self, data, method, n_plot, polynomial_function, objective_function_value, start_x, test_data = None, Pm_test = None,test_data_error = None):
        
        plt.clf()
    
        N = len(data)
        
        plt.scatter(np.arange(1, N + 1, 1), data[:N], label = 'data')
        plt.plot(np.arange(1, N + 1, 1), polynomial_function, color = 'red', label = 'fitted curve')

        if test_data_error:
            plt.plot(np.arange(1, len(test_data) + 1, 1), Pm_test, color = 'green', label = "prediction")
            plt.plot(np.arange(1, N + 1, 1), polynomial_function, color = 'red')
            plt.scatter(np.arange(1, len(test_data) + 1, 1), test_data,color = 'maroon', label = 'test data')
            plt.scatter(np.arange(1, N + 1, 1), data[:N], color = '#1f77b4')

            plt.title(f"Optimal {method} solution: \n MSE of prediction = {test_data_error}")
        
        else:
            plt.title(f"{method} {n_plot}: f(x) = {objective_function_value}\n Starting point: {tuple(start_x)}")

        plt.legend()
        
        if test_data_error:
            plt.savefig(f"Plots/{method}/{method}-Prediction")
        else:
            plt.savefig(f"Plots/{method}/{method + str(n_plot)}")

        plt.figure().clear()
        plt.close()