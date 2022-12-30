class Solution:
    """
    Data class holder for stats calculations
    """
    def __init__(self, function_value,x_opt, function_calls) -> None:
        
        self.function_value = function_value
        self.x_opt = x_opt
        self.function_calls = function_calls