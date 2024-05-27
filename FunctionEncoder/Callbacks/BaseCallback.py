

class BaseCallback:
    """ Base class for all callbacks."""
    def __init__(self):
        """ Constructor for BaseCallback"""
        pass

    def on_step(self, function_encoder) -> dict:
        """ Called at the end of each training step. 

        Args:
        function_encoder (FunctionEncoder): The function encoder that is being trained.

        Returns:
        dict: A dictionary of information to log. 
        """
        pass