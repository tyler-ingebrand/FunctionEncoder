from typing import Any


class BaseCallback:
    """ Base class for all callbacks."""
    def __init__(self):
        """ Constructor for BaseCallback"""
        pass

    def on_training_start(self, locals: dict) -> None:
        """ Called at the start of training.

        Args:
        locals (dict): A dictionary of local variables.
        """
        pass

    def on_step(self, locals: dict) -> None:
        """ Called at the end of each training step. 

        Args:
        locals (dict): A dictionary of local variables.
        """
        pass

    def on_training_end(self, locals: dict) -> None:
        """ Called at the end of training.

        Args:
        locals (dict): A dictionary of local variables.
        """
        pass