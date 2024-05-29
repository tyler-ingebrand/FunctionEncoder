from typing import List, Any

from FunctionEncoder import BaseCallback


class ListCallback(BaseCallback):
    """A list of callbacks"""
    def __init__(self, callbacks:List[BaseCallback]):
        """ Stores a list of callbacks.
        """

        super(ListCallback, self).__init__()
        self.callbacks = callbacks

    def on_step(self, locals:dict) -> None:
        """Calls all callbacks at the end of each training step.
        """
        for callback in self.callbacks:
            callback.on_step(locals)

    def on_training_start(self, locals:dict) -> None:
        """ Calls all callbacks at the start of training.
        """
        for callback in self.callbacks:
            callback.on_training_start(locals)

    def on_training_end(self, locals:dict) -> None:
        """ Calls all callbacks at the end of training.
        """
        for callback in self.callbacks:
            callback.on_training_end(locals)

    def __getitem__(self, key):
        assert type(key) == int, "Key must be an integer"
        return self.callbacks[key]
