from model.torch_ext import *


class ModuleWrapper:
    """
    The purpose of this class is to be able to query the size of a module even though it could end up being invalid
    """

    def __init__(self, module, input_size, output_size, **kwargs):
        if isinstance(module, str):
            module = get_class(module)

        self.module = module

        self.input_size = input_size
        self._output_size = kwargs.pop("output_size", output_size)
        self.parameters = kwargs

    @property
    def output_size(self):
        if self.module is None:
            return 0

        if self.input_size == 0 and self.parameters.get("in_channels_cond", 0) == 0:
            return 0
        else:
            return self._output_size


    def get_module(self):
        if self.output_size > 0:
            return self.module(self.input_size, self.output_size, **self.parameters)
        else:
            return None

