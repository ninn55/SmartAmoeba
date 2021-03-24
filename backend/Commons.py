from VariableNameHelper import VariableNameHelper

class ModelIR(object):
    """
    Store ops and tensors as list of object
    """
    def __init__(self):
        self._ops = list() # List of OperatorInterface
        self._tensors = list() # List of numpy array
    
    @property
    def Operator(self):
        return self._ops

    @Operator.setter
    def Operator(self, op):
        self._ops.append(op)

    @property
    def Tensor(self):
        return self._tensors

    @Tensor.setter
    def Tensor(self, tensor):
        self._tensors.append(tensor)

class TensorInterface(object):
    def __init__(self):
        self._name = str()
        self.shape = list()
        # Type 0 tensor means const
        # type 1 tensor means var
        # variable tensor's buffer is all 0 but shape infomation is valid
        self.tensorType = -1
        # The tensorSize is in pixel not bytes!
        self.tensorSize = -1
        self.tensor = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = VariableNameHelper.parse(name)

    def __str__(self):
        return "Tensor: " + self.name + \
                ", Shape: " + str(self.shape) + \
                ", Type: " + ("variable" if self.tensorType else "constant") + \
                ", Sizes: " + str(self.tensorSize) + "\n"