import tflite
from tflite import Model
from tflite import BuiltinOptions 
from tflite import Conv2DOptions, Pool2DOptions, ReshapeOptions, FullyConnectedOptions, SoftmaxOptions

from FileHelper import file2Buffer
from EnumDictionarys import *
from collections import OrderedDict
from VariableNameHelper import *

import numpy as np

class OperatorInterface(object):
    def __init__(self):
        self._name = str()
        self.opcode = -1
        self.inputTensors = []
        self.outputTensors = []
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = VariableNameHelper.parse(name)

    def __str__(self):
        return "Operation: " + self.name + "\n"

class SoftmaxOperatorInterface(OperatorInterface):
    def __init__(self):
        super().__init__()
        self.option = {"beta": 0.0}
    
    def __str__(self):
        return "Operation: " + self.name + "\n" + \
                "Beta :" + str(self.option["beta"]) + "\n"

class FullyConnectedOperatorInterface(OperatorInterface):
    def __init__(self):
        super().__init__()
        self.option = {\
            "fusedActivationCode": -1, \
            "weightFormat": -1, \
            "keepNumDims": False, \
            "asymmetricQuantizeInputs": False \
            }
    
    def __str__(self):
        return "Operation: " + self.name + "\n" + \
                "Fused activation functions " + ActivationFunctionTypeEnumDict[self.option["fusedActivationCode"]] + "\n"

class ReshapeOperatorInterface(OperatorInterface):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return "Operation: " + self.name + "\n" + \
                "No extra options," + "\n"

class Conv2DOperatorInterface(OperatorInterface):
    def __init__(self):
        super().__init__()
        self.option = {\
            "strideW" : 0, "strideH" : 0, \
            "dilationW" : 0, "dilationH" : 0, \
            "fusedActivationCode" : -1, \
            "paddingCode" : -1 \
        }

    def __str__(self):
        info  = "Operation: " + self.name + "\n"
        info += "Options for this Conv2D layer are :" + "\n"
        info += "Fused activation functions " + ActivationFunctionTypeEnumDict[self.option["fusedActivationCode"]] + "\n"
        info += "Padding options :" + PaddingEnumDict[self.option["paddingCode"]] + "\n"
        info += "Stride options (w, h):  " + str((self.option["strideW"], self.option["strideH"])) + "\n"
        info += "Dilation options  (w, h): " + str((self.option["dilationW"], self.option["dilationH"])) + "\n"
        return info

class MaxPool2DOperatorInterface(OperatorInterface):
    """MaxPool2DOperatorInterface
    From tensorflow lite flatbuffer buffer data to high level IR
    """
    def __init__(self):
        super().__init__()
        self.option = {\
            "strideW" : 0, "strideH" : 0, \
            "filterW" : 0, "filterH" : 0, \
            "fusedActivationCode" : -1, \
            "paddingCode" : -1 \
        }

    def __str__(self):
        info  = "Operation: " + self.name + "\n"
        info += "Options for this MaxPool2D layer are :" + "\n"
        info += "Fused activation functions " + ActivationFunctionTypeEnumDict[self.option["fusedActivationCode"]] + "\n"
        info += "Padding options :" + PaddingEnumDict[self.option["paddingCode"]] + "\n"
        info += "Stride options (w, h):  " + str((self.option["strideW"], self.option["strideH"])) + "\n"
        info += "Filter Size options  (w, h): " + str((self.option["filterW"], self.option["filterH"])) + "\n"
        return info

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
                ", Type: " + "variable" if self.tensorType else "constant" + \
                ", Sizes: " + str(self.tensorSize) + "\n"

class ModelIR(object):
    def __init__(self):
        self._ops = list() # List of OperatorInterface
        self._tensors = list() # List of numpy array
    
    @property
    def Operator(self):
        return

    @Operator.setter
    def Operator(self, op: OperatorInterface):
        self._ops.append(op)

    @property
    def Tensor(self):
        return

    @Tensor.setter
    def Tensor(self, tensor: TensorInterface):
        self._tensors.append(tensor)

    def __str__(self):
        info = ""
        for op in self._ops:
            info += "\n"
            info += str(op)
            info += "\n Input: \n"
            for i in op.inputTensors:
                info += str(self._tensors[i])
            info += "\n Output: \n"
            for i in op.outputTensors:
                info += str(self._tensors[i])
            info += "\n"
        return info

class ModelHelperTFLite(object):
    def __init__(self, modelbuffer, IR: ModelIR):
        self.model = tflite.Model.Model.GetRootAsModel(modelbuffer, 0)
        self.subgraph = self.model.Subgraphs(0)
        self.IR = IR
        self._fillIR()
        
    def _fillIR(self):
        self._fillTensors()
        self._fillOperators()

    def _fillTensors(self):
        for i in range(self.subgraph.TensorsLength()):
            tensor = TensorInterface()
            tensor.name = self.subgraph.Tensors(i).Name().decode()
            tensor.shape = [self.subgraph.Tensors(i).Shape(j) for j in range(self.subgraph.Tensors(i).ShapeLength())]
            tensor.tensorSize = 1
            for k in tensor.shape:
                tensor.tensorSize *= k
            if self.model.Buffers(self.subgraph.Tensors(i).Buffer()).DataLength() == 0:
                tensor.tensor = np.zeros(tensor.shape)
                tensor.tensorType = 1
            else:
                buffer = []
                for j in range(self.model.Buffers(self.subgraph.Tensors(i).Buffer()).DataLength()):
                    buffer.append(self.model.Buffers(self.subgraph.Tensors(i).Buffer()).Data(j))
                tensor.tensor = np.frombuffer(bytearray(buffer), dtype='<f4').reshape(tensor.shape)
                tensor.tensorType = 0
            self.IR.Tensor = tensor
            
    def _fillOperators(self):
        for i in range(self.subgraph.OperatorsLength()):
            opcode = self.model.OperatorCodes(self.subgraph.Operators(i).OpcodeIndex()).BuiltinCode()
            
            if opcode == 3 and self.subgraph.Operators(i).BuiltinOptionsType() == BuiltinOptions.BuiltinOptions().Conv2DOptions:
                options = Conv2DOptions.Conv2DOptions()
                options.Init(self.subgraph.Operators(i).BuiltinOptions().Bytes, self.subgraph.Operators(i).BuiltinOptions().Pos)
                op = Conv2DOperatorInterface()

                op.option["strideW"] = options.StrideW()
                op.option["strideH"] = options.StrideH()
                op.option["dilationW"] = options.DilationWFactor()
                op.option["dilationH"] = options.DilationHFactor()
                op.option["fusedActivationCode"] = options.FusedActivationFunction()
                op.option["paddingCode"] = options.Padding()

            elif opcode == 17 and self.subgraph.Operators(i).BuiltinOptionsType() == BuiltinOptions.BuiltinOptions().Pool2DOptions:
                options = Pool2DOptions.Pool2DOptions()
                options.Init(self.subgraph.Operators(i).BuiltinOptions().Bytes, self.subgraph.Operators(i).BuiltinOptions().Pos)
                op = MaxPool2DOperatorInterface()

                op.option["strideW"] = options.StrideW()
                op.option["strideH"] = options.StrideH()
                op.option["filterW"] = options.FilterWidth()
                op.option["filterH"] = options.FilterHeight()
                op.option["fusedActivationCode"] = options.FusedActivationFunction()
                op.option["paddingCode"] = options.Padding()

            elif opcode == 22 and self.subgraph.Operators(i).BuiltinOptions() is None:
                op = ReshapeOperatorInterface()
            elif opcode == 9 and self.subgraph.Operators(i).BuiltinOptionsType() == BuiltinOptions.BuiltinOptions().FullyConnectedOptions:
                options = FullyConnectedOptions.FullyConnectedOptions()
                options.Init(self.subgraph.Operators(i).BuiltinOptions().Bytes, self.subgraph.Operators(i).BuiltinOptions().Pos)
                op = FullyConnectedOperatorInterface()

                op.option["fusedActivationCode"] = options.FusedActivationFunction()

                if options.WeightsFormat() != 0 or options.KeepNumDims() or options.AsymmetricQuantizeInputs():
                    raise RuntimeError("Operator not supported")
            elif opcode == 25 and self.subgraph.Operators(i).BuiltinOptionsType() == BuiltinOptions.BuiltinOptions().SoftmaxOptions:
                options = SoftmaxOptions.SoftmaxOptions()
                options.Init(self.subgraph.Operators(8).BuiltinOptions().Bytes, self.subgraph.Operators(8).BuiltinOptions().Pos)
                op = SoftmaxOperatorInterface()

                op.option["beta"] = options.Beta()
            else:
                op = OperatorInterface()
            
            op.opcode = opcode
            op.name = BuiltinOperatorEnumDict[self.model.OperatorCodes(self.subgraph.Operators(i).OpcodeIndex()).BuiltinCode()] + "_" + str(i)
            op.inputTensors = [self.subgraph.Operators(i).Inputs(j) for j in range(self.subgraph.Operators(i).InputsLength())]
            op.outputTensors = [self.subgraph.Operators(i).Outputs(j) for j in range(self.subgraph.Operators(i).OutputsLength())]

            self.IR.Operator = op

if __name__ == "__main__":
    modelIR = ModelIR()
    buffer = file2Buffer("./bin/model.tflite")
    modelHelperTFLite = ModelHelperTFLite(buffer, modelIR)
    print(modelIR)
    