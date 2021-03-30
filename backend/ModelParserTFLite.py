import tflite
from tflite import Model
from tflite import BuiltinOptions 
from tflite import (Conv2DOptions, 
                    Pool2DOptions, 
                    ReshapeOptions, 
                    FullyConnectedOptions, 
                    SoftmaxOptions, 
                    AddOptions,
                    DepthwiseConv2DOptions)

from FileHelper import file2Buffer
from EnumDictionarys import (ActivationFunctionTypeEnumDict,
                            PaddingEnumDict,
                            TensorTyperEnumDict,
                            BuiltinOperatorEnumDict)
from collections import OrderedDict
from VariableNameHelper import VariableNameHelper

import numpy as np
from Commons import (ModelIR, 
                    TensorInterface)

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

class AddOperatorInterface(OperatorInterface):
    def __init__(self):
        super().__init__()
        self.option = {"fusedActivationCode": -1}
    
    def __str__(self):
        return "Operation: " + self.name + "\n" + \
                "Fused activation functions " + ActivationFunctionTypeEnumDict[self.option["fusedActivationCode"]] + "\n"

class DepthwiseConv2DOperatorInterface(OperatorInterface):
    def __init__(self):
        super().__init__()
        self.option = {\
            "strideW" : 0, "strideH" : 0, \
            "dilationW" : 0, "dilationH" : 0, \
            "fusedActivationCode" : -1, \
            "paddingCode" : -1, \
            "depthMultiplier" : "deprecated in tf 2.1 and above" \
        }

    def __str__(self):
        info  = "Operation: " + self.name + "\n"
        info += "Options for this Conv2D layer are :" + "\n"
        info += "Fused activation functions " + ActivationFunctionTypeEnumDict[self.option["fusedActivationCode"]] + "\n"
        info += "Padding options :" + PaddingEnumDict[self.option["paddingCode"]] + "\n"
        info += "Stride options (w, h):  " + str((self.option["strideW"], self.option["strideH"])) + "\n"
        info += "Dilation options  (w, h): " + str((self.option["dilationW"], self.option["dilationH"])) + "\n"
        return info

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

class AveragePool2DOperatorInterface(OperatorInterface):
    """AveragePool2DOperatorInterface
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
        info += "Options for this AveragePool2D layer are :" + "\n"
        info += "Fused activation functions " + ActivationFunctionTypeEnumDict[self.option["fusedActivationCode"]] + "\n"
        info += "Padding options :" + PaddingEnumDict[self.option["paddingCode"]] + "\n"
        info += "Stride options (w, h):  " + str((self.option["strideW"], self.option["strideH"])) + "\n"
        info += "Filter Size options  (w, h): " + str((self.option["filterW"], self.option["filterH"])) + "\n"
        return info

#--------------------------------
# ModelHighLevelIR
# It's essentially a container for TensorInterface and OperatorInterface
#--------------------------------
class ModelHighLevelIR(ModelIR):
    def __init__(self):
        super(ModelHighLevelIR, self).__init__()

    def getLargestTensorSize(self) -> int:
        largestTensorSize = -1
        for i in range(len(self._tensors)):
            if self._tensors[i] > largestTensorSize:
                largestTensorSize = self._tensors[i]
        return largestTensorSize

    def __str__(self):
        info = ""
#--------------------------------
# Print out all infomation
#--------------------------------
        if True:
            info += "\n\nGraph info:\n\n"
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
#--------------------------------
# Print out all infomation
#--------------------------------
        if False:
            info += "\n\nTensor info:\n\n"
            for i in range(len(self._tensors)):
                info += str(i) + "th Tensor: "
                info += str(self._tensors[i])
        return info
#--------------------------------
# ModelHelperTFLite
# The core job of this 
#--------------------------------
class ModelHelperTFLite(object):
    def __init__(self, modelbuffer, IR = None):
        self.model = tflite.Model.Model.GetRootAsModel(modelbuffer, 0)
        self.subgraph = self.model.Subgraphs(0)
        self.IR = ModelHighLevelIR() if IR is None else IR
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
            # This condition is questionable, may change in future tensorflow version
            # This assumes in the tensorflow lite buffer definition
            # Specific tensor needs to be calculated at runtime is omitted in the buffer definition itself.
            if self.model.Buffers(self.subgraph.Tensors(i).Buffer()).DataLength() == 0:
                tensor.tensor = np.zeros(tensor.shape)
                tensor.tensorType = 1
            else:
                buffer = []
                for j in range(self.model.Buffers(self.subgraph.Tensors(i).Buffer()).DataLength()):
                    buffer.append(self.model.Buffers(self.subgraph.Tensors(i).Buffer()).Data(j))
                # Decode the tensor in IEEE-754 float32 format
                if self.subgraph.Tensors(i).Type() == 0:
                    tensor.tensor = np.frombuffer(bytearray(buffer), dtype='<f4').reshape(tensor.shape)
                    tensor.tensorTyperEnum = 0
                elif self.subgraph.Tensors(i).Type() == 2:
                    tensor.tensor = np.frombuffer(bytearray(buffer), dtype='<i4').reshape(tensor.shape)
                    tensor.tensorTyperEnum = 2
                elif self.subgraph.Tensors(i).Type() == 9:
                    tensor.tensor = np.frombuffer(bytearray(buffer), dtype='<i1').reshape(tensor.shape)
                    tensor.tensorTyperEnum = 9
                else:
                    raise RuntimeError("Not supported buffer format! Got type " + str(TensorTyperEnumDict[self.subgraph.Tensors(i).Type()]) + "!")
                tensor.tensorType = 0
            self.IR.Tensor = tensor
            
    def _fillOperators(self):
        for i in range(self.subgraph.OperatorsLength()):
            opcode = self.model.OperatorCodes(self.subgraph.Operators(i).OpcodeIndex()).BuiltinCode()
            if False:
                # For appearance
                pass
            # Support for operation conv2d
            elif opcode == 3 and self.subgraph.Operators(i).BuiltinOptionsType() == BuiltinOptions.BuiltinOptions().Conv2DOptions:
                options = Conv2DOptions.Conv2DOptions()
                options.Init(self.subgraph.Operators(i).BuiltinOptions().Bytes, self.subgraph.Operators(i).BuiltinOptions().Pos)
                op = Conv2DOperatorInterface()

                op.option["strideW"] = options.StrideW()
                op.option["strideH"] = options.StrideH()
                op.option["dilationW"] = options.DilationWFactor()
                op.option["dilationH"] = options.DilationHFactor()
                op.option["fusedActivationCode"] = options.FusedActivationFunction()
                op.option["paddingCode"] = options.Padding()
            # Support for operation MaxPool2D
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
            # Support for operation Reshape
            elif opcode == 22 and self.subgraph.Operators(i).BuiltinOptions() is None:
                op = ReshapeOperatorInterface()
            # Support for operation FullyConnected
            elif opcode == 9 and self.subgraph.Operators(i).BuiltinOptionsType() == BuiltinOptions.BuiltinOptions().FullyConnectedOptions:
                options = FullyConnectedOptions.FullyConnectedOptions()
                options.Init(self.subgraph.Operators(i).BuiltinOptions().Bytes, self.subgraph.Operators(i).BuiltinOptions().Pos)
                op = FullyConnectedOperatorInterface()

                op.option["fusedActivationCode"] = options.FusedActivationFunction()

                if options.WeightsFormat() != 0 or options.KeepNumDims() or options.AsymmetricQuantizeInputs():
                    raise RuntimeError("Operator not supported")
            # Support for operation Softmax
            elif opcode == 25 and self.subgraph.Operators(i).BuiltinOptionsType() == BuiltinOptions.BuiltinOptions().SoftmaxOptions:
                options = SoftmaxOptions.SoftmaxOptions()
                options.Init(self.subgraph.Operators(i).BuiltinOptions().Bytes, self.subgraph.Operators(i).BuiltinOptions().Pos)
                op = SoftmaxOperatorInterface()

                op.option["beta"] = options.Beta()
            # Support for operation add
            elif opcode == 0 and self.subgraph.Operators(i).BuiltinOptionsType() == BuiltinOptions.BuiltinOptions().AddOptions:
                options = AddOptions.AddOptions()
                options.Init(self.subgraph.Operators(i).BuiltinOptions().Bytes, self.subgraph.Operators(i).BuiltinOptions().Pos)
                op = AddOperatorInterface()

                op.option["fusedActivationCode"] = options.FusedActivationFunction()
            # Support for operation AveragePool2d
            elif opcode == 1 and self.subgraph.Operators(i).BuiltinOptionsType() == BuiltinOptions.BuiltinOptions().Pool2DOptions:
                options = Pool2DOptions.Pool2DOptions()
                options.Init(self.subgraph.Operators(i).BuiltinOptions().Bytes, self.subgraph.Operators(i).BuiltinOptions().Pos)
                op = AveragePool2DOperatorInterface()

                op.option["strideW"] = options.StrideW()
                op.option["strideH"] = options.StrideH()
                op.option["filterW"] = options.FilterWidth()
                op.option["filterH"] = options.FilterHeight()
                op.option["fusedActivationCode"] = options.FusedActivationFunction()
                op.option["paddingCode"] = options.Padding()
            # Support for operation DEPTHWISE_CONV_2D
            elif opcode == 4 and self.subgraph.Operators(i).BuiltinOptionsType() == BuiltinOptions.BuiltinOptions().DepthwiseConv2DOptions:
                options = DepthwiseConv2DOptions.DepthwiseConv2DOptions()
                options.Init(self.subgraph.Operators(i).BuiltinOptions().Bytes, self.subgraph.Operators(i).BuiltinOptions().Pos)
                op = DepthwiseConv2DOperatorInterface()

                op.option["strideW"] = options.StrideW()
                op.option["strideH"] = options.StrideH()
                op.option["dilationW"] = options.DilationWFactor()
                op.option["dilationH"] = options.DilationHFactor()
                op.option["fusedActivationCode"] = options.FusedActivationFunction()
                op.option["paddingCode"] = options.Padding()
            else:
                raise RuntimeError("Operation " + BuiltinOperatorEnumDict[opcode] + " Not supported")
            
            op.opcode = opcode
            op.name = BuiltinOperatorEnumDict[self.model.OperatorCodes(self.subgraph.Operators(i).OpcodeIndex()).BuiltinCode()] + "_" + str(i)
            op.inputTensors = [self.subgraph.Operators(i).Inputs(j) for j in range(self.subgraph.Operators(i).InputsLength())]
            op.outputTensors = [self.subgraph.Operators(i).Outputs(j) for j in range(self.subgraph.Operators(i).OutputsLength())]

            self.IR.Operator = op

if __name__ == "__main__":
    modelIR = ModelHighLevelIR()
    # buffer = file2Buffer("./bin/tinymlperf/aww_ref_model.tflite")
    buffer = file2Buffer("./bin/tinymlperf/vww_96_float.tflite")
    # buffer = file2Buffer("./bin/tinymlperf/pretrainedResnet.tflite")
    # buffer = file2Buffer("./bin/model.tflite")
    modelHelperTFLite = ModelHelperTFLite(buffer, modelIR)
    print(modelIR)
    