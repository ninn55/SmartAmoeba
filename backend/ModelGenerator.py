from ModelParserTFLite import *
import re
from abc import ABC, abstractmethod

class LayerGeneratorInterface(object, ABC):
    optionsTemplate = {}
    def __init__(self, option, name, isInputLayer = False, isOutputLayer = False):
        self.options = option
        # Special options
        self.name = name
        self.isInputLayer = isInputLayer
        self.isOutputLayer = isOutputLayer
    
    @abstractmethod
    def _layerInitiateTemplate(self):
        raise RuntimeError("Not implemented")
    
    @abstractmethod
    def _layerCalculateTemplate(self):
        raise RuntimeError("Not implemented")
    
    @abstractmethod
    def _layerProducer(self, template: str) -> str:
        # Takes template and layer options and produce final implementation
        raise RuntimeError("Not implemented")

    def layerImplementation(self):
        # call this method after object construction to get the final layer implementation
        template = self._layerCalculateTemplate()
        return self._layerProducer(template)
    
    def __str__(self):
        return "Layer name:" + self.name + "\n"

class FusedConv2DGeneratorInterface(LayerGeneratorInterface):
    """FusedConv2DReLuGeneratorInterface
    Low level IR options for fused conv2d and relu layer
    """
    optionsTemplate = {\
            "inputConvolutionFilterTensorName": None, \
            "inputConvolutionBiasTensorName": None, \
            "inputWidth" : -1, \
            "inputHeight": -1, \
            "kernelSize" : -1, \
            "inputChannel" : -1, \
            "outputChannel" : -1, \
            "paddingCode" : 1, \
            "fusedActivationCode" : 1, \
            "strideW" : 0, "strideH" : 0, \
            "dilationW" : 0, "dilationH" : 0
        }
    def __init__(self, option, name, isInputLayer = False, isOutputLayer = False):
        super(FusedConv2DGeneratorInterface, self).__init__(option, name, isInputLayer, isOutputLayer)        
        
    def _layerCalculateTemplate(self):
        if self.options["fusedActivationCode"] == 0 and self.options["paddingCode"] == 1 and \
            (self.option["strideW"], self.option["strideH"], self.option["dilationW"], self.option["dilationH"]) == (0, 0, 0, 0):
            return """
//<<layerName>>
def_type *out;
GET_BUFFER_ADDR(out);
conv_2d(in_ts, out, <<inputTensorCore>>, <<inputTensorBias>>, <<inputWidth>>, <<inputHeight>>, <<inputChannel>>, <<outputChannel>>, <<kernelSize>>);
        """

        elif self.options["fusedActivationCode"] == 1 and self.options["paddingCode"] == 1 and \
            (self.option["strideW"], self.option["strideH"], self.option["dilationW"], self.option["dilationH"]) == (0, 0, 0, 0):
            return """
//<<layerName>>
def_type *out;
GET_BUFFER_ADDR(out);
fused_conv_2d_relu(in_ts, out, <<inputConvolutionFilterTensorName>>, <<inputConvolutionBiasTensorName>>, <<inputWidth>>, <<inputHeight>>, <<inputChannel>>, <<outputChannel>>, <<kernelSize>>);
        """

        else:
            raise RuntimeError("Not supported")

    def _layerInitiateTemplate(self):
        raise RuntimeError("Not implemented")

    def _layerProducor(self, template: str) -> str:
        if self.options["fusedActivationCode"] == 0 and self.options["paddingCode"] == 1 and \
            (self.option["strideW"], self.option["strideH"], self.option["dilationW"], self.option["dilationH"]) == (0, 0, 0, 0):
            tmp = template
            tmp = re.sub("<<layerName>>", self.name, tmp)
            tmp = re.sub("<<inputConvolutionFilterTensorName>>", self.options["inputConvolutionFilterTensorName"], tmp)
            tmp = re.sub("<<inputConvolutionBiasTensorName>>", self.options["inputConvolutionBiasTensorName"], tmp)
            tmp = re.sub("<<inputWidth>>", self.options["inputWidth"], tmp)
            tmp = re.sub("<<inputHeight>>", self.options["inputHeight"], tmp)
            tmp = re.sub("<<inputChannel>>", self.options["inputChannel"], tmp)
            tmp = re.sub("<<outputChannel>>", self.options["outputChannel"], tmp)
            tmp = re.sub("<<kernelSize>>", self.options["kernelSize"], tmp)
            return tmp
        elif self.options["fusedActivationCode"] == 1 and self.options["paddingCode"] == 1 and \
            (self.option["strideW"], self.option["strideH"], self.option["dilationW"], self.option["dilationH"]) == (0, 0, 0, 0):
            tmp = template
            tmp = re.sub("<<layerName>>", self.name, tmp)
            tmp = re.sub("<<inputConvolutionFilterTensorName>>", self.options["inputConvolutionFilterTensorName"], tmp)
            tmp = re.sub("<<inputConvolutionBiasTensorName>>", self.options["inputConvolutionBiasTensorName"], tmp)
            tmp = re.sub("<<inputWidth>>", self.options["inputWidth"], tmp)
            tmp = re.sub("<<inputHeight>>", self.options["inputHeight"], tmp)
            tmp = re.sub("<<inputChannel>>", self.options["inputChannel"], tmp)
            tmp = re.sub("<<outputChannel>>", self.options["outputChannel"], tmp)
            tmp = re.sub("<<kernelSize>>", self.options["kernelSize"], tmp)
            return tmp
        else:
            raise RuntimeError("Not implemented")


class FusedMaxPool2DGeneratorInterface(LayerGeneratorInterface):
    optionsTemplate = {\
            "inputWidth" : -1, \
            "inputHeight": -1, \
            "inChannel" : 0, \
            "strideW" : 0, "strideH" : 0, \
            "filterW" : 0, "filterH" : 0, \
            "fusedActivationCode" : -1, \
            "paddingCode" : -1 \
        }

    def __init__(self, option, name, isInputLayer = False, isOutputLayer = False):
        super(FusedMaxPool2DGeneratorInterface, self).__init__(option)

    def _layerInitiateTemplate(self):
        raise RuntimeError("Not implemented")

    def _layerCalculateTemplate(self):
        if self.options["strideW"] == self.options["filterW"] and self.options["strideH"] == self.options["filterH"] and \
            self.options["fusedActivationCode"] == 0 and self.options["paddingCode"] == 1:
            return """
//<<layerName>>
in_ts = out;
GET_BUFFER_ADDR(out);
maxpool_2d(in_ts, out, <<inputWidth>>, <<inputHeight>>, <<inputChannel>>, <<PoolSize>>);
            """
        else:
            raise RuntimeError("Not implemented") 

    def _layerProducor(self, template: str) -> str:
        if self.options["strideW"] == self.options["filterW"] and self.options["strideH"] == self.options["filterH"] and \
            self.options["fusedActivationCode"] == 0 and self.options["paddingCode"] == 1:
            tmp = template
            tmp = re.sub("<<layerName>>", self.name, tmp)
            tmp = re.sub("<<inputWidth>>", self.options["inputWidth"], tmp)
            tmp = re.sub("<<inputHeight>>", self.options["inputHeight"], tmp)
            tmp = re.sub("<<inputChannel>>", self.options["inChannel"], tmp)
            tmp = re.sub("<<PoolSize>>", self.options["filterW"], tmp)
            return tmp
        else:
            raise RuntimeError("Not implemented")

class FusedFullyConnectedGeneratorInterface(LayerGeneratorInterface):
    optionsTemplate = {\
            "inputFilterTensorName": None, \
            "inputBiasTensorName": None, \
            "inputWidth" : -1, \
            "inputHeight": -1, \
            "filterW" : 0, "filterH" : 0,\
            "fusedActivationCode" : -1,
        }
    def __init__(self, option, name, isInputLayer = False, isOutputLayer = False):
        super(FusedFullyConnectedGeneratorInterface, self).__init__(option)

    def _layerInitiateTemplate(self):
        raise RuntimeError("Not implemented")

    def _layerCalculateTemplate(self):
        if self.options["fusedActivationCode"] == 0:
            return """
//<<layerName>>
in_ts = out;
GET_BUFFER_ADDR(out);
matmul(in_ts, out, <<inputTensorCore>>, <<inputTensorBias>>, <<inputWidth>>, <<inputHeight>>, <<MatmulFilterWidth>>, <<MatmulFilterHeight>>);
            """
        elif self.options["fusedActivationCode"] == 1:
            return """
//<<layerName>>
in_ts = out;
GET_BUFFER_ADDR(out);
fused_matmul_relu(in_ts, out, <<inputTensorCore>>, <<inputTensorBias>>, <<inputWidth>>, <<inputHeight>>, <<MatmulFilterWidth>>, <<MatmulFilterHeight>>);
            """
        else:
            raise RuntimeError("Not implemented")
        
        def _layerProducor(self, template: str) -> str:
            if self.options["fusedActivationCode"] == 0:
                tmp = template
                tmp = re.sub("<<layerName>>", self.name, tmp)
                tmp = re.sub("<<inputTensorCore>>", self.options["inputFilterTensorName"], tmp)
                tmp = re.sub("<<inputTensorBias>>", self.options["inputBiasTensorName"], tmp)
                tmp = re.sub("<<inputWidth>>", self.options["inputWidth"], tmp)
                tmp = re.sub("<<inputHeight>>", self.options["inputHeight"], tmp)
                tmp = re.sub("<<MatmulFilterWidth>>", self.options["filterW"], tmp)
                tmp = re.sub("<<MatmulFilterHeight>>", self.options["filterH"], tmp)
                return tmp
            elif self.options["fusedActivationCode"] == 1:
                tmp = template
                tmp = re.sub("<<layerName>>", self.name, tmp)
                tmp = re.sub("<<inputTensorCore>>", self.options["inputFilterTensorName"], tmp)
                tmp = re.sub("<<inputTensorBias>>", self.options["inputBiasTensorName"], tmp)
                tmp = re.sub("<<inputWidth>>", self.options["inputWidth"], tmp)
                tmp = re.sub("<<inputHeight>>", self.options["inputHeight"], tmp)
                tmp = re.sub("<<MatmulFilterWidth>>", self.options["filterW"], tmp)
                tmp = re.sub("<<MatmulFilterHeight>>", self.options["filterH"], tmp)
                return tmp
            else:
                raise RuntimeError("Not implemented")
        

class ReLuGeneratorInterface(LayerGeneratorInterface):
    optionsTemplate = {\
            "inputHeight" : -1, \
            "inputWidth": -1, \
            "outputChannel" : -1,
        }
    def __init__(self, option, name, isInputLayer = False, isOutputLayer = False):
        super(ReLuGeneratorInterface, self).__init__(option)

    def _layerInitiateTemplate(self):
        raise RuntimeError("Not implemented")
    
    def _layerCalculateTemplate(self):
        return """
// <<layerName>>
relu(out, <<inputHeight>>, <<inputWidth>>, <<outputChannel>>);
        """

    def _layerProducor(self, template: str) -> str:
        tmp = template
        tmp = re.sub("<<layerName>>", self.name, tmp)
        tmp = re.sub("<<inputHeight>>", self.options["inputHeight"], tmp)
        tmp = re.sub("<<inputWidth>>", self.options["inputWidth"], tmp)
        tmp = re.sub("<<outputChannel>>", self.options["outputChannel"], tmp)
        return tmp

class ReshapeGeneratorInterface(LayerGeneratorInterface):
    optionsTemplate = {\
            "inputWidth" : -1, \
            "inputHeight": -1, \
            "inputChannel" : -1, \
            "outputWidth" : -1, \
            "outputHeight" : -1, \
            "outputChannel" : -1, 
        }
    def __init__(self, option, name, isInputLayer = False, isOutputLayer = False):
        super(FusedMaxPool2DGeneratorInterface, self).__init__(option)

    def _layerInitiateTemplate(self):
        raise RuntimeError("Not implemented")
    
    def _layerCalculateTemplate(self):
        if self.options["outputChannel"] == 1 and (self.options["outputWidth"] == 1 or self.options["outputHeight"] == 1):
            return """
//<<layerName>>
in_ts = out;
GET_BUFFER_ADDR(out);
flatten(in_ts, out, <<inputHeight>>, <<inputWidth>>, <<inputChannel>>);
            """
        else:
            raise RuntimeError("Not implemented")

class TensorGenerator(object): 
    def __init__(self):
        pass

class GraphGenerator(object): 
    def __init__(self):
        pass

class ModelGenerator(object): 
    def __init__(self):
        pass

if __name__ == "__main__":
    pass