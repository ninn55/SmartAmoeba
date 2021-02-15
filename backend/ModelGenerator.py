from ModelParserTFLite import *

class LayerGeneratorInterface(object):
    def __init__(self):
        self.options = {}
    
    def layerTemplate(self):
        raise RuntimeError("Not implemented")

class FusedConv2DGeneratorInterface(LayerGeneratorInterface):
    """FusedConv2DReLuGeneratorInterface
    Low level IR options for fused conv2d and relu layer
    """
    def __init__(self):
        self.options = {\
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
    
    def layerTemplate(self):
        if self.options["fusedActivationCode"] == 0 and self.options["paddingCode"] == 1 and \
            (self.option["strideW"], self.option["strideH"], self.option["dilationW"], self.option["dilationH"]) == (0, 0, 0, 0):
            return """
W = <<inputWidth>>;
H = <<inputHeight>>;
def_type *in_ts = (def_type*)input_tensor;
unsigned OW, OH, ICH, OCH, KS;
KS = <<kernelSize>>;
OW = W - KS + 1;
OH = H - KS + 1;
ICH = <<inputChannel>>;
OCH = <<outputChannel>>;
def_type *out; // = buffer;
GET_BUFFER_ADDR(out);
conv_2d_NO_PADDING_NO_DILATE_NOSTRIDE(in_ts, out, Conv2D1, Conv2D1_bias, W, H, ICH, OCH, KS);
        """

        elif self.options["fusedActivationCode"] == 1 and self.options["paddingCode"] == 1 and \
            (self.option["strideW"], self.option["strideH"], self.option["dilationW"], self.option["dilationH"]) == (0, 0, 0, 0):
            return """
W = <<inputWidth>>;
H = <<inputHeight>>;
def_type *in_ts = (def_type*)input_tensor;
unsigned OW, OH, ICH, OCH, KS;
KS = <<kernelSize>>;
OW = W - KS + 1;
OH = H - KS + 1;
ICH = <<inputChannel>>;
OCH = <<outputChannel>>;
def_type *out; // = buffer;
GET_BUFFER_ADDR(out);
conv_2d_NO_PADDING_NO_DILATE_NOSTRIDE(in_ts, out, Conv2D1, Conv2D1_bias, W, H, ICH, OCH, KS);

W = OW;
H = OH;
relu(out, W, H, OCH);
        """

        else:
            raise RuntimeError("Not supported")


class FusedMaxPool2DGeneratorInterface(LayerGeneratorInterface):
    def __init__(self):
        self.options = {\
            "inputWidth" : -1, \
            "inputHeight": -1, \
            "inChannel" : 0, \
            "strideW" : 0, "strideH" : 0, \
            "filterW" : 0, "filterH" : 0, \
            "fusedActivationCode" : -1, \
            "paddingCode" : -1 \
        }

class FullyConnectedGeneratorInterface(LayerGeneratorInterface):
    def __init__(self):
        self.option = {\
            "inputWidth" : -1, \
            "inputHeight": -1, \
            "filterW" : 0, "filterH" : 0
        }

class ReshapeGeneratorInterface(LayerGeneratorInterface):
    def __init__(self):
        self.option = {}

class ModelGenerator(object): 
    def __init__(self, IR):
        self.IR = IR
    
    def 

if __name__ == "__main__":
    pass