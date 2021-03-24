from ModelParserTFLite import *
from ModelGenerator import *

class ModelConverter(object):
    def __init__(self, lowlevelIR: ModelIR, highlevelIR: ModelGenerator):
        self.lowlevelIR = lowlevelIR
        self.highlevelIR = highlevelIR
    
    def convert(self) -> int:
        pass

    def __call__(self):
        return convert

    @property
    def lowlevelIR(self):
        return self.lowlevelIR

    @property
    def highlevelIR(self):
        return self.highlevelIR

if __name__ == "__main__":
    pass