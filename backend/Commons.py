from VariableNameHelper import VariableNameHelper
import numpy as np

class ModelIR(object):
    """
    Store ops and tensors as list of object
    Ingereted from tensorflow lite v3.0 schema
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
    """
    A Base class to store 
    """
    def __init__(self):
        self._name = str()
        self.shape = list()
        # Type 0 tensor means const
        # type 1 tensor means var
        # variable tensor's buffer is all 0 but shape infomation is valid
        self.tensorType = -1
        # Indexed into TensorTyperEnumDict
        self.tensorTyperEnum = -1
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

class AdjacencyMatrix(object):
    """
    Transfer IR into a Adjacent Matrix
    The AM is calculated during construct
    First construct then __call__
    """
    def __init__(self, IR = None):
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("requires networkx " "networkx.org") from e

        self.G = nx.DiGraph() 
        # Unified Represention of model
        self.IR = ModelIR() if IR is None else IR
        # Tensor can be found in IR._tensors
        # Ops can be found in IR._ops
        # The index of this array is indexed into IR._ops which is a list of ops instances
        # The value of this array element is indexed into IR._tensors which is a list of Tensor interface
        #
        # Known flaw assumes only one tensor passed from two ops
        # a non-zero element Aij indicates an edge from i to j or
        # A adjacent matrix annotation of the model IR
        self.AM = np.zeros((len(self.IR._ops), ) * 2)
        self._construct()
        self._buildG()

    def _construct(self):
        for i in range(len(self.IR._ops)):
            for j in self.IR._ops[i].inputTensors:
                for k in self.IR.findOpWithOutputIndex(j):
                    if self.AM[k][i] == 0 or self.AM[k][i] == j:
                        self.AM[k][i] = j
                    # self.AM[k][i] != j and self.AM[k][i] != 0
                    else:
                        raise RuntimeError("Not implemented")

            for j in self.IR._ops[i].outputTensors:
                for k in self.IR.findOpWithInputIndex(j):
                    if self.AM[i][k] == 0 or self.AM[i][k] == j:
                        self.AM[i][k] == j
                    else:
                        # self.AM[k][i] != j and self.AM[k][i] != 0
                        raise RuntimeError("Not implemented")

    def _buildG(self):
        for i in range(self.AM.shape[0]):
            self.G.add_node(i, shape = "record", fontname = "Arial",\
                            label = self.IR._ops[i].name)
        for i in range(self.AM.shape[0]): 
            for j in range(self.AM.shape[1]): 
                if self.AM[i][j] > 0: 
                    self.G.add_edge(i,j)
                    # self.G.add_edge(i,j, label = modelIR._tensors[int(self.AM[0][1])]._name)

    def GenerateDot(self) -> str:
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("requires networkx " "networkx.org") from e
        return nx.nx_agraph.to_agraph(self.G).to_string()

    def GenerateImage(self, name = "Common") -> bytes:
        try:
            import graphviz
        except ImportError as e:
            raise ImportError("requires graphviz " "graphviz.readthedocs.io") from e
        return graphviz.Source(self.GenerateDot(), format='png').pipe()

    def __call__(self):
        return self.AM

if __name__ == "__main__":
    from ModelParserTFLite import ModelHighLevelIR
    from FileHelper import file2Buffer
    from ModelParserTFLite import ModelHelperTFLite
    modelIR = ModelHighLevelIR()
    # buffer = file2Buffer("./bin/tinymlperf/aww_ref_model.tflite")
    # buffer = file2Buffer("./bin/tinymlperf/vww_96_float.tflite")
    buffer = file2Buffer("./bin/tinymlperf/pretrainedResnet.tflite")
    # buffer = file2Buffer("./bin/model.tflite")
    modelHelperTFLite = ModelHelperTFLite(buffer, modelIR)
    am = AdjacencyMatrix(modelIR)
    with open("com.dot", "w") as f:
        f.write(am.GenerateDot())
    with open("com.png", "wb") as f:
        f.write(am.GenerateImage())