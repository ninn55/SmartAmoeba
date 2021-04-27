from VariableNameHelper import VariableNameHelper
from collections import defaultdict
import numpy as np
MAX_STRING_LEN_GRAPH = 20

class ModelIR(object):
    """
    Store ops and tensors as list of object
    Ingereted from tensorflow lite v3.0 schema
    """
    def __init__(self):
        self._ops = list() # List of OperatorInterface
        self._tensors = list() # List of numpy array
        # List of indexes into _ops
        # denote the computation order
        self._order = list()
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
        
        # Unified Represention of model
        self.IR = ModelIR() if IR is None else IR

        # Another graph representation
        # As a dictionary, keys are int values are list of int
        # both keys and values are indexed into IR._ops 
        # The index of this array is indexed into IR._ops which is a list of ops instances
        #
        # defaultdict can access all potential keys since type of dictionary value is set
        # A adjacent matrix annotation of the model IR
        self.AM = defaultdict(list)
        # Vertices count
        self.VCount = len(self.IR._ops)
        self.computeOrder = []
        # Directed graph in networkx
        self.G = nx.DiGraph()

        self._construct()
        self._sort()
        self.IR._order = self.computeOrder
        self._buildG()

    def _construct(self):
        for i in range(self.VCount):
            for j in self.IR._ops[i].outputTensors:
                for k in self.IR.findOpWithInputIndex(j):
                    if k not in self.AM[j]:
                        self.AM[i].append(k)

    def _sort(self):
        # DFS Topological Sorting for DAG
        # Algorithm reference wikipedia.org/wiki/Topological_sorting#Depth-first_search
        temporaryMark = defaultdict(bool)
        permanentMark = defaultdict(bool)
        L = []

        def visit(self, i: int):
            nonlocal temporaryMark, permanentMark, L
            if permanentMark[i]:
                return
            if temporaryMark[i]:
                raise RuntimeError("Not a DAG")
            
            temporaryMark[i] = True
            for j in self.AM[i]:
                visit(self, j)
            temporaryMark[i] = False
            permanentMark[i] = True
            L.append(i)
            return
        
        visit(self, 0)
        self.computeOrder = L[::-1]

    def _buildG(self):
        for i in range(self.VCount):
            self.G.add_node(i, shape = "record", fontname = "Arial",\
                            label = "Op Name: %(name)s\\n\
                                     Compute Order: %(order)s\
                                    " \
                                % {"name": self.IR._ops[i].name,
                                    "order": self.computeOrder.index(i) + 1 
                                },\
                            xlabel = "Input Tensors: \\n%(inputs)s\\nOutput Tensors: \\n%(outputs)s\\n" % {
                                "inputs": "\\n".join([j[:MAX_STRING_LEN_GRAPH] + " ... ..." if len(j) > MAX_STRING_LEN_GRAPH else j for j in [self.IR._tensors[j]._name for j in self.IR._ops[i].inputTensors]]),
                                "outputs": "\\n".join([j[:MAX_STRING_LEN_GRAPH] + " ... ..." if len(j) > MAX_STRING_LEN_GRAPH else j for j in [self.IR._tensors[j]._name for j in self.IR._ops[i].outputTensors]])
                            }
                            )
        
        for i in self.AM.keys():
            for j in self.AM[i]:
                # Assume there are only one tensor between two operations
                self.G.add_edge(i, j, label = " X ".join([str(k) for k in self.IR._tensors[self.IR.findTensorFromOPtoOP(i, j)[0]].shape]))

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
        
        return  graphviz.Source(self.GenerateDot(), format='png').pipe()

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