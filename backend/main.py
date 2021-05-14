TEST_MEMPLAN = True

def test_memplan(buffer):
    # Local Dependency
    from ModelParserTFLite import ModelHighLevelIR
    from ModelParserTFLite import ModelHelperTFLite
    from memplan import MemPlaner
    from Commons import (AdjacencyMatrix, \
                        TensorDependencyGraph)
    # Standard Lib Dependency
    from copy import deepcopy
    # Third Party Dependency

    def fakeOperation(op_index: int):
        # running the operation op_index indexed into modelIR._ops
        pass

    modelIR = ModelHighLevelIR()
    modelHelperTFLite = ModelHelperTFLite(buffer, modelIR)
    # fill _order
    am = AdjacencyMatrix(modelIR)
    # fill _tensorNeededByOp
    tdg = TensorDependencyGraph(modelIR)
    # Tested correct
    td = deepcopy(modelIR._tensorNeededByOp)
    mp = MemPlaner()
    allocated = {}
    for i in range(modelIR.OpCount):
        print("Before Allocation: ", dict(td))
        currentOp = modelIR.findOpbyOrder(i)
        # Before Running the Op
        for key, values in td.items():
            if currentOp in values:
                values.remove(currentOp)
                if str(key) not in allocated.keys():
                    allocated[str(key)] = mp.alloc(modelIR._tensors[key].tensorSize)
        print("Before running: ", dict(td))
        # Running Operation
        fakeOperation(currentOp)
        # After Running Operation
        # Clean Tensors no longger needed by any future operation
        for key, values in td.copy().items():
            if len(values) == 0:
                mp.free(allocated[str(key)])
                allocated.pop(str(key))
                td.pop(key)
    # This is a number count, not byte or bit count
    # As in x float32 or x int8
    print("Needed heap size: ", mp.getsize())
    print("This is a number count, not byte or bit count")

if __name__ == "__main__":
    from FileHelper import file2Buffer
    # buffer = file2Buffer("./bin/tinymlperf/aww_ref_model.tflite")
    # buffer = file2Buffer("./bin/tinymlperf/vww_96_float.tflite")
    buffer = file2Buffer("./bin/tinymlperf/pretrainedResnet.tflite")
    # buffer = file2Buffer("./bin/model.tflite")
    if TEST_MEMPLAN:
        test_memplan(buffer)