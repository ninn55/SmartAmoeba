import os
import binascii

def buf2File(buf: bytearray, filename: str):
    with open(filename, "wb") as f:
        f.write(buf)
    
def file2Buffer(filename: str) -> bytearray:
    with open(filename, "rb") as f:
        return bytearray(f.read())

def toByteBuffer(inputFilename: str, outputFilename: str):
    def convert_to_c_array(bytes) -> str:
        hexstr = binascii.hexlify(bytes).decode("UTF-8")
        hexstr = hexstr.upper()
        array = ["0x" + hexstr[i:i + 2] for i in range(0, len(hexstr), 2)]
        array = [array[i:i+10] for i in range(0, len(array), 10)]
        return ",\n  ".join([", ".join(e) for e in array])
    
    if not os.path.isfile(inputFilename):
        raise FileNotFoundError(inputFilename + "Not Found!")

    tflite_binary = open(inputFilename, 'rb').read()
    ascii_bytes = convert_to_c_array(tflite_binary)
    c_file = "const unsigned char tf_model[] = {\n  " +\
             ascii_bytes +\
             "\n};\nunsigned int tf_model_len = " + \
            str(len(tflite_binary)) + ";"
    open(outputFilename, "w").write(c_file)