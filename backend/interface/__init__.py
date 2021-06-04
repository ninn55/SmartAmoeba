def _fix_sys_path():
    # Ref: https://stackoverflow.com/a/62373228
    import os
    import sys
    from interface import tflite as tflite

    TFLITE_FLATBUFFER = os.path.abspath(os.path.dirname(tflite.__file__))
    TFLITE_PROJECT_DIR = os.path.dirname(TFLITE_FLATBUFFER)
    # Add TFLite Package to path
    sys.path.insert(0, TFLITE_PROJECT_DIR)

_fix_sys_path()