# uAISS

A tinyML inference stack.

## update to new Flatbuffer Schema

Models in tflite format is generated from Flatbuffer format with specific schema:

First download/update newest version of schema file from official tensorflow repository [here](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs) with

```bash
# (⌒_⌒;)
wget -P backend https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
```

Then download flatc compiler from official git repo

```bash
wget https://github.com/google/flatbuffers/releases/download/v2.0.0/Linux.flatc.binary.clang++-9.zip
unzip Linux.flatc.binary.clang++-9.zip
chmod +x flatc
./flatc -v
```

Generate python lib files from flatbuffer schema

```bash
flatc --python -o .\backend\interface  .\backend\schema.fbs
```
If you see error message when running flatbuffer compiler:

```
/usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found
```

You can fixed it with the following step:

First Verify `GLIBCXX_3.4.26` is not in system so file
```bash
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```

If not, update libc++ to the newest version
```bash
add-apt-repository ppa:ubuntu-toolchain-r/test 
apt update
apt install libstdc++6
```

## Graph Generation

In order to Generate dot files of the represented graph you need package `networkx` and `pygraphviz`.
To Generate png file `graphviz` is needed. 
To install them on ubuntu by

```bash
sudo apt update && sudo apt install -y graphviz graphviz-dev
python3 -m pip install networkx pygraphviz graphviz
```
