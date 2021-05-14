# uAISS

A tinyML inference stack.

## Schema reload

Generate python lib files from flatbuffer schema
flatc --python -o .\backend\ .\backend\schema.fbs

## Graph Generation

In order to Generate dot files of the represented graph you need package `networkx` and `pygraphviz`.
To Generate png file `graphviz` is needed. 
To install them on ubuntu by

```bash
sudo apt update && sudo apt install -y graphviz graphviz-dev
python3 -m pip install networkx pygraphviz graphviz
```
