#!/bin/bash

# serve model as background process
# wait for server to start (on safe side wait for 5 sec)
ollama serve & sleep 5

# download and setup models
ollama pull marco-o1
echo "marco-o1 pulled ✅"
ollama pull deepseek-r1
echo "deepseek-r1 pulled ✅"


ollama create minicpm-v-2 -f ./Modelfile
echo "minicpm-v-2 pulled ✅"