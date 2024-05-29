# FslQNLP

## Overview
This contains the resources and the code to run the meaning comparison task in NLP on a quantum circuit using FSL. Two files are responsible for this, the MC_run.ipynb allows for a single run to verify all the datasets are in order, while the MC_exe.py runs with the presets for various initial seeds and averages. A couple of of things need to be prepared before being able to run any of the file.

## Preparations
The requirement_venv.txt contains all the modules and their versions. Python 3.11.9 is used and encouraged, since the same version of the packages may not exist for other versions of python. The GloVe embeddings or any other embeddings need to be downloaded and saved in the appropriate resource folder. resources\embeddings.