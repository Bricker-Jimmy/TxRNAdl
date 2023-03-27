T3RNAdl / T4RNAdl
===

[Home Page](http://www.szu-bioinf.org/)

# Pretrain model Preparation

Download `DNABERT6` from: [Github](https://github.com/jerryji1993/DNABERT)

Extract from zip:

```tree
/pathto/dnabert6/
├── config.json
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt
```

Change `model_path > dnabert6 > huggingface` to your dnabert6 folder path (`/pathto/dnabert6/`) in `config/config.yml`.

# Install CUDA

For More detail, Visit [NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

# Train Model

Firstly change directory into `TxRNAdl` folder:

```bash
$ cd TxRNAdl
$ ls
# config  data  out  README.md  src
```

Install python required package:

```bash
$ pip3 install -r config/requirements.txt
```

Run Python Script:

```bash
# T3
# Train model
$ python3 -u src/T3/LMs.py
# Evaluate Model
$ python3 -u src/T3/performance_test.py

# T4
# Train model
$ python3 -u src/T4/LMs_t4.py
# Evaluate Model
$ python3 -u src/T4/performance_test.py

```

