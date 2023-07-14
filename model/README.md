# model

## Environment

1. First, make sure your conda environment is updated:

```shell
conda update --all
```


2. Create a new environment with Python 3.9:
```shell
conda create -n new_env python=3.9
```

3. Activate the new environment:
```shell
conda activate new_env
```

4. (Optional, since we have installed wrong versions) 
Uninstall the current versions of torch, torchvision, and torchtext:

```shell
pip uninstall torch torchvision torchtext
```

5. Install the compatible versions of torch, torchvision, and torchtext:

```shell
pip install torch==1.7.1 torchvision==0.8.2 torchtext==0.8.1
```

6. Verify the installations:
```shell
python -c "import torch, torchvision, torchtext; print(torch.__version__, torchvision.__version__, torchtext.__version__)"
```


## prepare_data_feature1.py

