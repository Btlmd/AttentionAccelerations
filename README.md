# Attention Accelerations

<center>
<image src="lra.png" width="80%" />
</center>

This repo provides code for the course project #7 of Numerical Analysis, THU-CST, 2023 Spring. 

On Long Range Arena, we tried to reproduce results in [Skyformer](https://arxiv.org/abs/2111.00035), [CosFormer](https://arxiv.org/abs/2202.08791), [LARA](https://arxiv.org/abs/2204.04667) and [MEGA](https://arxiv.org/abs/2209.10655). We also measured the training speed and inference speed of these models.

## Data Preparation

- Download Preprocessed data from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/76489e9a0b154692a502/)

- Unzip `lra_data_mega.zip` and `lra_skyformer.zip` and make the directory structure as follows:

```
data/skyformer
├── lra-image.dev.pickle
├── lra-image.test.pickle
├── lra-image.train.pickle
├── ...
├── lra-text.dev.pickle
├── lra-text.test.pickle
└── lra-text.train.pickle
data/mega
├── aan
│   ├── dict-bin
│   ├── label-bin
│   ├── src-bin
│   └── src1-bin
├── cifar10
│   ├── input
│   └── label
├── imdb-4000
│   ├── label-bin
│   └── src-bin
├── listops
│   ├── label-bin
│   └── src-bin
├── path-x
│   ├── input
│   └── label
└── pathfinder
    ├── input
    └── label
```

## Installation

Prepare the environment by

```bash
conda create -n acce python=3.8
conda activate acce

# install `torch==1.8.0` follow your CUDA version, e.g.
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# install skyformer dependencies
pip install -r skyformer/requirements.txt

# install mega and its dependencies
pip install -e mega
```

## Run

### Training & Inference

- CosFormer, LARA and Skyformer

    ```bash
    cd skyformer
    python main.py --mode train --attn <attention-name> --task <task-name>
    ```

    - `<attention-name>`:
        - `softmax`: baseline attention
        - `skyformer`
        - `cosformer`
        - `lara`

    - `<task-name>`: 
        - `lra-listops`
        - `lra-pathfinder`
        - `lra-retrieval`
        - `lra-text`
        - `lra-image`

- MEGA

    ```bash
    cd mega
    bash training_scripts/run_<task-name>.sh
    ```

    - `<task-name>`: 
        - `listops`
        - `pathfinder`
        - `retrieval`
        - `text`
        - `image`

- The scripts select the best checkpoint on vavlidation set and evaluate on test set at the end of training.

### Speed Test

- CosFormer, LARA and Skyformer

    ```bash
    cd skyformer
    bash speed_tests.sh
    ```

    It runs speed tests for all `softmax`, `skyformer`, `cosformer` and `lara` on all 5 tasks.

- MEGA

    ```bash
    cd mega
    bash timing_scripts/speed_tests.sh
    ```

   It runs speed tests for all `MEGA-∞` and `MEGA-128` on all 5 tasks.

## Acknowledgement and Refernce

This repo is derived from [Skyformer](https://github.com/pkuzengqi/Skyformer) and [MEGA](https://github.com/facebookresearch/mega), with implementation refernce from [CosFormer](https://github.com/OpenNLPLab/cosFormer) and [LARA](https://github.com/HKUNLP/efficient-attention). We thank the authors for their great work.