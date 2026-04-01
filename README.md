# Top-quark Pair Quantum Correlation Measurement

This repository contains the code and documentation for the downstream task: **quantum correlation studies in the $t\bar{t}$ process**.

---

## Installation

```aiignore
git clone https://github.com/EveNet-HEP/TT2L-QC-Study.git
cd TT2L-QC-Study
```

To run the code, it is recommended to use Docker.

### Step 1. Clone EveNet-Full

```bash
git clone --recursive https://github.com/EveNet-HEP/EveNet-Full.git
```

### Step 2. Pull and run the Docker image

```bash
docker pull docker.io/avencast1994/evenet:1.5
docker run --gpus all -it \
  -v /path/to/your/workspace:/workspace/ \
  docker.io/avencast1994/evenet:1.5
```

### Enable Weights & Biases (wandb) logging

```aiignore
export WANDB_API_KEY=[your_wandb_api_key]
```

---

## Data Inputs

All preprocessed datasets are hosted on Hugging Face. Please download them in advance:

Link: https://huggingface.co/datasets/Avencast/EveNet-TT2L-QuantumCorrelation/tree/main

Dataset structure:

- `train/`: training samples  
- `val/`: validation samples (used to monitor training)  
- `test/`: test samples (used for final evaluation)  

---

## Training Process

### Build your training YAML file

After downloading the dataset, update the example YAML file with your local paths:

```bash
YAML_name="<you prefer>"
Save_folder="<Save_folder>"
Data_path="<Data_path>"
Wandb_project="<Your WandB Project Name>" # optional

cp examples/Train_examples.yaml ${YAML_name}.yaml
sed "s|<download_dir>|${Data_path}|g" ${YAML_name}.yaml
sed "s|<Your WandB Project Name>|${Wandb_project}|g" ${YAML_name}.yaml
sed "s|<save_dir>|${Save_folder}|g" ${YAML_name}.yaml
```

### Additional YAML options

1. Modify training hyperparameters

```YAML
options:
    Training:
        total_epochs: <Total epochs for training>
        model_checkpoint_save_top_k: <Number of best checkpoints to save>
        learning_rate: <Learning rate for the generation head>
        learning_rate_body: <Learning rate for the PET body>
```

2. Fine-tune from a pretrained model

```YAML
options:
    Training:
        pretrain_model_load_path: <Your Model Path>
```

3. Limit dataset size (for studies on sample size impact)

```YAML
options:
    Dataset:
        dataset_limit: <Ratio>
```

### Train the model

```bash
python3 EveNet-Full/scripts/train.py <your YAML file>
```

---

## Evaluation Process

### Build your evaluation YAML file

Similar to the training setup:

```bash
YAML_name="<you prefer>"
Save_folder="<Save_folder>"
Data_path="<Data_path>"
ckpt_name="<Your checkpoint name>"

cp examples/Predict_examples.yaml ${YAML_name}.yaml
sed "s|<download_dir>|${Data_path}|g" ${YAML_name}.yaml
sed "s|<save_dir>|${Save_folder}|g" ${YAML_name}.yaml
sed "s|<ckpt_name>|${ckpt_name}|g" ${YAML_name}.yaml
```

### Generate predictions

```bash
python3 EveNet-Full/scripts/predict.py <your YAML file>
```

---

## Unfolding Study

It is recommended to build ROOT from source (see: https://root.cern.ch/install/#build-from-source).  
Otherwise, the unfolding script may fail.

### Install RooUnfold

```aiignore
git clone --recursive ssh://git@gitlab.cern.ch:7999/yulei/RooUnfold.git
```

### Build RooUnfold

```bash
mkdir build
cd build
cmake ..
make -j4
cd ..
source build/setup.sh
```

After building RooUnfold, make sure the `libRooUnfold.dylib` path matches the one specified in `analysis_core/unfold.py`.

### Run the analysis

```bash
python analysis.py <your predict file path>
```

The results will be saved in the `results_<file tag>/` folder.

## Computing Estimation

The experiments in this work were performed on a computing cluster with **NVIDIA A100 40GB GPUs**.

A typical single training run completes in about **2 hours** using 16 GPUs.  
The estimated runtime for different hardware configurations is:

| Hardware Configuration              | Estimated Runtime | Notes |
|-----------------------------------|------------------|------|
| 16 × A100 40GB                    | ~2 hours         | Cluster setup used in this work |
| 1 × A100 40GB                     | ~30–40 hours     | Approximate linear scaling |
| 1 × RTX 4090 (24GB)               | ~2–3 days        | May require smaller batch size |
| 1 × RTX 4080 / 4070 Ti (consumer) | ~3–5 days        | Likely requires gradient accumulation |

It is important to note that this table reflects the cost of **a single training run only**.  
The full study reported in the paper involves **multiple trainings with different configurations and systematic checks**, and therefore requires substantially more total compute.

The actual runtime may vary depending on:

- data loading and I/O performance,
- software environment (CUDA, PyTorch, etc.),
- mixed precision settings,
- batch size and gradient accumulation.

Due to the smaller GPU memory on consumer hardware compared to A100 40GB, reproducing the training may require reducing the per-device batch size, which can further increase the runtime.