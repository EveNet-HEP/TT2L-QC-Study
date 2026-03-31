# Top-quark Pair Quantum Correlation Measurement

This repository contains the code and documentation for the downstream task: **quantum correlation studies in the $t\bar{t}$ process**.

---

## Installation

```aiignore
git clone https://github.com/EveNet-HEP/TT2L-QC-Study.git
cd TT2L_QE-Study
```

To run the code, it is recommended to use Docker.

### Step 1. Pull and run the Docker image

```bash
docker pull docker.io/avencast1994/evenet:1.5
docker run --gpus all -it \
  -v /path/to/your/data:/workspace/data \
  docker.io/avencast1994/evenet:1.5
```

### Step 2. Clone the code

```bash
git clone --recursive https://github.com/EveNet-HEP/EveNet-Full.git
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
sed 's/<download_dir>/${Data_path}/g' ${YAML_name}.yaml
sed 's/<Your WandB Project Name>"/${Wandb_project}/g' ${YAML_name}.yaml
sed 's/<save_dir>"/${Save_folder}/g' ${YAML_name}.yaml
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
python3 EveNet_Full/scripts/train.py <your YAML file>
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
sed 's/<download_dir>/${Data_path}/g' ${YAML_name}.yaml
sed 's/<save_dir>"/${Save_folder}/g' ${YAML_name}.yaml
sed 's/<ckpt_name>"/${ckpt_name}/g' ${YAML_name}.yaml
```

### Generate predictions

```bash
python3 EveNet_Full/scripts/predict.py <your YAML file>
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

The results will be saved in the `results/` folder.