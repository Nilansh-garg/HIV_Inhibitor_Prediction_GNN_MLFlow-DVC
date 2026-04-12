# HIV_Inhibitor_Prediction_GNN_MLFlow-DVC
This project implements a GNN-based MLOps pipeline using Graph Attention Networks and PyTorch Geometric to classify HIV inhibitors from molecular structures. It integrates DVC and MLflow within a structured CCDS format to ensure reproducible, production-ready drug discovery and experiment tracking.

# HIV_Inhibitor_Prediction_GNN_MLFlow-DVC

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml


# How to run?
### STEPS:

Clone the repository

```bash
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n HIV_Inhibitor python=3.8 -y
```

```bash
conda activate HIV_Inhibitor
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


