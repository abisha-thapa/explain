# Verifying Relational Explanations: A Probabilistic Approach


### Package Requirements
- Python 3.8+
- PyTorch 
- DGL 1.12+
- pandas
- networkx
- scipy
- nimfa
- pgmpy

### Steps to run the project
#### 1. Run the python file genLRApprox.py
```
python genLRApprox.py -dataset_name bashapes -start_rank 100 -end_rank 1000 -inc_rank 100
```
#### Configurable parameters for genLRApprox.py
- dataset_name - Name of the dataset to use
- start_rank - Starting value of the low rank approximation
- end_rank - Ending value of the low rank approximation
- inc_rank - Increment value for the low rank approximation

#### 2. Run the main.py file
```
python main.py -dataset_name bashapes -num_classes 5 -start_rank 100 -end_rank 1000 -inc_rank 100 -exp_limit_nodes 1
```
#### Configurable parameters for main.py
- dataset_name - Name of the dataset to use
- num_classes - Number of class labels in the dataset
- start_rank - Starting value of the low rank approximation
- end_rank - Ending value of the low rank approximation
- inc_rank - Increment value for the low rank approximation
- exp_limit_nodes - Limiting value for number of explanation nodes

