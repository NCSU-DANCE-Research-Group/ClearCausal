# ClearCausal
Welcome to the code artifact for the paper: `ClearCausal: Cross Layer Causal Analysis for Automatic Microservice Performance Debugging` which presents a new dependency-aware cross-layer causal analysis system that achieves fine-grained function-level root cause localization for microservice systems.

	@inproceedings{tunde2024clearcausal,
	title={ClearCausal: Cross Layer Causal Analysis for Automatic Microservice Performance Debugging},
	author={Tunde-Onadele, Olufogorehan and Qin, Feiran and Gu, Xiaohui and Lin, Yuhang},
	booktitle={2024 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS)},
	year={2024},
	organization={IEEE}
	}

# Getting Started
Run the paper demo
## 1. Prepare the Environment

### 1-1. Install Docker
Install Docker for your [platform](https://docs.docker.com/engine/install/).

### 1-2. Build and Run the Docker Image
```shell
docker build -t clearcausal .
docker run -it clearcausal
poetry shell
```

### 1-3. Prepare the Dataset
Ensure you have enough space (~2GB).

#### Automated Download
```shell
chmod +x download_all_and_extract.sh
bash download_all_and_extract.sh
```

#### Manual Download
Download and unzip the dataset from [Zenodo](https://zenodo.org/records/13208928).

Example for B1.zip:
```shell
unzip B1.zip
```  
After the unzip, the bug file folder contents (e.g data_B1_21/) should be in the ClearCausal root directory.  

<!-- #### Create a Virtual Environment
Check your Python version (e.g., python3.8).
```shell
python3 -m pip --version
```
Install dependencies:
```shell
sudo apt install python3.8-venv
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
python3 -m pip install -r req.txt
```

Alternatively, use `poetry`:
```shell
curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry shell
``` -->

## 2. Modify the Run Script
The provided script runs experiments automatically: `./run_3_experiments_multi_outer.sh`.

### 2-1. Choose Bug IDs
Edit the script to select the bugs whose data you have downloaded.
```shell
vim run_3_experiments_multi_outer.sh
```  
On line 4, provide a space-separated list of bug IDs, e.g.:  
```
bug_ids=("1" "2")
```

### 2-2. Choose Root Cause Analysis Mode
Select the analysis mode (`True` for function analysis, `False` for service analysis). On line 9, set the `function_mode` variable:
```shell
function_mode="False"
```

## 3. Run the Script
```shell
./run_3_experiments_multi_outer.sh
```
The script generates result folders for each bug, prefixed with `res_avg_std`. By default, the `combined_correlation_results_wdependency` file contains the ClearCausal results.  

  
# Additional Options
We provide more details in the [wiki](https://github.com/NCSU-DANCE-Research-Group/ClearCausal/wiki), including steps to customize the artifact for running alternative approaches or new data. 
