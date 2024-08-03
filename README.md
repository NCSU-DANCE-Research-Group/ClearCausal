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
### 1. Collect data
To run the paper demo, download and unzip the data from [Zenodo](https://zenodo.org/uploads/12602272?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjgxYzIzNTM0LWQ0MzItNDAzOC1hNmZhLWY3OTAzZDNlOWMxNiIsImRhdGEiOnt9LCJyYW5kb20iOiIwZjZhMmRmMGYzZGQzYWJlNjQxODg0OTM1NmQ4MzYwOSJ9.PCbxw2l6IWIoETIkZteCqhtXObniw8yyxNWH8A5xY6RAq-JTD9UlzoZXKc-gx_l5qxus97St47GgfmNQNwCStA)
The dataset is separated by bug ID referenced in the paper.  
For instance, you may download B1.zip and unzip like so: `unzip B1.zip`

### 2-A. Create a virtual environment with the needed libraries (using req.txt)
Check your python3 version (e.g. python3.8)  
```  
python3 -m pip --version  
```  
```  
sudo apt install python3.8-venv && \  
python3 -m pip install --user virtualenv && \  
python3 -m venv env  
```  
```  
source env/bin/activate && \  
python3 -m pip install -r req.txt  
```

### 2-B. Alternatively, use `Poetry` to install the dependencies (Python >= 3.10, using pyproject.toml)
#### 2-B-1. Install Poetry with the [official installer](https://python-poetry.org/docs/#installing-with-the-official-installer)
```shell
curl -sSL https://install.python-poetry.org | python3 -
```
#### 2-B-2. Install the dependencies and enter the python environment
```shell
poetry install
poetry shell
```

### 2-C. Alternatively, find our docker container and enter the container instance  
#### 2-C-1 Install Docker for your [platform](https://docs.docker.com/engine/install/)
#### 2-C-2 Build the Docker Image and Run
```shell
docker build -t clearcausal .
docker run -it -v ~/Downloads:/app/Downloads clearcausal
```

- `-v ~/Downloads:/app/Downloads`: Binds the `~/Downloads` directory from your local machine to the `/app/Downloads` directory inside the container. You can modify the location as needed.   
  
### 3. Modify the run script
We provide a script to run the experiments automatically: `./run_3_experiments_multi_outer.sh`  
```
vi run_3_experiments_multi_outer.sh  
```
#### 3-1. Choose bug IDs to include in the demo. 
Select the bugs whose data you have downloaded as described above.  
In the parenthesis on line 4, provide a space-separated list of bug IDs like so:  
`bug_ids=("1" "2")`  
#### 3-2. Choose the root cause analysis mode  
`True`: root cause function analysis.  
`False`: root cause service analysis.  
On line 9, choose service or function analysis with the `function_mode` variable like so:  
`function_mode = "False"`

### 4. Run the script
```
./run_3_experiments_multi_outer.sh  
```
The script outputs result folders for each bug with names that begin with `res_avg_std`. With the default settings, the `combined_correlation_results_wdependency` file contains the ClearCausal results.  
Each folder contain the following files:
  * `combined_correlation_results_wdependency` - Results with dependency filtering   
  * `combined_correlation_results_wnamespace`  - Results with namespace filtering  
  * `combined_correlation_results_pure` - Results without filtering  
The corresponding files that contain the word 'all' include the results for each repetition of the experiment.  

  
# Additional Options
We provide more details in the [wiki](https://github.com/NCSU-DANCE-Research-Group/ClearCausal/wiki), including steps to customize the artifact for running alternative approaches or new data. 
