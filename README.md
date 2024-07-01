# FCA
Code for the paper: FCA: Full Stack Causal Analysis for Microservice Performance Debugging

	@inproceedings{tunde2024fca,
	title={FCA: Full Stack Causal Analysis for Microservice Performance Debugging},
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

### 2. Create a virtual environment with the needed libraries (using req.txt)
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
  
### Alternatively, find our docker container and enter the container instance  
(steps coming soon)  

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


# Alternatively, run with new data 
Alternatively, run FCA with data from your cluster, you may use:
`./collect_multiple_runs.sh`

### Collect new data
`./collect_multiple_runs.sh`
Be sure to change what script you would like to use to collect the new data.

### Run one experiment
`./run_1_experiment.sh`
This script requires a `data` folder in the same directory.

### Run three experiments for a specific bug
`./run_3_experiments.sh`
This script requires three folders in the same directory. Specific directories are listed in the script. Be sure to change the `dates`, `bugid`, and `folder` variables. It should output a new `res_avg_std_` folder.

### Change to a different analysis mode
#### Change to root cause service/function analysis mode
`ROOT_CAUSE_FUNCTION_ANALYSIS` in the `settings.py`:  
`True`: root cause function analysis.  
`False`: root cause service analysis.

### Change to FCA mode (default) / MI + Anomaly detection (but no Alignment) / the pure mode (no Anomaly detection or Alignment)
`ANOMALY_DETECTION_MODEL` and `ALIGN_ANOMALY` in the `calculate_correlation.py`

|         | ANOMALY_DETECTION_MODEL | ALIGN_ANOMALY |
|---------|-------------------------|---------------|
| FCA (default)    | "som"            | True          |
| MI + AD (but no Alignment) | "som"            | False         |
| Pure (no Anomaly detection or Alignment)    | None                    | False         |
