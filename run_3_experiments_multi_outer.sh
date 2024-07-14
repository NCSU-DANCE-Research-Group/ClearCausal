now=$(date)

# 1. bug IDs - choose bugs
bug_ids=("1")  # ("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
num_bugs=$((${#bug_ids[@]} - 1))

# 2. choose service or function analysis
# True: perform root cause function analysis, False: perform root cause service analysis
function_mode="False"

# 3. update task in run_3_experiments_multi.sh which is used to name exp folders 
task_name="fca_service" # mi mi_ad microscope fca service function 


# Line number in settings.py to modify
setting_line_bug=5
setting_line_function_mode=16

# Iterate for each bug ID
for i in $(seq 0 $num_bugs)
do
    bug_id="${bug_ids[i]}"
    
    # Change line 5 in setting.py to look like 'BUGID = bug_id'
    sed -i "${setting_line_bug}s/.*/BUGID = $bug_id/" setting.py
	
    # Change line 16 in setting.py to look like 'ROOT_CAUSE_FUNCTION_ANALYSIS = function_mode'
    sed -i "${setting_line_function_mode}s/.*/ROOT_CAUSE_FUNCTION_ANALYSIS = $function_mode/" setting.py

    # Run experiment for bug_id
    ./run_3_experiments_multi.sh "$bug_id" "$task_name"   
    echo "run_3_experiments_multi.sh $bug_id $task_name"
done
