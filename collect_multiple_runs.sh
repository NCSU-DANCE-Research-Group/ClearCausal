#!/bin/bash

# Define the number of times to run the commands
num_runs=30
# Bug ID
bug_id=2

# Loop to run the commands for the specified number of times
for ((i=1; i<=$num_runs; i++))
do
    # ./run_mediamicroservices.sh # bug 3, 7, 9
    ./run_socialnetwork.sh # bug 2, 8
    # ./run_onlineboutique.sh # bug 1, 10
    # Rename the data folder, append the bug id, current timestamp in seconds to the folder name
    mv data data_B${bug_id}_$(date +"%s")
    # Rename the image folder, append the bug id, current timestamp in seconds to the folder name
    # mv image image_B${bug_id}_$(date +"%s") # we don't need the image anymore
done
