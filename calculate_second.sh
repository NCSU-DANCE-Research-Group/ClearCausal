#!/bin/bash

# Specify the target time in "YYYY-MM-DD HH:MM:SS" format
target_time="2023-07-24 20:55:31"

# Convert the target time to Unix timestamp
target_timestamp=$(date -d "$target_time" +%s)

# Get the current timestamp
current_timestamp=$(date +%s)

# Calculate the time difference in seconds and consider the time zone
time_diff=$((target_timestamp - current_timestamp + 900 - 4 * 3600))

# Display the result
echo "Seconds from now to $target_time: $time_diff seconds"
