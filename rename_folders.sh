#!/bin/bash

index=1
bug=2

for folder in data_B$bug\_*; do
    new_name=data_B$bug\_$index
    echo "Renaming '$folder' to '$new_name'"
    mv "$folder" "$new_name"
    ((index++))
done
