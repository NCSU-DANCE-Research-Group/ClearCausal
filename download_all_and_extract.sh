#!/bin/bash

# Base URL
base_url="https://webresources.aaaab3n.moe/share/acsos"

# Download and unzip each file
for i in {1..10}; do
  file_name="B${i}.zip"
  wget "${base_url}/${file_name}"
  unzip "${file_name}"
  rm "${file_name}" # Optionally remove the zip file after unzipping
done
