#!/bin/bash

read -p "Enter the folder path containing the .yaml files: " folder_path

read -p "Enter the value you want to set for 'lr': " new_value

for file in "$folder_path"/*.yaml; do
    sed -i "s/^\(\s*lr:\s*\).*/\1 $new_value/" "$file"
done

echo "All config files have been updated with lr = $new_value."
