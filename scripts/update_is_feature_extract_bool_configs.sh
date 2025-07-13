#!/bin/bash

read -p "Enter the folder path containing the .yaml files: " folder_path

read -p "Enter the value you want to set for 'is_feature_extract': " new_value

for file in "$folder_path"/*.yaml; do
    sed -i "s/^\(\s*is_feature_extract:\s*\).*/\1 $new_value/" "$file"
done

echo "All config files have been updated with is_feature_extract = $new_value."
