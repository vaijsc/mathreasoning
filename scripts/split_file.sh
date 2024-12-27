#!/bin/bash -e

# Check for correct number of arguments
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 input_file n_splits output_prefix"
  exit 1
fi

input_file=$1 # path to input file
num_parts=$2  # Number of files to split into
output_prefix=$3
cache_dir="cache"

# Count the number of samples in the JSON file
total_samples=$(jq '. | length' "$input_file")

# Calculate the number of samples per file
samples_per_file=$(( (total_samples + num_parts - 1) / num_parts ))

# Split the file into subfiles
for ((i=0; i<num_parts; i++)); do
    start=$((i * samples_per_file))
    end=$((start + samples_per_file))
    # Check if this is the last segment
    if [ $i -eq $((num_parts - 1)) ]; then
        echo "detected last segment"
        end=$total_samples                                   
    fi  
    output_file="${cache_dir}/${output_prefix}_${i}.json"
    
    # Extract the relevant samples and save them to a new file
    jq ".[$start:$end]" "$input_file" > "$output_file"
    echo "Created $output_file with samples $start to $end"
done

