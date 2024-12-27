#!/bin/bash -e

# Initialize an empty array to hold all JSON objects
merged_json='[]'
prefix=$1

# Loop through all JSON files in the current directory
for file in ${prefix}_*.json; do
  # Merge the contents of each JSON file with the existing data
  echo "file ${file}"
  merged_json=$(jq -s '.[0] + .[1]' <(echo "$merged_json") "$file")
done

# Save the merged JSON to a new file
echo "$merged_json" > ${prefix}.json

echo "All JSON files have been merged into ${prefix}.json"

