#!/bin/bash

JSON_FILE="/Users/tinnguyen/Downloads/archive/snake_135_descriptions.json"

# Read each key (species name) and its description
while IFS= read -r key; do
    # Extract the description for the key
    description=$(jq -r ".[\"$key\"]" "$JSON_FILE")

    # Echo the key and its description
    echo "Key: $key"
    echo "Description: $description"
done < <(jq -r 'keys[]' "$JSON_FILE")