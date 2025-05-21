#!/bin/bash

LINE_HEIGHT=18

split_description() {
    local description="$1"
    local max_length="$2"
    local split_desc=""
    local line=""

    for word in $description; do
        # Calculate the new length if we were to add the word
        local new_length=$((${#line} + ${#word}))

        if [ "${#line}" -gt 0 ] && [ "$new_length" -ge "$max_length" ]; then
            split_desc+="${line}\n"
            line=""
        fi
        line+="${word} "
    done
    split_desc+="${line}"

    echo -e "$split_desc"
}



# Path to the JSON file
ID_2_DESC_JSON_FILE="/Users/tinnguyen/Downloads/archive/snake_135_descriptions_ID.json"

ID_2_NAME_JSON_FILE="/Users/tinnguyen/Downloads/archive/snake_135_ID_2_classname.json"

# Path to the parent folder containing subfolders
PARENT_FOLDER="/Users/tinnguyen/Downloads/archive/train"

# Loop through each subfolder in the parent folder
for FOLDER in "$PARENT_FOLDER"/*; do
    if [ -d "$FOLDER" ]; then
        # Extract the folder name
        FOLDER_NAME=$(basename "$FOLDER")
        # echo "/Users/tinnguyen/Downloads/archive/snake_135/$FOLDER_NAME"
        # Replace spaces with underscores for JSON parsing (if needed)
        JSON_KEY="${FOLDER_NAME// /_}"

        # Get the description for the folder from the JSON file
        DESCRIPTION=$(jq -r ".[\"$JSON_KEY\"]" "$ID_2_DESC_JSON_FILE")

        CLASSNAME=$(jq -r ".[\"$JSON_KEY\"]" "$ID_2_NAME_JSON_FILE")

        # Navigate to the subfolder
        cd "$FOLDER"

        # Concatenate the first 5 images in the subfolder
        magick $(ls | head -5) +append temp_output.jpg

        # Add the description to the concatenated image
        DESCRIPTION=$(split_description "$DESCRIPTION" 220)

        line_count=$(echo "$DESCRIPTION" | wc -l)

        total_height=$(($line_count * $LINE_HEIGHT + 30))
        
        magick temp_output.jpg -gravity South -background White -splice 0x$total_height -pointsize 20 -annotate +0+0 "$CLASSNAME":"$DESCRIPTION" "/Users/tinnguyen/Downloads/archive/snake_135/$FOLDER_NAME.jpg"

        # Remove the temporary concatenated image
        rm temp_output.jpg

        # Move back to the parent directory
        cd "$PARENT_FOLDER"
    fi
done
