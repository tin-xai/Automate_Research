#!/bin/bash

# Define input (Folder A) and output (Folder B) directories
INPUT_DIR="/Users/tinnguyen/Downloads/2025/guitar/guitar_w_Q"
OUTPUT_DIR="/Users/tinnguyen/Downloads/2025/guitar/480_guitar_videos"

# Loop through all video files in Folder A
for file in "$INPUT_DIR"/*; do
    # Get the filename without the path
    filename=$(basename -- "$file")

    # Define output file path in Folder B
    output_file="$OUTPUT_DIR/$filename"

    # Convert video to 720p using hardware acceleration (Apple M1/M2)
    ffmpeg -i "$file" -vf "scale=1280:720" -c:v h264_videotoolbox -b:v 2500k -c:a aac "$output_file"

    echo "Converted: $file → $output_file"
    # delete input file
    # rm "$file"
done

echo "✅ All videos have been converted and saved in '$OUTPUT_DIR'"
