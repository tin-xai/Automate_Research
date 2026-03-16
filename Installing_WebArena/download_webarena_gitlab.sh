#!/usr/bin/env bash
set -e

# Create workspace
WORKDIR="$HOME/webarena_env/images"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "Downloading WebArena GitLab environment image..."
echo "This file is large (~10+ GB), so it may take a while."

URL="http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar"
FILE="gitlab-populated-final-port8023.tar"

# Download with resume support
if command -v wget >/dev/null 2>&1; then
    wget -c "$URL" -O "$FILE"
elif command -v curl >/dev/null 2>&1; then
    curl -L --continue-at - "$URL" -o "$FILE"
else
    echo "Error: Neither wget nor curl is installed."
    exit 1
fi

echo ""
echo "Download complete."
echo "File location: $WORKDIR/$FILE"
echo ""
echo "You can verify size with:"
echo "ls -lh $FILE"
