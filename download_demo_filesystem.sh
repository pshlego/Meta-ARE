#!/bin/bash
# Alternative script to download demo filesystem using git
# This avoids the need for huggingface_hub Python package

set -e  # Exit on any error

LOCAL_DIR="./local_demo_filesystem"
REPO_URL="https://huggingface.co/datasets/meta-agents-research-environments/gaia2_filesystem"

echo "Downloading demo filesystem from Hugging Face..."
echo "Repository: $REPO_URL"
echo "Local directory: $LOCAL_DIR"

# Create local directory
mkdir -p "$LOCAL_DIR"

# Use git to clone only the demo_filesystem folder
echo "Cloning repository..."
if command -v git &> /dev/null; then
    # Use sparse checkout to get only the demo_filesystem folder
    cd "$LOCAL_DIR"
    git init
    git remote add origin "$REPO_URL"
    git config core.sparseCheckout true
    echo "demo_filesystem/*" > .git/info/sparse-checkout
    git pull origin main
    
    # Move demo_filesystem contents to root
    if [ -d "demo_filesystem" ]; then
        mv demo_filesystem/* .
        rmdir demo_filesystem
        # Clean up git files
        rm -rf .git
        echo "✓ Demo filesystem downloaded successfully!"
        echo ""
        echo "To use this local filesystem, set the environment variable:"
        echo "export DEMO_FS_PATH=\"$(pwd)\""
        echo ""
        echo "Or create a .env file with:"
        echo "DEMO_FS_PATH=\"$(pwd)\""
    else
        echo "❌ demo_filesystem folder not found in repository"
        exit 1
    fi
else
    echo "❌ Git is not installed. Please install git or use the Python script instead."
    exit 1
fi