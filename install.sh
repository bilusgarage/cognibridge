#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set +e

# ---------------------------------------------------------
# THE FIX: Automatically navigate to the script's root folder
# ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR" || { echo "❌ ERROR: Failed to navigate to the CogniBridge directory."; exit 1; }
# ---------------------------------------------------------

# Function to run a command, print status, and handle errors
run_command() {
    local description="$1"
    shift
    local cmd=("$@")

    echo ""
    echo "--------------------------------------------------"
    echo "⏳ STEP: $description"
    echo "💻 RUNNING: ${cmd[*]}"
    echo "--------------------------------------------------"
    
    if "${cmd[@]}"; then
        echo "✅ SUCCESS: $description"
        echo ""
    else
        echo "❌ ERROR: Failed to execute $description."
        echo "Please check the terminal output above for clues."
        exit 1
    fi
}

# Ensure the script is actually in the repo root
if [ ! -f "requirements_main.txt" ]; then
    echo "❌ ERROR: Could not find 'requirements_main.txt'."
    echo "It looks like install.sh was moved out of the CogniBridge folder."
    exit 1
fi

echo "🚀 Starting CogniBridge Installation Protocol..."
echo ""

mkdir data inference_results

# 1. Create the NLP Brain (cogni39)
run_command "Creating 'cogni39' environment for Text Generation" \
    conda create -n cogni39 python=3.9.11 -c conda-forge -y

# 2. Install requirements into the MindNLP Brain
run_command "Installing main dependencies into 'cogni39'" \
    conda run -n cogni39 pip install -r requirements_main.txt

# 3. Create the Vision Brain (mindocr_env)
run_command "Creating 'mindocr_env' environment for Optical Character Recognition" \
    conda create -n mindocr_env python=3.9.11 -y

# 4. Install requirements into the Vision Brain
run_command "Installing OCR dependencies into 'mindocr_env'" \
    conda run -n mindocr_env pip install -r mindocr/requirements.txt

# 5. Install MindSpore 2.5.0 into the Vision Brain from .whl file
run_command "Installing MindSpore 2.5.0 into 'mindocr_env'" \
    conda run -n mindocr_env pip install mindspore_installation_package/mindspore-2.5.0-cp39*.whl

echo ""
echo "=================================================="
echo "🎉 INSTALLATION COMPLETE 🎉"
echo "To run the project, activate the main environment using:"
echo ">conda activate cogni39"
echo ">python src/CogniBridge.py"
echo "=================================================="
echo ""