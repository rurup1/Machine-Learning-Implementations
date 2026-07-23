#!/bin/bash
#SBATCH -J FlaxCNN
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=80G
#SBATCH -t 1:00:00
#SBATCH -o Output-%j.out
cd $SLURM_SUBMIT_DIR/FlaxCNN

VENV_DIR="venv"

MODEL_PATH="model.pth"

module load python 

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
else
    echo "Using existing virtual environment '$VENV_DIR'."
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip


# Ensure the model is cloned fresh each training cycle so as to not overtrain the same model over and over:
#   NOTE: Comment the following block out if you wish to maintain your model across multiple training sessions:
if [ -f "$MODEL_PATH" ]; then
    echo "File '$MODEL_PATH' exists. Deleting..."
    rm "$MODEL_PATH" || { echo "Error: Could not delete file."; exit 1; }
    echo "File deleted successfully."
else
    echo "File '$MODEL_PATH' does not exist. Nothing to delete."
fi


while IFS="" read -r p || [ -n "$p" ]
do
    echo "Installing $p ..."
    pip install $p
    echo ""
done < requirements.txt

echo "Starting to train..."
python3 -u train.py
