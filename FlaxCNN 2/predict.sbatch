#!/bin/bash
#SBATCH -J FlaxCNN
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH -o Output-%j.out
cd $SLURM_SUBMIT_DIR/FlaxCNN

VENV_DIR="venv_predict"

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


while IFS="" read -r p || [ -n "$p" ]
do
    echo "Installing $p ..."
    modified_req=$(echo "$p" | sed 's/jax\[.*\]/jax/g')
    pip install $modified_req
    echo ""
done < requirements.txt

echo "Starting to predict..."
python3 -u predict.py
