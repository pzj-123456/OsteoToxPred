
# AAA Model: Drug-Induced Osteotoxicity Prediction

## Overview

The **AAA** model is a multi-modal deep learning framework developed to predict drug-induced osteotoxicity. By integrating molecular fingerprints, molecular graph features, and attention mechanisms, it effectively captures complex structure-toxicity relationships. The model has achieved high performance with 0.85 accuracy (ACC) and 0.92 AUC.

## Requirements

### Key Dependencies:
- Python 3.9
- torch==2.4.1+cu124
- torch-geometric==2.6.1
- numpy==1.26.4
- scikit-learn==1.0.2
- numpy==1.26.4
- pandas==1.3.5
- rdkit==2024.3.5
- matplotlib
- seaborn



## Installation

1. Clone or download the repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have a CUDA-enabled GPU for efficient computation, especially when training the model.

## Training the Model

To train the model from scratch:

1. Prepare your training data in SMILES format.
2. Use the training command to start the training process:

   ```bash
   python main.py 
   ```

  
