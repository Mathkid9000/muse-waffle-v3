import torch, os, json
from pathlib import Path
from dotenv import load_dotenv

# Paths
load_dotenv('env.env')
STORAGE_DIR = Path(os.getenv('STORAGE_DIR'))
DATA_DIR = STORAGE_DIR / 'data'
CHECKPOINT_DIR = STORAGE_DIR / 'checkpoints'

VOCAB_LIST_PATH = STORAGE_DIR / 'vocab.json'
VOCAB_LIST_PLOT_PATH = STORAGE_DIR / 'vocab_plot.json'
DATASET_PATH = STORAGE_DIR / 'data.npz'

# Params
SEQ_LEN = 1024
BATCH_SIZE = 32
VAL_SPLIT = 0.05