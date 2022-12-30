import os

INPUT_DIMS : int = int(os.getenv("INPUT_DIMS", "100"))
OUTPUT_DIMS : int = int(os.getenv("OUTPUT_DIMS", "1"))
WIDTH : int = int(os.getenv("WIDTH", "256"))
DEPTH : int = int(os.getenv("DEPTH", "32"))
BATCH_SIZE : int = int(os.getenv("BATCH_SIZE", "100"))
NUM_BATCHES : int = int(os.getenv("NUM_BATCHES", "100"))
MIN_EPOCHS : int = int(os.getenv("MIN_EPOCHS", "25"))
MAX_EPOCHS : int = int(os.getenv("MAX_EPOCHS", "50"))
