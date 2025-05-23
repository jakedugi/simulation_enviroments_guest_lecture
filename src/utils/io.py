import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def colour_text(text, colour="green"):
    colours = dict(green="\033[92m", end="\033[0m")
    return f"{colours[colour]}{text}{colours['end']}"