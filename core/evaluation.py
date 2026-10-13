
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_logits(logits_path):
    logits = []
    with open(logits_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            logits.append([float(x) for x in parts])
    return np.array(logits)