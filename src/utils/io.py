
import numpy as np
import pandas as pd

def load_spectrum(filepath):
    """
    Loads a spectrum file:
      - CSV/TXT: two columns (x,y) or one column (y)
      - NPY: single array
    Returns tuple (x, y) where x can be None if only y is provided.
    """
    if filepath.endswith(".npy"):
        y = np.load(filepath)
        x = np.arange(len(y))
        return x, y

    try:
        data = pd.read_csv(filepath, header=None)
    except Exception:
        data = pd.read_table(filepath, header=None, sep="\s+")

    if data.shape[1] == 1:
        y = data.iloc[:, 0].values
        x = np.arange(len(y))
    else:
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values

    return x, y
