import os
import pickle
from typing import List


def extend_and_save_arr(arr: List, path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            arr.extend(pickle.load(f))
    pickle.dump(arr, open(path, "wb"))
