import json


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w") as f:
        return json.dump(obj, f)
