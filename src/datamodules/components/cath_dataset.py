# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

try:
    import rapidjson as json
except:
    import json


class CATHDataset:
    """
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.
    """

    def __init__(self, path: str, splits_path: str):
        with open(splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits["train"], \
                                          dataset_splits["validation"], dataset_splits["test"]

        self.train, self.val, self.test = [], [], []

        with open(path) as f:
            lines = f.readlines()

        for line in lines:
            entry = json.loads(line)
            name = entry["name"]
            coords = entry["coords"]

            entry["coords"] = list(zip(
                coords["N"], coords["CA"], coords["C"], coords["O"]
            ))

            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)