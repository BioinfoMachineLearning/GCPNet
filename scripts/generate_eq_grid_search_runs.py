# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import itertools
import json


# define constants #
TASK = "eq"  # TODO: Ensure Is Correct Before Each Grid Search!
SCRIPT_DIR = os.path.join("scripts")
SEARCH_SPACE_FILEPATH = os.path.join(SCRIPT_DIR, f"{TASK}_grid_search_runs.json")


def main():
    # TODO: Ensure Is Correct Before Each Grid Search!
    search_space_dict = {
        "gcp_version": [2],
        "key_names": ["NEL NML LR WD DO CHD"],
        "model.model_cfg.num_encoder_layers": [9],
        "model.layer_cfg.mp_cfg.num_message_layers": [8],
        "model.optimizer.lr": [1e-3],
        "model.optimizer.weight_decay": [5e-5],
        "model.model_cfg.dropout": [0.1],
        "model.model_cfg.chi_hidden_dim": [32]
    }

    # gather all combinations of hyperparameters while retaining field names for each chosen hyperparameter
    keys, values = zip(*search_space_dict.items())
    hyperparameter_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # save search space to storage as JSON file
    with open(SEARCH_SPACE_FILEPATH, "w") as f:
        f.write(json.dumps(hyperparameter_dicts))


if __name__ == "__main__":
    main()
