# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import json
import shutil
import wandb
from datetime import datetime
from typing import Any, Dict, List, Tuple

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


# define constants #
HIGH_MEMORY = False  # whether to use high-memory (HM) mode
HALT_FILE_EXTENSION = "done"  # TODO: Update `src.models.HALT_FILE_EXTENSION` As Well Upon Making Changes Here!
TIMESTAMP = datetime.now().strftime("%m%d%Y_%H_%M")

# choose a base experiment to run
TASK = "lba"  # TODO: Ensure Is Correct Before Each Grid Search!
MODEL_NAME = "GCPNet"  # TODO: Ensure Is Correct Before Each Grid Search!
EXPERIMENT = f"gcpnet_{TASK}_grid_search"  # TODO: Ensure Is Correct Before Each Grid Search!
TEMPLATE_RUN_NAME = f"{TIMESTAMP}_{MODEL_NAME}"
TIMEOUT_PERIOD = 1438 if HIGH_MEMORY else 118
FINAL_TEMPLATE_LINE = f"timeout {TIMEOUT_PERIOD}m jsrun -E LD_PRELOAD=/opt/ibm/spectrum_mpi/lib/pami_490/libpami.so -r1 -g6 -a6 -c42 -bpacked:7 python3 src/train.py experiment={EXPERIMENT}"
NUM_RUNS_PER_EXPERIMENT = {"lba": 3, "psr": 1, "cpd": 1,
                           "nms_small": 1, "nms_small_20body": 1, "nms_static": 1, "nms_dynamic": 1,
                           "ar": 1, "eq": 3}

# establish paths
OUTPUT_SCRIPT_FILENAME_PREFIX = "train"
SCRIPT_DIR = os.path.join("scripts")
OUTPUT_SCRIPT_DIR = os.path.join(SCRIPT_DIR, f"{TASK}_grid_search_scripts")
TEMPLATE_SCRIPT_FILEPATH = os.path.join(
    SCRIPT_DIR,
    "grid_search_hm_template_launcher_script.bash"
    if HIGH_MEMORY
    else "grid_search_template_launcher_script.bash"
)

assert TASK in NUM_RUNS_PER_EXPERIMENT.keys(), f"The task {TASK} is not currently available."


@typechecked
def build_arguments_string(
    run: Dict[str, Any],
    items_to_show: List[Tuple[str, Any]],
    final_line: str = FINAL_TEMPLATE_LINE,
    run_name: str = TEMPLATE_RUN_NAME,
    run_id: str = wandb.util.generate_id()
) -> str:
    # substitute latest grid search parameter values into final line of latest script
    final_line += f" logger=wandb logger.wandb.id={run_id} logger.wandb.name={run_name}_GCPv{run['gcp_version']}"

    # install a unique WandB run name
    for s, (key, value) in zip(run["key_names"].split(), items_to_show):
        final_line += f"_{s.strip()}:{value}"

    # establish directory in which to store and find checkpoints and other artifacts for run
    run_dir = os.path.join("logs", "train", "runs", run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    final_line += f" hydra.run.dir={run_dir}"
    final_line += f" ckpt_path={ckpt_path}"  # define name of latest checkpoint for resuming model

    # manually specify version of GCP module to use
    final_line += f" model.module_cfg.selected_GCP._target_=src.models.components.gcpnet.GCP{run['gcp_version']}"

    # add each custom grid search argument
    for key, value in items_to_show:
        final_line += f" {key}={value}"

    return final_line


@typechecked
def build_recursion_string(
    fit_end_indicator_filepath: str,
    cur_script_filename: str
) -> List[str]:
    return [
        "\n\n",
        "# wait until the time limit on the current run has expired\n"
        "wait\n",
        "\n",
        "# recursively launch a new job to resume model training if model training is not already concluded\n",
        f'if ! [[ -f "{fit_end_indicator_filepath}" ]]; then\n',
        f'    cd "$PROJDIR"/scripts/{TASK}_grid_search_scripts/ || exit\n',
        f'    bsub {cur_script_filename}\n',
        f'fi\n'
    ]


def main():
    # load search space from storage as JSON file
    search_space_filepath = os.path.join(SCRIPT_DIR, f"{TASK}_grid_search_runs.json")
    assert os.path.exists(
        search_space_filepath
    ), "JSON file describing grid search runs must be generated beforehand using `generate_grid_search_runs.py`"
    with open(search_space_filepath, "r") as f:
        grid_search_runs = json.load(f)

    # curate each grid search run
    grid_search_runs = [run for run in grid_search_runs for _ in range(NUM_RUNS_PER_EXPERIMENT[TASK])]
    for run_index, run in enumerate(grid_search_runs):
        # distinguish items to show in arguments list
        items_to_show = [(key, value) for (key, value) in run.items() if key not in ["gcp_version", "key_names"]]

        # build list of input arguments as well as scripting logic to recursively resume model training
        run_id = wandb.util.generate_id()
        fit_end_indicator_filepath = os.path.join(OUTPUT_SCRIPT_DIR, f"{run_id}.{HALT_FILE_EXTENSION}")
        cur_script_filename = f"{OUTPUT_SCRIPT_FILENAME_PREFIX}_{run_index}.bash"
        final_lines = [
            build_arguments_string(run, items_to_show, run_id=run_id),
            *build_recursion_string(fit_end_indicator_filepath, cur_script_filename)
        ]

        # write out latest script as copy of template launcher script
        output_script_filepath = os.path.join(
            OUTPUT_SCRIPT_DIR, cur_script_filename
        )
        shutil.copy(TEMPLATE_SCRIPT_FILEPATH, output_script_filepath)
        with open(output_script_filepath, "a") as f:
            f.writelines(final_lines)


if __name__ == "__main__":
    main()
