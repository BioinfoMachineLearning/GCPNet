#
# From https://github.com/mouthful/ClofNet/blob/master/newtonian
#

saved_dir="data" # the path of output directory

# suffix: the suffix of generated dataset (e.g., small_20body)
# simulation: the mode of force field
#     1. charged: electronstatic system (ES)
#     2. static: gravity + electronstatic force (G+ES)
#     3. dynamic: lorentz + electronstatic force (L+ES)
# n_balls: system size
# num_train: number of training trajectories
# other parameters: see generate_dataset.py

suffix=small
simulation=charged
n_balls=5
python -u generate_dataset.py --num-train 3000 --seed 43 --sufix ${suffix} --saved_dir ${saved_dir} --simulation ${simulation} --n_balls ${n_balls}

suffix=small_20body
simulation=charged
n_balls=20
python -u generate_dataset.py --num-train 3000 --seed 43 --sufix ${suffix} --saved_dir ${saved_dir} --simulation ${simulation} --n_balls ${n_balls}

suffix=static_20body
simulation=static
python -u generate_dataset.py --num-train 3000 --seed 43 --sufix ${suffix} --saved_dir ${saved_dir} --simulation ${simulation} --n_balls ${n_balls}

suffix=dynamic_20body
simulation=dynamic
python -u generate_dataset.py --num-train 3000 --seed 43 --sufix ${suffix} --saved_dir ${saved_dir} --simulation ${simulation} --n_balls ${n_balls}
