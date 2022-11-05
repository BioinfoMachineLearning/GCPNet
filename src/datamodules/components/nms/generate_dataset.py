"""
From https://github.com/mouthful/ClofNet/blob/master/newtonian
"""

import time
import numpy as np
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import os

from src.datamodules.components.nms.synthetic_sim import ChargedParticlesSim, DynamicSim, FixCharge, GravitySim, SpringSim

"""
nbody: python -u generate_dataset.py  --num-train 50000 --sample-freq 500 2>&1 | tee log_generating_100000.log &

nbody_small: python -u generate_dataset.py --num-train 10000 --seed 43 --sufix small 2>&1 | tee log_generating_10000_small.log &
"""

parser = argparse.ArgumentParser()
parser.add_argument("--simulation", type=str, default="charged",
                    help="What simulation to generate.")
parser.add_argument("--num-train", type=int, default=10000,
                    help="Number of training simulations to generate.")
parser.add_argument("--num-valid", type=int, default=2000,
                    help="Number of validation simulations to generate.")
parser.add_argument("--num-test", type=int, default=2000,
                    help="Number of test simulations to generate.")
parser.add_argument("--length", type=int, default=5000,
                    help="Length of trajectory.")
parser.add_argument("--length_test", type=int, default=5000,
                    help="Length of test set trajectory.")
parser.add_argument("--sample-freq", type=int, default=100,
                    help="How often to sample the trajectory.")
parser.add_argument("--n_balls", type=int, default=5,
                    help="Number of balls in the simulation.")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
parser.add_argument("--initial_vel", type=int, default=1,
                    help="consider initial velocity")
parser.add_argument("--sufix", type=str, default="",
                    help="add a sufix to the name")
parser.add_argument("--saved_dir", type=str, default="",
                    help="add a directory to save")

args = parser.parse_args()
print(args)

initial_vel_norm = 0.5
if not args.initial_vel:
    initial_vel_norm = 1e-16

if args.simulation == "springs":
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = "_springs"
elif args.simulation == "charged":
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm)
    suffix = "_charged"
elif args.simulation == "static":
    sim = GravitySim(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm)
    suffix = "_static"
elif args.simulation == "dynamic":
    sim = DynamicSim(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm)
    suffix = "_dynamic"
elif args.simulation == "fixcharge":
    sim = FixCharge(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm)
    suffix = "_fixcharge"
else:
    raise ValueError("Simulation {} not implemented".format(args.simulation))

suffix += str(args.n_balls) + "_initvel%d" % args.initial_vel + args.sufix
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq, multiprocess=False, num_workers=10):
    loc_all = list()
    vel_all = list()
    edges_all = list()
    charges_all = list()
    t = time.time()

    def collect(s):
        loc_all.append(s[0])
        vel_all.append(s[1])
        edges_all.append(s[2])
        charges_all.append(s[3])
        if len(loc_all) % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(len(loc_all), time.time() - t))

    if multiprocess:
        pool = Pool(num_workers)
        for i in range(num_sims):
            pool.apply_async(sim.sample_trajectory, (np.random.choice(
                list(range(1000000))), length, sample_freq), callback=collect)
        pool.close()
        pool.join()
    else:
        for i in tqdm(range(num_sims)):
            res = sim.sample_trajectory(np.random.choice(list(range(1000000))), T=length, sample_freq=sample_freq)
            collect(res)

    charges_all = np.stack(charges_all)
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all, charges_all


if __name__ == "__main__":
    multiprocess = True
    args.saved_dir = os.path.join(args.saved_dir, args.sufix)
    os.makedirs(args.saved_dir, exist_ok=True)
    if not args.saved_dir[-1] == "/":
        args.saved_dir = args.saved_dir + "/"

    print("Generating {} validation simulations".format(args.num_valid))
    loc_valid, vel_valid, edges_valid, charges_valid = generate_dataset(args.num_valid,
                                                                        args.length,
                                                                        args.sample_freq,
                                                                        multiprocess=multiprocess)
    np.save(args.saved_dir + "loc_valid" + suffix + ".npy", loc_valid)
    np.save(args.saved_dir + "vel_valid" + suffix + ".npy", vel_valid)
    np.save(args.saved_dir + "edges_valid" + suffix + ".npy", edges_valid)
    np.save(args.saved_dir + "charges_valid" + suffix + ".npy", charges_valid)

    print("Generating {} test simulations".format(args.num_test))
    loc_test, vel_test, edges_test, charges_test = generate_dataset(args.num_test,
                                                                    args.length_test,
                                                                    args.sample_freq,
                                                                    multiprocess=multiprocess)
    np.save(args.saved_dir + "loc_test" + suffix + ".npy", loc_test)
    np.save(args.saved_dir + "vel_test" + suffix + ".npy", vel_test)
    np.save(args.saved_dir + "edges_test" + suffix + ".npy", edges_test)
    np.save(args.saved_dir + "charges_test" + suffix + ".npy", charges_test)

    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, edges_train, charges_train = generate_dataset(args.num_train,
                                                                        args.length,
                                                                        args.sample_freq,
                                                                        multiprocess=multiprocess)

    np.save(args.saved_dir + "loc_train" + suffix + ".npy", loc_train)
    np.save(args.saved_dir + "vel_train" + suffix + ".npy", vel_train)
    np.save(args.saved_dir + "edges_train" + suffix + ".npy", edges_train)
    np.save(args.saved_dir + "charges_train" + suffix + ".npy", charges_train)
